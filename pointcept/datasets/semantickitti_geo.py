import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .geo_utils import GeoDatasetMixin

@DATASETS.register_module()
class SemanticKITTIGeoDataset(Dataset, GeoDatasetMixin):
    def __init__(self,
                 split='train',
                 data_root='data/semantic_kitti',
                 transform=None,
                 num_points=120000, # Increased to match PTv2 config (120k)
                 voxel_size=0.05,   # PTv2 uses 0.05m
                 test_mode=False,
                 loop=1,
                 labeled_ratio=0.001,
                 hash_seed_1=97734336,
                 hash_seed_2=60478499,
                 hash_seed_3=43328003,
                 stride=1.0,        # Outdoor scenes need larger stride
                 scan_mode='xyz',   # Sliding window mode
                 tta_conf=None,
                 ignore_index=255,  # -1 in config -> mapped to 255 internally usually
                 rot_z_range=[-1, 1], 
                 tilt_range=[0, 0], 
                 clip_range=[-35.2, -35.2, -4, 35.2, 35.2, 2],
                 scale_range=[0.9, 1.1],  # Default match utils
                 jitter_sigma=0.005,
                 jitter_clip=0.02,
                 color_drop_prob=0.2,
                 chromatic_autocontrast_p=0.2,
                 chromatic_translation_p=0.95,
                 chromatic_translation_ratio=0.05,
                 chromatic_jitter_std=0.05,
                 **kwargs):
        
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.test_mode = test_mode
        self.logger = get_root_logger()
        self.labeled_ratio = labeled_ratio
        self.stride = stride
        self.scan_mode = scan_mode
        self.ignore_index = ignore_index
        
        # Augmentation & Preprocessing Config
        self.rot_z_range = rot_z_range
        self.tilt_range = tilt_range 
        self.clip_range = np.array(clip_range)
        self.scale_range = scale_range
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.color_drop_prob = color_drop_prob
        self.chromatic_autocontrast_p = chromatic_autocontrast_p
        self.chromatic_translation_p = chromatic_translation_p
        self.chromatic_translation_ratio = chromatic_translation_ratio
        self.chromatic_jitter_std = chromatic_jitter_std

        self.tta_conf = tta_conf if tta_conf is not None else dict(enable=False)
        if self.test_mode and self.tta_conf.get('enable'):
            if self.logger:
                self.logger.info(f"[{self.split}] TTA Enabled with strategy: {self.tta_conf}")

        self.h1_k = int(hash_seed_1)
        self.h2_k = int(hash_seed_2)
        self.h3_k = int(hash_seed_3)

        # Standard SemanticKITTI Splits
        if split == 'train':
            self.seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif split == 'val':
            self.seq_list = ['08']
        elif split == 'test':
            self.seq_list = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        else:
            raise ValueError(f"Unknown split: {split}")

        # Label Mapping (Raw -> Learning)
        # Using built-in mapping to avoid external yaml dependency issues
        self.learning_map = self.get_learning_map(ignore_index)

        if self.split == 'train' and loop == 1:
            # Outdoor datasets often need fewer loops per epoch due to size, 
            # but setting loop=1 is fine if epoch number is high. 
            # PTv2 config uses 50 epochs. Let's keep it flexible.
            pass 
        self.loop = loop

        self.data_list = self.get_data_list()
        
        if not self.test_mode and len(self.data_list) > 0:
            self.data_list = self.data_list * self.loop
                
        if self.logger is not None:
            self.logger.info(f"[{self.split}] Dataset loaded. Total samples: {len(self.data_list)}")

    def get_data_list(self):
        data_list = []
        for seq in self.seq_list:
            seq_path = os.path.join(self.data_root, 'dataset', 'sequences', seq)
            velodyne_path = os.path.join(seq_path, 'velodyne')
            
            if not os.path.exists(velodyne_path):
                if self.logger: self.logger.warning(f"Sequence {seq} not found at {velodyne_path}")
                continue
                
            frames = sorted([os.path.splitext(f)[0] for f in os.listdir(velodyne_path) if f.endswith('.bin')])
            
            for frame in frames:
                data_list.append({
                    'velodyne_path': os.path.join(velodyne_path, frame + '.bin'),
                    'label_path': os.path.join(seq_path, 'labels', frame + '.label'),
                    'name': f"{seq}_{frame}"
                })
        return data_list

    def get_learning_map(self, ignore_index):
        """ Standard SemanticKITTI 19-class mapping """
        # map_dict: raw_id -> learning_id
        map_dict = {
            0 : ignore_index,     # "unlabeled"
            1 : ignore_index,     # "outlier" mapped to "unlabeled" 
            10: 0,     # "car"
            11: 1,     # "bicycle"
            13: 5,     # "bus" -> "other-vehicle"
            15: 3,     # "motorcycle"
            16: 5,     # "on-rails" -> "other-vehicle"
            18: 4,     # "truck"
            20: 5,     # "other-vehicle"
            30: 6,     # "person"
            31: 7,     # "bicyclist"
            32: 8,     # "motorcyclist"
            40: 9,     # "road"
            44: 10,    # "parking"
            48: 11,    # "sidewalk"
            49: 12,    # "other-ground"
            50: 13,    # "building"
            51: 14,    # "fence"
            52: ignore_index, # "other-structure" -> "unlabeled"
            60: 9,     # "lane-marking" -> "road"
            70: 15,    # "vegetation"
            71: 16,    # "trunk"
            72: 17,    # "terrain"
            80: 18,    # "pole"
            81: 18,    # "traffic-sign"
            99: ignore_index,     # "other-object" -> "unlabeled"
            252: 0,    # "moving-car" -> "car"
            253: 7,    # "moving-bicyclist" -> "bicyclist"
            254: 6,    # "moving-person" -> "person"
            255: 8,    # "moving-motorcyclist" -> "motorcyclist"
            256: 5,    # "moving-on-rails" -> "other-vehicle"
            257: 5,    # "moving-bus" -> "other-vehicle"
            258: 4,    # "moving-truck" -> "truck"
            259: 5     # "moving-other-vehicle" -> "other-vehicle"
        }
        
        max_key = max(map_dict.keys())
        map_array = np.full(max_key + 1, ignore_index, dtype=np.int64)
        for k, v in map_dict.items():
            map_array[k] = v
        return map_array

    def point_clip(self, coord, color, segment):
        """ Clip points outside the specified range (PTv2 strategy) """
        # clip_range: [min_x, min_y, min_z, max_x, max_y, max_z]
        min_lim = self.clip_range[:3]
        max_lim = self.clip_range[3:]
        
        mask = np.all((coord >= min_lim) & (coord <= max_lim), axis=1)
        return coord[mask], color[mask], segment[mask]

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        try:
            # 1. Load Data
            # SemanticKITTI raw: (N, 4) -> x, y, z, intensity
            points = np.fromfile(data_info['velodyne_path'], dtype=np.float32).reshape(-1, 4)
            coord = points[:, :3]
            intensity = points[:, 3].reshape(-1, 1)

            # 2. Load Label
            if not self.test_mode and os.path.exists(data_info['label_path']):
                label = np.fromfile(data_info['label_path'], dtype=np.uint32).reshape(-1)
                sem_label = label & 0xFFFF  # Lower 16 bits
                # Map labels
                valid_mask = sem_label < len(self.learning_map)
                segment = np.full_like(sem_label, self.ignore_index, dtype=np.int64)
                segment[valid_mask] = self.learning_map[sem_label[valid_mask]]
            else:
                segment = np.ones(coord.shape[0], dtype=np.int64) * self.ignore_index

            # [ADAPTATION] PointClip (Important for outdoor outliers)
            coord, intensity, segment = self.point_clip(coord, intensity, segment)

            # [ADAPTATION] Intensity -> RGB-like for Mixin
            # We replicate intensity to 3 channels to use GeoDatasetMixin's augmentations
            # (Translation/Jitter/Contrast on intensity is valid sensor noise simulation)
            color = np.concatenate([intensity, intensity, intensity], axis=1)

        except Exception as e:
            if self.logger: self.logger.error(f"Error loading {data_info['velodyne_path']}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        if not self.test_mode:
            # 1. Hash Masking (Weak Supervision)
            if self.labeled_ratio < 1.0:
                h1 = np.abs(coord[:, 0] * self.h1_k).astype(np.int64)
                h2 = np.abs(coord[:, 1] * self.h2_k).astype(np.int64)
                h3 = np.abs(coord[:, 2] * self.h3_k).astype(np.int64)
                seed_hash = h1 ^ h2 ^ h3
                threshold = int(self.labeled_ratio * 100000)
                label_mask = (seed_hash % 100000) < threshold
                segment[~label_mask] = self.ignore_index

            # 2. Random Sampling (SphereCrop replacement)
            # PTv2 uses SphereCrop(120k). KNN with a random center is similar.
            indices = self.get_knn_indices(coord, center=None, num_points=self.num_points)
            coord_c, color_c, segment_c = coord[indices], color[indices], segment[indices]
            
            # 3. Augmentation (via Mixin)
            # Note: tilt_range is [0,0] by default, so only Rot-Z is applied
            coord_c, color_c = self.apply_training_augmentation(
                coord_c, color_c, 
                rot_z_range=self.rot_z_range, 
                tilt_range=self.tilt_range,
                scale_range=self.scale_range,
                jitter_sigma=self.jitter_sigma,
                jitter_clip=self.jitter_clip,
                color_drop_prob=self.color_drop_prob,
                chromatic_autocontrast_p=self.chromatic_autocontrast_p,
                chromatic_translation_p=self.chromatic_translation_p,
                chromatic_translation_ratio=self.chromatic_translation_ratio,
                chromatic_jitter_std=self.chromatic_jitter_std
            )

            return self.prepare_input_dict(coord_c, color_c, segment_c, indices)
        else:
            # 4. Validation/Test (Sliding Window via Mixin)
            # Note: PTv2 does NOT crop in val/test (whole scene). 
            # If num_points is small, sliding window is needed. 
            # If GPU memory allows, set num_points very large to process whole scene.
            # Here we follow Mixin's sliding window for safety.
            fragment_list, uncovered_count = self.get_sliding_window_fragments(
                coord, color, segment, 
                num_points=self.num_points, 
                stride=self.stride, 
                scan_mode=self.scan_mode, 
                tta_conf=self.tta_conf
            )

            if uncovered_count > 0 and self.logger:
                 self.logger.info(
                    f"[{self.scan_mode.upper()} Scan] {uncovered_count} points "
                    f"({uncovered_count/len(coord):.2%}) uncovered in {data_info['name']}"
                )

            return dict(
                name=data_info['name'],
                fragment_list=fragment_list, 
                segment=segment 
            )

    def prepare_input_dict(self, coord, color, segment, indices, is_test_fragment=False):
        """
        Critical function for GeoProp/GeoPTV3.
        Note: We use the 3-channel intensity 'color' here.
        """
        coord_t = torch.from_numpy(coord).float()
        color_t = torch.from_numpy(color).float()
        target_t = torch.from_numpy(segment).long()

        # Local coordinate system
        ptv3_coord = coord_t - coord_t.min(0)[0]
        
        # PTV3 Input Features: [coord, intensity]
        # PTv2 uses in_channels=4 (coord + strength).
        # We constructed 'color' as (strength, strength, strength). 
        # To align with PTv3 backbone expecting 6 channels (3 coord + 3 color), 
        # we pass our 3-channel intensity. 
        # If backbone expects 4, we should change this, but GeoPTV3 usually expects 6.
        # Assuming GeoPTV3 input dim is flexible or set to 6.
        
        # Normalize Intensity? PTv2 doesn't explicitly normalize in config (it's loaded raw).
        # But 'color_t' here has been augmented (Contrast/Jitter), so it's fine.
        ptv3_color = color_t 
        
        # Standard input features for PTV3 Backbone: [local_coord, color]
        ptv3_feat = torch.cat([ptv3_coord, ptv3_color], dim=1)
        
        grid_coord = (ptv3_coord / self.voxel_size).int()

        # Custom inputs for JAFAR module (GeoProp)
        iso_coord = ptv3_coord.clone()
        jafar_color = color_t
        jafar_feat = torch.cat([jafar_color, iso_coord], dim=1) 

        input_dict = dict(
            coord=ptv3_coord, 
            grid_coord=grid_coord,
            ptv3_feat=ptv3_feat,     
            jafar_coord=coord_t,    # Raw world coordinates
            jafar_feat=jafar_feat,  # Features for JAFAR module
            iso_coord=iso_coord,    # Normalized coordinates
            index=torch.from_numpy(indices).long(), 
            offset=torch.tensor([coord_t.shape[0]], dtype=torch.int32) 
        )
        
        if not is_test_fragment:
            input_dict['segment'] = target_t
            
        return input_dict