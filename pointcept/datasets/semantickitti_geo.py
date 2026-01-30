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
                 num_points=120000, 
                 voxel_size=0.05,  
                 test_mode=False,
                 loop=1,
                 labeled_ratio=0.001,
                 hash_seed_1=97734336,
                 hash_seed_2=60478499,
                 hash_seed_3=43328003,
                 stride=1.0,        
                 scan_mode='xyz',   
                 tta_conf=None,
                 ignore_index=255, 
                 rot_z_range=[-1, 1], 
                 tilt_range=[0, 0], 
                 clip_range=[-35.2, -35.2, -4, 35.2, 35.2, 2],
                 scale_range=[0.9, 1.1], 
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
        self.learning_map = self.get_learning_map(ignore_index)

        # Directory to store generated masks
        # e.g. data/semantic_kitti/masks/balanced_0.001
        self.mask_root = os.path.join(self.data_root, 'masks', f'balanced_{self.labeled_ratio}')
        if not self.test_mode:
            os.makedirs(self.mask_root, exist_ok=True)
            if self.logger:
                self.logger.info(f"[{self.split}] Balanced Masks will be saved to: {self.mask_root}")

        if self.split == 'train' and loop == 1:
            pass # Keep user loop setting
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
                    'name': f"{seq}_{frame}" # e.g. 00_000000
                })
        return data_list

    def get_learning_map(self, ignore_index):
        """ Standard SemanticKITTI 19-class mapping """
        map_dict = {
            0 : ignore_index, 1 : ignore_index, 10: 0, 11: 1, 13: 5, 15: 3, 16: 5, 18: 4, 
            20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 
            52: ignore_index, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 18, 99: ignore_index,
            252: 0, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5
        }
        
        max_key = max(map_dict.keys())
        map_array = np.full(max_key + 1, ignore_index, dtype=np.int64)
        for k, v in map_dict.items():
            map_array[k] = v
        return map_array

    def point_clip(self, coord, color, segment):
        """ Clip points outside the specified range """
        min_lim = self.clip_range[:3]
        max_lim = self.clip_range[3:]
        
        mask = np.all((coord >= min_lim) & (coord <= max_lim), axis=1)
        return coord[mask], color[mask], segment[mask]

    def get_class_balanced_mask(self, segment, name):
        """
        Generates or loads a class-balanced mask for the given frame.
        Guarantees that RARE classes are sampled at least min_points.
        """
        seq_id, frame_id = name.split('_')
        save_dir = os.path.join(self.mask_root, seq_id)
        save_path = os.path.join(save_dir, f"{frame_id}.npy")

        # 1. Load if exists (Fast Path)
        if os.path.exists(save_path):
            return np.load(save_path)
        
        # 2. Generate if missing (Slow Path - only happens once)
        os.makedirs(save_dir, exist_ok=True)
        
        total_points = segment.shape[0]
        final_mask = np.zeros(total_points, dtype=bool)
        
        # Get classes present in this frame
        present_classes = np.unique(segment)
        present_classes = present_classes[present_classes != self.ignore_index]
        
        # Deterministic Random Number Generator based on file name
        # Ensures that if we re-generate, we get the SAME mask.
        seed = int(hash(name) % 1e8)
        rng = np.random.default_rng(seed)

        for cls_id in present_classes:
            cls_indices = np.where(segment == cls_id)[0]
            n_cls = len(cls_indices)
            
            # Sampling Strategy: Ratio * Count, but at least 1 point
            # This ensures even 10-point classes get sampled.
            n_sample = max(int(n_cls * self.labeled_ratio), 1)
            
            # Use choice to pick random points
            selected = rng.choice(cls_indices, n_sample, replace=False)
            final_mask[selected] = True
            
        # Save for next time
        np.save(save_path, final_mask)
        return final_mask

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        try:
            # 1. Load Raw Data
            points = np.fromfile(data_info['velodyne_path'], dtype=np.float32).reshape(-1, 4)
            coord = points[:, :3]
            intensity = points[:, 3].reshape(-1, 1)

            # 2. Load Label
            if not self.test_mode and os.path.exists(data_info['label_path']):
                label_raw = np.fromfile(data_info['label_path'], dtype=np.uint32).reshape(-1)
                sem_label = label_raw & 0xFFFF 
                
                # Map raw labels to learning labels
                valid_mask = sem_label < len(self.learning_map)
                segment = np.full_like(sem_label, self.ignore_index, dtype=np.int64)
                segment[valid_mask] = self.learning_map[sem_label[valid_mask]]
            else:
                segment = np.ones(coord.shape[0], dtype=np.int64) * self.ignore_index

            # 3. [CRITICAL STEP] Apply Sparse Mask (Class Balanced) BEFORE Clipping
            # This logic mimics "loading a sparse dataset"
            if not self.test_mode:
                # Generate or Load the persistent mask
                balanced_mask = self.get_class_balanced_mask(segment, data_info['name'])
                
                # Apply mask: Points NOT in mask become IGNORE_INDEX
                # This ensures we only supervise on the selected balanced points
                segment[~balanced_mask] = self.ignore_index

            # 4. Point Clip (Apply to coords, intensity AND the already-sparsified segment)
            coord, intensity, segment = self.point_clip(coord, intensity, segment)

            # 5. Build "RGB" from Intensity for Augmentation Compatibility
            color = np.concatenate([intensity, intensity, intensity], axis=1)

        except Exception as e:
            if self.logger: self.logger.error(f"Error loading {data_info['velodyne_path']}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        if not self.test_mode:
            # 6. KNN Sampling (Architecture Requirement)
            indices = self.get_knn_indices(coord, center=None, num_points=self.num_points)
            coord_c, color_c, segment_c = coord[indices], color[indices], segment[indices]
            
            # 7. Apply Full Augmentation (via Mixin)
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
            # Test Mode: Sliding Window
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
        """
        coord_t = torch.from_numpy(coord).float()
        color_t = torch.from_numpy(color).float()
        target_t = torch.from_numpy(segment).long()

        ptv3_coord = coord_t - coord_t.min(0)[0]
        # Normalize Intensity? 
        # Since we applied AutoContrast in augmentation, it's already locally normalized.
        ptv3_color = color_t 
        
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