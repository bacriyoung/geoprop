import os
import glob
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .geo_utils import GeoDatasetMixin  # Import mixin

@DATASETS.register_module()
class ScanNetGeoDataset(Dataset, GeoDatasetMixin):
    def __init__(self,
                 split='train',
                 data_root='data/scannet',
                 transform=None,
                 num_points=100000,
                 voxel_size=0.02,
                 test_mode=False,
                 loop=1,
                 labeled_ratio=0.001,
                 hash_seed_1=27798836,
                 hash_seed_2=38957831,
                 hash_seed_3=26839085,
                 stride=0.5,
                 scan_mode='xy',
                 tta_conf=None,
                 ignore_index=255,
                 # Config interface for augmentation parameters
                 rot_z_range=[-1, 1],
                 tilt_range=[-1/64, 1/64],
                 scale_range=[0.9, 1.1],
                 jitter_sigma=0.005,
                 jitter_clip=0.02,
                 color_drop_prob=0.2,
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
        
        # Save augmentation config
        self.rot_z_range = rot_z_range
        self.tilt_range = tilt_range
        self.scale_range = scale_range
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.color_drop_prob = color_drop_prob

        self.tta_conf = tta_conf if tta_conf is not None else dict(enable=False)
        if self.test_mode and self.tta_conf.get('enable'):
            if self.logger:
                self.logger.info(f"[{self.split}] TTA Enabled with strategy: {self.tta_conf}")

        self.h1_k = int(hash_seed_1)
        self.h2_k = int(hash_seed_2)
        self.h3_k = int(hash_seed_3)

        if self.split == 'train' and loop == 1:
            if self.logger is not None:
                self.logger.warning("⚠️ [Dataset] 'loop' arg appears to be 1 for training. Forcing override to 8.")
            self.loop = 8
        else:
            self.loop = loop

        self.raw_scene_list = self.get_file_list()
        
        self.data_list = []
        if len(self.raw_scene_list) > 0:
            if not self.test_mode:
                self.data_list = self.raw_scene_list * self.loop
            else:
                self.data_list = self.raw_scene_list
                
            if self.logger is not None:
                self.logger.info(f"[{self.split}] Dataset loaded. Total samples: {len(self.data_list)}")
        else:
            print(f"❌ [Dataset] No files found in {self.data_root} for split {self.split}")

    def get_file_list(self):
        """
        Get the list of scene folders based on the split.
        Pointcept ScanNet structure: data_root/split/scene_id
        """
        if isinstance(self.data_root, str): self.data_root = [self.data_root]
        data_list = []
        
        for root in self.data_root:
            if not os.path.isabs(root): root = os.path.abspath(root)
            split_path = os.path.join(root, self.split)
            if not os.path.exists(split_path): continue

            for scene_id in os.listdir(split_path):
                scene_dir = os.path.join(split_path, scene_id)
                if not os.path.isdir(scene_dir): continue
                if "coord.npy" in os.listdir(scene_dir):
                    data_list.append(scene_dir)
                    
        return sorted(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        scene_dir = self.data_list[idx]
        try:
            coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(scene_dir, "color.npy")).astype(np.float32)
            
            if os.path.exists(os.path.join(scene_dir, "segment20.npy")):
                segment = np.load(os.path.join(scene_dir, "segment20.npy")).astype(np.int64).reshape(-1)
            elif os.path.exists(os.path.join(scene_dir, "segment.npy")):
                segment = np.load(os.path.join(scene_dir, "segment.npy")).astype(np.int64).reshape(-1)
            else:
                segment = np.ones(coord.shape[0], dtype=np.int64) * self.ignore_index
            segment[segment == -1] = self.ignore_index

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Error loading {scene_dir}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        if not self.test_mode:
            # Hash Masking
            if self.labeled_ratio < 1.0:
                h1 = np.abs(coord[:, 0] * self.h1_k).astype(np.int64)
                h2 = np.abs(coord[:, 1] * self.h2_k).astype(np.int64)
                h3 = np.abs(coord[:, 2] * self.h3_k).astype(np.int64)
                seed_hash = h1 ^ h2 ^ h3
                threshold = int(self.labeled_ratio * 100000)
                label_mask = (seed_hash % 100000) < threshold
                segment[~label_mask] = self.ignore_index

            # KNN Selection
            indices = self.get_knn_indices(coord, center=None, num_points=self.num_points) 
            coord_c, color_c, segment_c = coord[indices], color[indices], segment[indices]
            
            # Apply Training Augmentation (from Mixin)
            coord_c, color_c = self.apply_training_augmentation(
                coord_c, color_c, 
                rot_z_range=self.rot_z_range, 
                tilt_range=self.tilt_range
                scale_range=self.scale_range,
                jitter_sigma=self.jitter_sigma,
                jitter_clip=self.jitter_clip,
                color_drop_prob=self.color_drop_prob
            )

            return self.prepare_input_dict(coord_c, color_c, segment_c, indices)
        else:
            # Apply Sliding Window & TTA (from Mixin)
            fragment_list, uncovered_count = self.get_sliding_window_fragments(
                coord, color, segment, 
                num_points=self.num_points, 
                stride=self.stride, 
                scan_mode=self.scan_mode, 
                tta_conf=self.tta_conf
            )

            if uncovered_count > 0:
                if self.logger is not None:
                    log_func = self.logger.warning if self.scan_mode == 'xyz' else self.logger.info
                    log_func(
                        f"[{self.scan_mode.upper()} Scan] {uncovered_count} points "
                        f"({uncovered_count/len(coord):.2%}) were NOT covered by sliding windows in {os.path.basename(scene_dir)}!"
                    )

            return dict(
                name=os.path.basename(scene_dir),
                fragment_list=fragment_list, 
                segment=segment 
            )

    def prepare_input_dict(self, coord, color, segment, indices, is_test_fragment=False):
        coord_t = torch.from_numpy(coord).float()
        color_t = torch.from_numpy(color).float()
        target_t = torch.from_numpy(segment).long()

        ptv3_coord = coord_t - coord_t.min(0)[0]
        ptv3_color = color_t / 255.0
        ptv3_feat = torch.cat([ptv3_coord, ptv3_color], dim=1)
        
        grid_coord = (ptv3_coord / self.voxel_size).int()

        iso_coord = ptv3_coord.clone()
        
        jafar_color = color_t / 255.0
        jafar_feat = torch.cat([jafar_color, iso_coord], dim=1) 

        input_dict = dict(
            coord=ptv3_coord, 
            grid_coord=grid_coord,
            ptv3_feat=ptv3_feat,     
            jafar_coord=coord_t,
            jafar_feat=jafar_feat,
            iso_coord=iso_coord, 
            index=torch.from_numpy(indices).long(), 
            offset=torch.tensor([coord_t.shape[0]], dtype=torch.int32) 
        )
        
        if not is_test_fragment:
            input_dict['segment'] = target_t
            
        return input_dict