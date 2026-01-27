import os
import glob
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS

@DATASETS.register_module()
class ScanNetGeoDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data/scannet',
                 transform=None,
                 num_points=100000,  # Adjusted default for ScanNet (approx. crop size)
                 voxel_size=0.02,
                 test_mode=False,
                 loop=1,
                 labeled_ratio=1.0,  # Default 1.0 for fully supervised ScanNet
                 hash_seed_1=97734336,
                 hash_seed_2=60478499,
                 hash_seed_3=43328003,
                 stride=0.5,
                 scan_mode='xyz',
                 tta_conf=None,
                 ignore_index=-1,    # ScanNet usually uses -1 for ignored classes
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

        self.tta_conf = tta_conf if tta_conf is not None else dict(enable=False)
        if self.test_mode and self.tta_conf.get('enable'):
            if self.logger:
                self.logger.info(f"[{self.split}] TTA Enabled with strategy: {self.tta_conf}")

        self.h1_k = int(hash_seed_1)
        self.h2_k = int(hash_seed_2)
        self.h3_k = int(hash_seed_3)

        if self.split == 'train' and loop == 1:
            if self.logger is not None:
                self.logger.warning("⚠️ [Dataset] 'loop' arg appears to be 1 for training. Forcing override to 30.")
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
            
            # Target path: e.g., data/scannet/train
            split_path = os.path.join(root, self.split)
            
            if not os.path.exists(split_path):
                continue

            # Iterate over scene folders (e.g., scene0001_00)
            for scene_id in os.listdir(split_path):
                scene_dir = os.path.join(split_path, scene_id)
                if not os.path.isdir(scene_dir):
                    continue
                
                # Ensure coordinate file exists
                if "coord.npy" in os.listdir(scene_dir):
                    data_list.append(scene_dir)
                    
        return sorted(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        scene_dir = self.data_list[idx]
        try:
            # Load basic assets
            coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(scene_dir, "color.npy")).astype(np.float32)
            
            # Load labels: Prioritize segment20.npy (20 classes)
            if os.path.exists(os.path.join(scene_dir, "segment20.npy")):
                segment = np.load(os.path.join(scene_dir, "segment20.npy")).astype(np.int64).reshape(-1)
            elif os.path.exists(os.path.join(scene_dir, "segment.npy")):
                # Fallback if only 'segment.npy' exists
                segment = np.load(os.path.join(scene_dir, "segment.npy")).astype(np.int64).reshape(-1)
            else:
                # If no labels exist (e.g., test set), fill with ignore_index
                segment = np.ones(coord.shape[0], dtype=np.int64) * self.ignore_index

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Error loading {scene_dir}: {e}")
            # Randomly retry another sample on failure
            return self.__getitem__(np.random.randint(0, len(self)))

        if not self.test_mode:
            # ------------------------------------------------------------------
            # Training Mode: Random Sampling & Augmentation
            # ------------------------------------------------------------------
            
            # 1. Masking logic for semi-supervised setting (if labeled_ratio < 1)
            if self.labeled_ratio < 1.0:
                h1 = np.abs(coord[:, 0] * self.h1_k).astype(np.int64)
                h2 = np.abs(coord[:, 1] * self.h2_k).astype(np.int64)
                h3 = np.abs(coord[:, 2] * self.h3_k).astype(np.int64)
                seed_hash = h1 ^ h2 ^ h3
                threshold = int(self.labeled_ratio * 100000)
                label_mask = (seed_hash % 100000) < threshold
                segment[~label_mask] = self.ignore_index

            # 2. KNN Crop (Select a chunk of points for training)
            indices = self.get_knn_indices(coord, center=None) 
            coord_c, color_c, segment_c = coord[indices], color[indices], segment[indices]
            
            # 3. Hardcoded Augmentation (Rotation, Scaling, Flipping, Color Jitter)
            angle = np.random.uniform(0, 2 * np.pi)
            cosval, sinval = np.cos(angle), np.sin(angle)
            R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]], dtype=np.float32)
            coord_c = np.dot(coord_c, R.T)
            
            scale = np.random.uniform(0.9, 1.1)
            coord_c *= scale
            
            if np.random.random() > 0.5: coord_c[:, 0] = -coord_c[:, 0]
            if np.random.random() > 0.5: coord_c[:, 1] = -coord_c[:, 1]
            
            # Color jittering
            if np.random.random() < 0.5:
                noise = np.random.randn(1).astype(np.float32)
                color_c = color_c * (1 + 0.1 * noise) + 0.1 * np.random.randn(1).astype(np.float32)
        
            # Randomly drop color information
            if np.random.random() < 0.2:
                color_c[:] = 0.0

            return self.prepare_input_dict(coord_c, color_c, segment_c, indices)
        else:
            # ------------------------------------------------------------------
            # Test/Val Mode: Sliding Window
            # ------------------------------------------------------------------
            fragment_list = []
            coord_min = np.min(coord, axis=0)
            coord_max = np.max(coord, axis=0)

            visited_mask = np.zeros(coord.shape[0], dtype=bool)
            
            stride_x, stride_y = self.stride, self.stride
            
            grid_x = np.arange(coord_min[0], coord_max[0] + stride_x, stride_x)
            grid_y = np.arange(coord_min[1], coord_max[1] + stride_y, stride_y)
            
            # Handle Z-axis scanning
            if self.scan_mode == 'xyz':
                stride_z = self.stride
                grid_z = np.arange(coord_min[2], coord_max[2] + stride_z, stride_z)
            else:
                z_center = (coord_min[2] + coord_max[2]) / 2.0
                grid_z = [z_center] 
            
            # Test Time Augmentation (TTA)
            transforms_to_apply = [dict(scale=1.0, flip_x=False, flip_y=False, rot_z=0)]
            
            if self.tta_conf.get('enable'):
                for s in self.tta_conf.get('scale_list', []):
                    transforms_to_apply.append(dict(scale=s, flip_x=False, flip_y=False, rot_z=0))
                
                if self.tta_conf.get('flip_x'):
                    transforms_to_apply.append(dict(scale=1.0, flip_x=True, flip_y=False, rot_z=0))
                if self.tta_conf.get('flip_y'):
                    transforms_to_apply.append(dict(scale=1.0, flip_x=False, flip_y=True, rot_z=0))
                
                if self.tta_conf.get('rot_z'):
                    for k in [1, 2, 3]:
                        transforms_to_apply.append(dict(scale=1.0, flip_x=False, flip_y=False, rot_z=k))

            # Sliding Window Loop
            for x in grid_x:
                for y in grid_y:
                    for z in grid_z:
                        center = np.array([x, y, z])
                        
                        # Find nearest neighbors to the window center
                        indices = self.get_knn_indices(coord, center=center)

                        visited_mask[indices] = True
                        
                        coord_chunk = coord[indices]
                        color_chunk = color[indices]
                        segment_chunk = segment[indices]

                        # Apply TTA or basic transform
                        for t_cfg in transforms_to_apply:
                            coord_aug = coord_chunk.copy()
                            
                            rot_k = t_cfg.get('rot_z', 0) 
                            if rot_k > 0:
                                angle = rot_k * np.pi / 2
                                c, s = np.cos(angle), np.sin(angle)
                                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                                coord_aug = coord_aug @ R.T
                            
                            scale_val = t_cfg.get('scale', 1.0) 
                            if scale_val != 1.0:
                                coord_aug *= scale_val
                                
                            if t_cfg.get('flip_x', False): 
                                coord_aug[:, 0] = -coord_aug[:, 0]
                            if t_cfg.get('flip_y', False): 
                                coord_aug[:, 1] = -coord_aug[:, 1]
                        
                            chunk_dict = self.prepare_input_dict(
                                coord_aug,      
                                color_chunk, 
                                segment_chunk, 
                                indices,        
                                is_test_fragment=True
                            )
                            fragment_list.append(chunk_dict)

            # Warning if sliding window missed some points
            uncovered_count = np.sum(~visited_mask)
            if uncovered_count > 0:
                if self.logger is not None:
                    log_func = self.logger.warning if self.scan_mode == 'xyz' else self.logger.info
                    log_func(
                        f"[{self.scan_mode.upper()} Scan] {uncovered_count} points "
                        f"({uncovered_count/len(coord):.2%}) were NOT covered by sliding windows in {os.path.basename(scene_dir)}! "
                        f"Consider decreasing 'stride' or switching scan mode."
                    )

            return dict(
                name=os.path.basename(scene_dir),
                fragment_list=fragment_list, 
                segment=segment 
            )

    def prepare_input_dict(self, coord, color, segment, indices, is_test_fragment=False):
        """
        Critical function for GeoProp/GeoPTV3:
        Generates standard inputs (ptv3_feat) AND custom inputs (iso_coord, jafar_feat).
        """
        coord_t = torch.from_numpy(coord).float()
        color_t = torch.from_numpy(color).float()
        target_t = torch.from_numpy(segment).long()

        # Shift to local coordinate system (min-offset)
        ptv3_coord = coord_t - coord_t.min(0)[0]
        # Normalize color to [0, 1]
        ptv3_color = color_t / 255.0
        
        # Standard input features for PTV3 Backbone: [local_coord, color]
        ptv3_feat = torch.cat([ptv3_coord, ptv3_color], dim=1)
        
        # Grid coordinates for voxelization (if used)
        grid_coord = (ptv3_coord / self.voxel_size).int()

        # Custom inputs for JAFAR module (GeoProp):
        # 'iso_coord' is used for isometric mapping constraints
        iso_coord = ptv3_coord.clone()
        
        jafar_color = color_t / 255.0
        # 'jafar_feat' combines color and coordinates
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

    def get_knn_indices(self, coord, center=None):
        """
        Selects 'num_points' neighbors around a center point.
        """
        N = coord.shape[0]
        target_N = self.num_points
        
        if center is None:
            # Random center for training
            center_idx = np.random.choice(N)
            center_point = coord[center_idx]
        else:
            # Specific center for sliding window
            center_point = center

        # Calculate squared distance
        dist = np.sum((coord - center_point)**2, axis=1)
        
        if N < target_N:
            # Padding with replacement if not enough points
            base = np.arange(N)
            pad = np.random.choice(N, target_N - N, replace=True)
            indices = np.concatenate([base, pad])
        else:
            # Select nearest k points
            indices = np.argpartition(dist, target_N)[:target_N]
            
        np.random.shuffle(indices)
        return indices