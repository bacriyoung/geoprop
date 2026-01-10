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
class S3DISCoTrainDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data/s3dis',
                 transform=None,
                 num_points=80000,
                 voxel_size=0.02,
                 test_mode=False,
                 loop=1,
                 labeled_ratio=0.001,
                 hash_seed_1=57361723,
                 hash_seed_2=92990218,
                 hash_seed_3=69232043,
                 # Stride for sliding window, smaller means higher overlap/accuracy
                 stride=0.5,
                 scan_mode='xyz',
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
        
        self.h1_k = int(hash_seed_1)
        self.h2_k = int(hash_seed_2)
        self.h3_k = int(hash_seed_3)

        if self.split == 'train' and loop == 1:
            if self.logger is not None:
                self.logger.warning("⚠️ [Dataset] 'loop' arg appears to be 1 for training. Forcing override to 30.")
            self.loop = 30
        else:
            self.loop = loop

        self.raw_room_list = self.get_file_list()
        
        # Build Data List
        self.data_list = []
        if len(self.raw_room_list) > 0:
            if not self.test_mode:
                self.data_list = self.raw_room_list * self.loop
            else:
                self.data_list = self.raw_room_list
                
            if self.logger is not None:
                self.logger.info(f"[{self.split}] Dataset loaded. Total samples: {len(self.data_list)}")
        else:
            print(f"❌ [Dataset] No files found in {self.data_root}")

    def get_file_list(self):
        if isinstance(self.data_root, str): self.data_root = [self.data_root]
        data_list = []
        for root in self.data_root:
            if not os.path.isabs(root): root = os.path.abspath(root)
            if not os.path.exists(root): continue
            for dirpath, _, filenames in os.walk(root):
                if "coord.npy" in filenames:
                    if "Area_5" in dirpath:
                        if self.split == 'train': continue
                    else:
                        if self.split == 'val' or self.split == 'Area_5': continue
                    data_list.append(dirpath)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        room_dir = self.data_list[idx]
        try:
            coord = np.load(os.path.join(room_dir, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(room_dir, "color.npy")).astype(np.float32)
            segment = np.load(os.path.join(room_dir, "segment.npy")).astype(np.int64).reshape(-1)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Error loading {room_dir}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        # ==================================================================
        # Training Mode: Random KNN Crop (Keep original logic)
        # ==================================================================
        if not self.test_mode:
            if self.split == 'train':
                # Semiautomatic labeling logic
                h1 = np.abs(coord[:, 0] * self.h1_k).astype(np.int64)
                h2 = np.abs(coord[:, 1] * self.h2_k).astype(np.int64)
                h3 = np.abs(coord[:, 2] * self.h3_k).astype(np.int64)
                seed_hash = h1 ^ h2 ^ h3
                threshold = int(self.labeled_ratio * 100000)
                label_mask = (seed_hash % 100000) < threshold
                segment[~label_mask] = 255

            indices = self.get_knn_indices(coord, center=None) 
            coord_c, color_c, segment_c = coord[indices], color[indices], segment[indices]
            
            # Simple augmentation for train split
            if self.split == 'train':
                angle = np.random.uniform(0, 2 * np.pi)
                cosval, sinval = np.cos(angle), np.sin(angle)
                R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]], dtype=np.float32)
                coord_c = np.dot(coord_c, R.T)
                scale = np.random.uniform(0.9, 1.1)
                coord_c *= scale
                if np.random.random() > 0.5: coord_c[:, 0] = -coord_c[:, 0]
                if np.random.random() > 0.5: coord_c[:, 1] = -coord_c[:, 1]

            return self.prepare_input_dict(coord_c, color_c, segment_c, indices)

        # ==================================================================
        # Test/Val Mode: Dense Sliding KNN Window (Dual Mode: XY / XYZ)
        # ==================================================================
        else:
            fragment_list = []
            coord_min = np.min(coord, axis=0)
            coord_max = np.max(coord, axis=0)

            # Initialize a mask to track which points have been visited
            visited_mask = np.zeros(coord.shape[0], dtype=bool)
            
            # [Modified] Unified Scanning Logic for 'xy' and 'xyz' modes
            stride_x, stride_y = self.stride, self.stride
            
            # Generate grids for X and Y axes
            grid_x = np.arange(coord_min[0], coord_max[0] + stride_x, stride_x)
            grid_y = np.arange(coord_min[1], coord_max[1] + stride_y, stride_y)
            
            # [Modified] Branching logic for Z-axis generation based on scan_mode
            if self.scan_mode == 'xyz':
                # 'xyz' mode: Full 3D scanning, moves along Z axis as well
                stride_z = self.stride
                grid_z = np.arange(coord_min[2], coord_max[2] + stride_z, stride_z)
            else:
                # 'xy' mode (Default): Fix Z at the room center, scan only XY plane
                # This is faster but might miss points near ceiling/floor if num_points is small
                z_center = (coord_min[2] + coord_max[2]) / 2.0
                grid_z = [z_center] # Wrap in list to make the loop generic

            for x in grid_x:
                for y in grid_y:
                    for z in grid_z:
                        center = np.array([x, y, z])
                        
                        # Core: Get fixed-size KNN indices to match training distribution
                        indices = self.get_knn_indices(coord, center=center)

                        # Update coverage mask
                        visited_mask[indices] = True
                        
                        coord_chunk = coord[indices]
                        color_chunk = color[indices]
                        segment_chunk = segment[indices]
                        
                        # Normalize and wrap into dict
                        # is_test_fragment=True prevents adding 'segment' to individual chunks to save GPU memory
                        chunk_dict = self.prepare_input_dict(
                            coord_chunk, 
                            color_chunk, 
                            segment_chunk, 
                            indices, 
                            is_test_fragment=True
                        )
                        fragment_list.append(chunk_dict)

            # [New] Check for uncovered points after scanning the whole scene
            uncovered_count = np.sum(~visited_mask)
            if uncovered_count > 0:
                if self.logger is not None:
                    # Use different log levels: Warning for XYZ (should be full coverage), Info for XY
                    log_func = self.logger.warning if self.scan_mode == 'xyz' else self.logger.info
                    log_func(
                        f"[{self.scan_mode.upper()} Scan] {uncovered_count} points "
                        f"({uncovered_count/len(coord):.2%}) were NOT covered by sliding windows in {os.path.basename(room_dir)}! "
                        f"Consider decreasing 'stride' or switching scan mode."
                    )

            # Returns scene-level dict, compatible with SemSegTester.test() logic
            return dict(
                name=os.path.basename(room_dir),
                fragment_list=fragment_list, 
                segment=segment # Whole scene ground truth for evaluation
            )

    def prepare_input_dict(self, coord, color, segment, indices, is_test_fragment=False):
        coord_t = torch.from_numpy(coord).float()
        color_t = torch.from_numpy(color).float()
        target_t = torch.from_numpy(segment).long()

        # JAFAR Feature Preparation: Normalized RGB and XYZ
        jafar_color = color_t / 255.0
        xyz_min = coord_t.min(0)[0]
        xyz_max = coord_t.max(0)[0]
        xyz_norm = (coord_t - xyz_min) / (xyz_max - xyz_min + 1e-6)
        jafar_feat = torch.cat([jafar_color, xyz_norm], dim=1) 

        # PTv3 Feature Preparation: RGB in [-1, 1]
        ptv3_feat = color_t / 127.5 - 1.0 
        
        # Consistent Coordinate Normalization: Min-Shift (Aligns with training)
        ptv3_coord = coord_t - coord_t.min(0)[0]
        grid_coord = (ptv3_coord / self.voxel_size).int()

        input_dict = dict(
            coord=ptv3_coord, 
            grid_coord=grid_coord,
            ptv3_feat=ptv3_feat,    
            jafar_coord=coord_t,
            jafar_feat=jafar_feat,
            index=torch.from_numpy(indices).long(), # Global indices for voting
            offset=torch.tensor([coord_t.shape[0]], dtype=torch.int32) # Offset for PTv3/Spconv
        )
        
        # Only add segment if not in fragment mode to avoid redundant memory usage in testing
        if not is_test_fragment:
            input_dict['segment'] = target_t
            
        return input_dict

    def get_knn_indices(self, coord, center=None):
        N = coord.shape[0]
        target_N = self.num_points
        
        if center is None:
            # Random center for training
            center_idx = np.random.choice(N)
            center_point = coord[center_idx]
        else:
            # Specified center for sliding window
            center_point = center

        # Calculate squared Euclidean distance to center
        dist = np.sum((coord - center_point)**2, axis=1)
        
        # Ensure we always return exactly target_N points
        if N < target_N:
            base = np.arange(N)
            pad = np.random.choice(N, target_N - N, replace=True)
            indices = np.concatenate([base, pad])
        else:
            # Partition is much faster than full sort for getting top-K
            indices = np.argpartition(dist, target_N)[:target_N]
            
        # Shuffle to break spatial ordering, important for robust learning/inference
        np.random.shuffle(indices)
        return indices