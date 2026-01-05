import os
import glob
import numpy as np
import torch
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
                 # 🟢 回归纯随机策略，但保留参数接口
                 hash_seed_1=73856093,
                 hash_seed_2=19349663,
                 hash_seed_3=83492791): 
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.test_mode = test_mode
        self.logger = get_root_logger()
        self.labeled_ratio = labeled_ratio
        
        # 变成 int 防止溢出
        self.h1_k = int(hash_seed_1)
        self.h2_k = int(hash_seed_2)
        self.h3_k = int(hash_seed_3)

        if self.split == 'train' and loop == 1:
            if self.logger is not None:
                self.logger.warning("⚠️ [Dataset] 'loop' arg appears to be 1 for training. Forcing override to 30.")
            self.loop = 30
        else:
            self.loop = loop
        
        self.data_list = self.get_file_list()
        
        if len(self.data_list) > 0:
            self.data_list = self.data_list * self.loop
            if self.logger is not None:
                self.logger.info(f"[{self.split}] Dataset loaded. Ratio: {self.labeled_ratio*100}%. Loop: {self.loop}. Total: {len(self.data_list)}")
                self.logger.info(f"[{self.split}] 🔙 Strategy: Pure Coordinate Hashing (Back to Basics).")
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
                        if self.split == 'val': continue
                    data_list.append(dirpath)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        room_dir = self.data_list[idx]
        try:
            # 1. 这里的 coord 是完整的房间数据
            coord = np.load(os.path.join(room_dir, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(room_dir, "color.npy")).astype(np.float32)
            segment = np.load(os.path.join(room_dir, "segment.npy")).astype(np.int64).reshape(-1)
        except:
            return self.__getitem__(np.random.randint(0, len(self)))

        # ==========================================================
        # 🟢 [Optimized Step 1] 几何哈希采样 (在 Crop 前计算，确保全场景点标签固定)
        # ==========================================================
        if self.split == 'train':
            # 基于原始完整坐标计算 Hash，确保无论怎么 Crop，同一个坐标点标签状态永恒不变
            h1 = np.abs(coord[:, 0] * self.h1_k).astype(np.int64)
            h2 = np.abs(coord[:, 1] * self.h2_k).astype(np.int64)
            h3 = np.abs(coord[:, 2] * self.h3_k).astype(np.int64)
            seed_hash = h1 ^ h2 ^ h3
            threshold = int(self.labeled_ratio * 100000)
            label_mask = (seed_hash % 100000) < threshold
            
            # 将无标签点设为 ignore_index
            segment[~label_mask] = 255

        # 2. 裁剪 (此时 segment 已经包含了固定的弱监督标签)
        coord, color, segment = self.crop_fixed_size(coord, color, segment)

        # ==========================================================
        # 🟢 [Optimized Step 2] 数据增强 (温和版)
        # ==========================================================
        if self.split == 'train':
            # Rotate
            angle = np.random.uniform(0, 2 * np.pi)
            cosval, sinval = np.cos(angle), np.sin(angle)
            R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]], dtype=np.float32)
            coord = np.dot(coord, R.T)
            # Scale
            scale = np.random.uniform(0.9, 1.1)
            coord *= scale
            
            # Jitter: 🔴 弱监督下建议默认关闭，用户可根据需求在 README 自行开启
            # sigma, clip = 0.001, 0.005
            # jitter = np.clip(sigma * np.random.randn(*coord.shape), -1 * clip, clip)
            # coord += jitter
            
            # Flip
            if np.random.random() > 0.5: coord[:, 0] = -coord[:, 0]
            if np.random.random() > 0.5: coord[:, 1] = -coord[:, 1]

        # ==========================================================
        # 🟢 [Step 3] 格式化
        # ==========================================================
        coord_t = torch.from_numpy(coord).float()
        color_t = torch.from_numpy(color).float()
        target = torch.from_numpy(segment).long()

        jafar_color = color_t / 255.0
        xyz_min = coord_t.min(0)[0]
        xyz_max = coord_t.max(0)[0]
        xyz_norm = (coord_t - xyz_min) / (xyz_max - xyz_min + 1e-6)
        jafar_feat = torch.cat([jafar_color, xyz_norm], dim=1) 

        ptv3_feat = color_t / 127.5 - 1.0 
        ptv3_coord = coord_t - coord_t.min(0)[0]
        grid_coord = (ptv3_coord / self.voxel_size).int()

        return dict(
            coord=ptv3_coord, 
            grid_coord=grid_coord,
            ptv3_feat=ptv3_feat,   
            segment=target,
            jafar_coord=coord_t,
            jafar_feat=jafar_feat
        )

    def crop_fixed_size(self, coord, color, label):
        N = coord.shape[0]
        target_N = self.num_points
        if N < target_N:
            indices = np.arange(N)
            pad_indices = np.random.choice(N, target_N - N, replace=True)
            indices = np.concatenate([indices, pad_indices])
        else:
            center_idx = np.random.choice(N)
            center = coord[center_idx]
            dist = np.sum((coord - center)**2, axis=1)
            indices = np.argpartition(dist, target_N)[:target_N]
        np.random.shuffle(indices)
        return coord[indices], color[indices], label[indices]