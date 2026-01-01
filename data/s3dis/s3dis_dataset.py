import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from tqdm import tqdm

class S3DISDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.root = "data/s3dis/stanford_indoor3d"  
        self.label_ratio = cfg['dataset'].get('label_ratio', 0.001)
        self.files = self._get_files()
        self.data_list = []
        print(f"[{split.upper()}] Loading {len(self.files)} rooms...")
        for f in tqdm(self.files):
            self.data_list.append(np.load(f))

    def _get_files(self):
        all_files = sorted(glob.glob(os.path.join(self.root, "*.npy")))
        if self.split == 'train': return [f for f in all_files if 'Area_5' not in f]
        elif self.split == 'val': return [f for f in all_files if 'Area_5' in f]
        else: return [f for f in all_files if 'Area_5' in f]

    def __len__(self):
        if self.split == 'train': return 4000 
        return len(self.files) * 20 

    def __getitem__(self, idx):
        room_idx = idx % len(self.data_list)
        data = self.data_list[room_idx]
        N = data.shape[0]
        points = data[:, :3]
        colors = data[:, 3:6]
        labels = data[:, 6]
        
        # Block Cropping
        while True:
            center = points[np.random.choice(N)]
            block_min = center - 1.0; block_max = center + 1.0
            mask = np.where((points[:,0] >= block_min[0]) & (points[:,0] < block_max[0]) &
                            (points[:,1] >= block_min[1]) & (points[:,1] < block_max[1]))[0]
            if len(mask) > 1024:
                if len(mask) > 4096: mask = np.random.choice(mask, 4096, replace=False)
                break
        
        xyz = points[mask]
        rgb = colors[mask]
        lbl = labels[mask]
        xyz_norm = xyz - xyz.min(0)
        
        # --- Universal Coordinate Hashing for Seeds ---
        # This ensures CONSISTENCY across Train/Val/Inference
        # Hash = abs(x*P1 ^ y*P2 ^ z*P3)
        seed_hash = (np.abs(xyz[:, 0] * 73856093) ^ 
                     np.abs(xyz[:, 1] * 19349663) ^ 
                     np.abs(xyz[:, 2] * 83492791)).astype(np.int64)
        
        # Threshold check
        threshold = int(self.label_ratio * 100000)
        is_seed = (seed_hash % 100000) < threshold
        
        seed_mask = torch.from_numpy(is_seed.astype(bool))
        
        # Prepare Tensors
        xyz_t = torch.from_numpy(xyz_norm).float()
        sft_t = torch.from_numpy(xyz).float() 
        rgb_t = torch.from_numpy(rgb / 255.0).float()
        lbl_t = torch.from_numpy(lbl).long()
        
        return xyz_t, sft_t, rgb_t, lbl_t, seed_mask