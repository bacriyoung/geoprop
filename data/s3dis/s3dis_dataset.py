import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import logging

class S3DISDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        
        # [Fix 1: Remove Redundancy] 
        # Directly use the merged config from main.py
        ds_cfg = cfg['dataset']
        
        self.root = ds_cfg.get('root_dir')
        if not self.root:
            # Fallback check
            raise ValueError("root_dir not found in config. Ensure s3dis.yaml is merged in main.py")
            
        self.num_points = ds_cfg.get('num_points', 24000)
        self.block_size = ds_cfg.get('block_size', 2.0)
        self.label_ratio = ds_cfg.get('label_ratio', 0.001)

        self.files, target_areas = self._get_files(ds_cfg)
        
        if len(self.files) == 0:
            raise RuntimeError(f"[{split.upper()}] No files found in {self.root}!")

        print(f"[{split.upper()}] S3DIS Dataset: Loaded {len(self.files)} rooms from Areas {target_areas}")

        self.data_list = []
        for f in self.files:
            self.data_list.append(np.load(f))

    def _get_files(self, ds_cfg):
        search_pattern = os.path.join(self.root, "**", "*.npy")
        all_files = sorted(glob.glob(search_pattern, recursive=True))
        
        # Access split config (merged)
        split_cfg = ds_cfg.get('split', {})
        if not split_cfg:
             split_cfg = {'train_areas': [1,2,3,4,6], 'val_areas': [5], 'inference_areas': [5]}

        if self.split == 'train': areas = split_cfg['train_areas']
        elif self.split == 'val': areas = split_cfg['val_areas']
        else: areas = split_cfg['inference_areas']
            
        filtered = []
        for f in all_files:
            base = os.path.basename(f)
            for a in areas:
                if f"Area_{a}_" in base:
                    filtered.append(f)
                    break
        return filtered, areas

    def __len__(self):
        return len(self.files) * (5 if self.split == 'train' else 1)

    def __getitem__(self, idx):
        if self.split != 'train':
            np.random.seed(idx)
            
        room_idx = idx % len(self.data_list)
        data = self.data_list[room_idx]
        
        N = data.shape[0]
        # Center Crop (Standard V2)
        center_idx = np.random.choice(N)
        center = data[center_idx, :3]
        
        mask = np.all((data[:, :3] >= center - self.block_size/2) & 
                      (data[:, :3] <= center + self.block_size/2), axis=1)
        block_data = data[mask]
        
        # Resample
        if len(block_data) < self.num_points: 
            choice = np.random.choice(len(block_data), self.num_points, replace=True)
        else: 
            choice = np.random.choice(len(block_data), self.num_points, replace=False)
            
        sample = block_data[choice]
        xyz = sample[:, :3].astype(np.float32)
        rgb = sample[:, 3:6].astype(np.float32)
        lbl = sample[:, 6].astype(int)
        
        if rgb.max() > 1.1: rgb /= 255.0

        # [Fix 2: Critical Logic Fix]
        # Compute Seed Hash BEFORE Augmentation.
        # This ensures the seeds are physically tied to the original geometry.
        # Even if we jitter the points later, 'seed_mask' remains true for the same physical points.
        h1 = np.abs(xyz[:, 0] * 73856093).astype(np.int64)
        h2 = np.abs(xyz[:, 1] * 19349663).astype(np.int64)
        h3 = np.abs(xyz[:, 2] * 83492791).astype(np.int64)
        seed_hash = h1 ^ h2 ^ h3
        threshold = int(self.label_ratio * 100000)
        is_seed = (seed_hash % 100000) < threshold
        seed_mask = torch.from_numpy(is_seed.astype(bool))

        # [Augmentation] Now we can jitter safely
        if self.split == 'train' and self.cfg['train']['augmentation']['enabled']:
            np.random.seed() 
            sigma = self.cfg['train']['augmentation']['jitter_sigma']
            clip = self.cfg['train']['augmentation']['jitter_clip']
            xyz += np.clip(sigma * np.random.randn(*xyz.shape), -clip, clip)
        
        xyz_t = torch.from_numpy(xyz).float()
        rgb_t = torch.from_numpy(rgb).float()
        lbl_t = torch.from_numpy(lbl).long()
        
        # Normalize & Center
        xyz_min = xyz_t.min(0)[0]
        xyz_max = xyz_t.max(0)[0]
        xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min + 1e-6)
        
        sft_feat = torch.cat([rgb_t, xyz_norm], dim=1) 
        xyz_centered = xyz_t - xyz_t.mean(0) 
        
        return xyz_centered, sft_feat, rgb_t, lbl_t, seed_mask