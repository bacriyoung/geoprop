import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import yaml
import logging

class S3DISDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        
        # 1. Load config strict V2.0 way
        config_path = os.path.join('config', 's3dis', 's3dis.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.ds_cfg = yaml.safe_load(f)
            
        ds_info = self.ds_cfg['dataset']
        self.root = ds_info['root_dir']
        
        # [RESTORED] V2.0 Parameters
        self.num_points = ds_info.get('num_points', 24000)
        self.block_size = ds_info.get('block_size', 2.0)
        self.label_ratio = cfg['dataset'].get('label_ratio', ds_info.get('label_ratio', 0.001))

        # 2. Load Files
        self.files, target_areas = self._get_files(ds_info)
        
        if len(self.files) == 0:
            raise RuntimeError(f"[{split.upper()}] No files found in {self.root}!")

        # [RESTORED] Log format
        print(f"[{split.upper()}] S3DIS Dataset: Loaded {len(self.files)} rooms from Areas {target_areas}")

        self.data_list = []
        for f in self.files:
            self.data_list.append(np.load(f))

    def _get_files(self, ds_info):
        search_pattern = os.path.join(self.root, "**", "*.npy")
        all_files = sorted(glob.glob(search_pattern, recursive=True))
        
        split_cfg = ds_info['split']
        if self.split == 'train': areas = split_cfg['train_areas']
        elif self.split == 'val': areas = split_cfg['val_areas']
        else: areas = split_cfg['inference_areas']
            
        filtered = []
        for f in all_files:
            for a in areas:
                if f"Area_{a}_" in os.path.basename(f):
                    filtered.append(f)
                    break
        return filtered, areas

    def __len__(self):
        # [RESTORED] V2.0 Logic: 5x for training
        return len(self.files) * (5 if self.split == 'train' else 1)

    def __getitem__(self, idx):
        # [RESTORED] V2.0 Determinism for val
        if self.split != 'train':
            np.random.seed(idx)
            
        room_idx = idx % len(self.data_list)
        data = self.data_list[room_idx]
        
        # [RESTORED] V2.0 Sampling Logic (Center Crop)
        center_idx = np.random.choice(len(data))
        center = data[center_idx, :3]
        
        mask = np.all((data[:, :3] >= center - self.block_size/2) & 
                      (data[:, :3] <= center + self.block_size/2), axis=1)
        block_data = data[mask]
        
        # [RESTORED] Resampling to num_points (24000)
        if len(block_data) < self.num_points: 
            choice = np.random.choice(len(block_data), self.num_points, replace=True)
        else: 
            choice = np.random.choice(len(block_data), self.num_points, replace=False)
            
        sample = block_data[choice]
        xyz = sample[:, :3].astype(np.float32)
        rgb = sample[:, 3:6].astype(np.float32)
        lbl = sample[:, 6].astype(int)
        
        if rgb.max() > 1.1: rgb /= 255.0

        # [RESTORED] Augmentation
        if self.split == 'train' and self.cfg['train']['augmentation']['enabled']:
            # Reset seed for randomness in train
            np.random.seed() 
            sigma = self.cfg['train']['augmentation']['jitter_sigma']
            clip = self.cfg['train']['augmentation']['jitter_clip']
            xyz += np.clip(sigma * np.random.randn(*xyz.shape), -clip, clip)
        
        xyz_t = torch.from_numpy(xyz).float()
        rgb_t = torch.from_numpy(rgb).float()
        lbl_t = torch.from_numpy(lbl).long()
        
        # [RESTORED] Normalization & Centering
        xyz_min = xyz_t.min(0)[0]
        xyz_max = xyz_t.max(0)[0]
        xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min + 1e-6)
        
        sft_feat = torch.cat([rgb_t, xyz_norm], dim=1) # [N, 6]
        xyz_centered = xyz_t - xyz_t.mean(0) # [N, 3] Center at 0
        
        # --- V3.0 Addition: Seed Hash (Append only) ---
        # Using ORIGINAL xyz (before aug/center) for consistency? 
        # Ideally yes, but sample is already cropped. 
        # Using sampled xyz is fine as long as we use the values.
        # Use int64 cast for safety.
        h1 = np.abs(xyz[:, 0] * 73856093).astype(np.int64)
        h2 = np.abs(xyz[:, 1] * 19349663).astype(np.int64)
        h3 = np.abs(xyz[:, 2] * 83492791).astype(np.int64)
        seed_hash = h1 ^ h2 ^ h3
        threshold = int(self.label_ratio * 100000)
        is_seed = (seed_hash % 100000) < threshold
        seed_mask = torch.from_numpy(is_seed.astype(bool))
        
        # Return V2.0 tuple + seed_mask
        return xyz_centered, sft_feat, rgb_t, lbl_t, seed_mask