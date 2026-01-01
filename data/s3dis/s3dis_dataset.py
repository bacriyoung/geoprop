import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import yaml
from tqdm import tqdm
import logging

class S3DISDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        
        # 1. Load config/s3dis/s3dis.yaml
        config_path = os.path.join('config', 's3dis', 's3dis.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Dataset config not found at {config_path}")
            
        with open(config_path, 'r') as f:
            self.ds_cfg = yaml.safe_load(f)
            
        # 2. Parse Config
        ds_info = self.ds_cfg.get('dataset', {})
        self.root = ds_info.get('root_dir')
        if not self.root:
            raise ValueError("'dataset.root_dir' is missing in config/s3dis/s3dis.yaml")

        self.label_ratio = cfg['dataset'].get('label_ratio', ds_info.get('label_ratio', 0.001))

        # 3. Load Files
        self.files = self._get_files(ds_info)
        
        if len(self.files) == 0:
            raise RuntimeError(f"[{split.upper()}] No files found in {self.root}. Check 'root_dir' and 'split' in s3dis.yaml!")

        self.data_list = []
        logger = logging.getLogger("geoprop")
        logger.info(f"[{split.upper()}] Loading {len(self.files)} rooms from {self.root}...")
        
        for f in tqdm(self.files, desc=f"Loading {split}"):
            self.data_list.append(np.load(f))

    def _get_files(self, ds_info):
        search_pattern = os.path.join(self.root, "**", "*.npy")
        all_files = sorted(glob.glob(search_pattern, recursive=True))
        
        split_cfg = ds_info.get('split', {})
        
        if self.split == 'train':
            target_areas = split_cfg.get('train_areas', [1, 2, 3, 4, 6])
        elif self.split == 'val':
            target_areas = split_cfg.get('val_areas', [5])
        else: # inference
            target_areas = split_cfg.get('inference_areas', [5])
            
        filtered_files = []
        for f in all_files:
            basename = os.path.basename(f)
            for area_id in target_areas:
                if f"Area_{area_id}_" in basename:
                    filtered_files.append(f)
                    break
        
        return filtered_files

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
        
        # --- V3.0 Universal Seed (Coordinate Hash) ---
        # [FIX] Cast to int64 BEFORE XOR operation
        h1 = np.abs(xyz[:, 0] * 73856093).astype(np.int64)
        h2 = np.abs(xyz[:, 1] * 19349663).astype(np.int64)
        h3 = np.abs(xyz[:, 2] * 83492791).astype(np.int64)
        seed_hash = h1 ^ h2 ^ h3
        
        threshold = int(self.label_ratio * 100000)
        # Modulo on the integer hash
        is_seed = (seed_hash % 100000) < threshold
        seed_mask = torch.from_numpy(is_seed.astype(bool))
        
        xyz_t = torch.from_numpy(xyz_norm).float()
        sft_t = torch.from_numpy(xyz).float() 
        rgb_t = torch.from_numpy(rgb / 255.0).float()
        lbl_t = torch.from_numpy(lbl).long()
        
        return xyz_t, sft_t, rgb_t, lbl_t, seed_mask