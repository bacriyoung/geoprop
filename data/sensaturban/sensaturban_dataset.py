import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import logging
from tqdm import tqdm

# Import plyfile for reading raw SensatUrban data (.ply)
try:
    from plyfile import PlyData, PlyElementParseError
except ImportError:
    PlyData = None
    PlyElementParseError = Exception

class SensatUrbanDataset(Dataset):
    """
    Dataset class for SensatUrban. 
    Supports loading from raw .ply files or pre-processed .npy files.
    Includes robust error handling for truncated files.
    """
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.logger = logging.getLogger("geoprop")
        
        # Check dependency
        if PlyData is None:
            raise ImportError("SensatUrban raw data is .ply format. Please run: pip install plyfile")
        
        ds_cfg = cfg['dataset']
        self.root = ds_cfg.get('root_dir')
        if not self.root:
            raise ValueError("root_dir not found in config.")
            
        self.num_points = ds_cfg.get('num_points', 65536) 
        self.block_size = ds_cfg.get('block_size', 4.0)
        self.label_ratio = ds_cfg.get('label_ratio', 0.001)

        self.files = self._get_files()
        
        if len(self.files) == 0:
            raise RuntimeError(f"[{split.upper()}] No files found in {self.root}! Ensure paths are correct and files are .ply or .npy.")

        self.logger.info(f"[{split.upper()}] Found {len(self.files)} potential block files. Loading into memory...")

        # [Robust Loading Logic]
        self.data_list = []
        valid_files = []
        
        # Use tqdm to show loading progress as SensatUrban blocks are large
        pbar = tqdm(self.files, desc=f"Loading {split} data", unit="block")
        
        for f in pbar:
            try:
                if f.endswith('.ply'):
                    data = self._load_ply(f)
                else:
                    data = np.load(f)
                
                self.data_list.append(data)
                valid_files.append(f)
                
            except (PlyElementParseError, EOFError, OSError) as e:
                # [CRITICAL FIX] Catch truncated files and skip them instead of crashing
                pbar.write(f"\n[WARNING] Skipping corrupted file: {os.path.basename(f)}")
                pbar.write(f"          Reason: {str(e)}")
                continue
            except Exception as e:
                pbar.write(f"\n[ERROR] Unknown error loading {os.path.basename(f)}: {e}")
                continue
                
        self.files = valid_files # Update file list to match loaded data
        
        if len(self.data_list) == 0:
             raise RuntimeError(f"All files in {self.root} failed to load! Please check your dataset integrity.")

        self.logger.info(f"[{split.upper()}] SensatUrban Dataset: Successfully loaded {len(self.data_list)} valid blocks.")

    def _get_files(self):
        # Search for both .ply (raw) and .npy (processed)
        files = []
        # Search recursively
        for ext in ["*.ply", "*.npy"]:
            pattern = os.path.join(self.root, "**", ext)
            files.extend(sorted(glob.glob(pattern, recursive=True)))
        
        # Remove duplicates if any and sort
        return sorted(list(set(files)))

    def _load_ply(self, path):
        """
        Helper to read SensatUrban .ply files and convert to (N, 7) float32 array.
        Structure: [x, y, z, r, g, b, label]
        """
        ply = PlyData.read(path)
        data = ply['vertex'].data
        
        # Extract coordinates
        x, y, z = data['x'], data['y'], data['z']
        
        # Extract colors (SensatUrban usually has red, green, blue)
        if 'red' in data.dtype.names:
            r, g, b = data['red'], data['green'], data['blue']
        else:
            r = g = b = np.zeros_like(x)
            
        # Extract labels (Test set might not have 'class')
        if 'class' in data.dtype.names:
            lbl = data['class']
        else:
            lbl = np.zeros_like(x) # Dummy label for test
        
        # Stack into (N, 7) array
        features = np.vstack([x, y, z, r, g, b, lbl]).T
        return features.astype(np.float32)

    def __len__(self):
        # Epoch multiplier
        return len(self.files) * (4 if self.split == 'train' else 1)

    def __getitem__(self, idx):
        if self.split != 'train':
            np.random.seed(idx)
            
        room_idx = idx % len(self.data_list)
        data = self.data_list[room_idx]
        
        N = data.shape[0]
        
        # Random Center Crop
        center_idx = np.random.choice(N)
        center = data[center_idx, :3]
        
        # Crop logic
        mask = np.all((data[:, :3] >= center - self.block_size/2) & 
                      (data[:, :3] <= center + self.block_size/2), axis=1)
        block_data = data[mask]
        
        # Resample
        if len(block_data) < self.num_points: 
            # Pad with replacement if not enough points
            choice = np.random.choice(len(block_data), self.num_points, replace=True)
        else: 
            choice = np.random.choice(len(block_data), self.num_points, replace=False)
            
        sample = block_data[choice]
        
        xyz = sample[:, :3]
        rgb = sample[:, 3:6]
        lbl = sample[:, 6].astype(int)
        
        # Normalize RGB if it's in 0-255 range
        if rgb.max() > 1.1: 
            rgb /= 255.0

        # Seed Hash Logic (V2 Compatible)
        h1 = np.abs(xyz[:, 0] * 73856093).astype(np.int64)
        h2 = np.abs(xyz[:, 1] * 19349663).astype(np.int64)
        h3 = np.abs(xyz[:, 2] * 83492791).astype(np.int64)
        seed_hash = h1 ^ h2 ^ h3
        threshold = int(self.label_ratio * 100000)
        is_seed = (seed_hash % 100000) < threshold
        seed_mask = torch.from_numpy(is_seed.astype(bool))

        # Augmentation
        if self.split == 'train' and self.cfg['train'].get('augmentation', {}).get('enabled', False):
            aug_cfg = self.cfg['train']['augmentation']
            np.random.seed() 
            
            if aug_cfg.get('rotate', False):
                angle = np.random.uniform(0, 2 * np.pi)
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                xyz = np.dot(xyz, R.T)
            
            if aug_cfg.get('scale', False):
                scale = np.random.uniform(0.8, 1.2)
                xyz *= scale
                
            if aug_cfg.get('flip', False):
                if np.random.random() > 0.5: xyz[:, 0] = -xyz[:, 0]
                if np.random.random() > 0.5: xyz[:, 1] = -xyz[:, 1]

            sigma = aug_cfg.get('jitter_sigma', 0.001)
            clip = aug_cfg.get('jitter_clip', 0.005)
            xyz += np.clip(sigma * np.random.randn(*xyz.shape), -clip, clip)
        
        xyz_t = torch.from_numpy(xyz).float()
        rgb_t = torch.from_numpy(rgb).float()
        lbl_t = torch.from_numpy(lbl).long()
        
        # Normalize XYZ to local block
        xyz_min = xyz_t.min(0)[0]
        xyz_max = xyz_t.max(0)[0]
        xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min + 1e-6)
        
        sft_feat = torch.cat([rgb_t, xyz_norm], dim=1) 
        xyz_centered = xyz_t - xyz_t.mean(0) 
        
        return xyz_centered, sft_feat, rgb_t, lbl_t, seed_mask