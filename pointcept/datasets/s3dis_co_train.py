import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose

@DATASETS.register_module()
class S3DISCoTrainDataset(Dataset):
    def __init__(self,
                 split='train',
                 data_root='data/s3dis',
                 transform=None, # [Important] Accept pipeline from Config
                 loop=1,
                 labeled_ratio=0.001,
                 hash_seed_1=57361723,
                 hash_seed_2=92990218,
                 hash_seed_3=69232043,
                 test_mode=False,
                 **kwargs): 
        
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform) # Use standard pointcept transforms
        self.labeled_ratio = labeled_ratio
        self.test_mode = test_mode
        self.logger = get_root_logger()
        
        # Hash seeds for coordinate-based masking
        self.h1_k = int(hash_seed_1)
        self.h2_k = int(hash_seed_2)
        self.h3_k = int(hash_seed_3)

        # Force loop for training to ensure sufficient iterations
        if self.split == 'train' and loop == 1:
            self.loop = 30
        else:
            self.loop = loop
            
        self.data_list = self.get_file_list()
        
        if len(self.data_list) > 0:
            self.data_list = self.data_list * self.loop
            if self.logger is not None:
                self.logger.info(f"[{self.split}] CoTrain Dataset Loaded.")
                self.logger.info(f"   - Ratio: {self.labeled_ratio*100}%")
                self.logger.info(f"   - Loop: {self.loop}")
                self.logger.info(f"   - Strategy: Coordinate Hashing (Mask before Transform)")
        else:
            raise ValueError(f"❌ [Dataset] No files found in {self.data_root}")

    def get_file_list(self):
        if isinstance(self.data_root, str): 
            data_roots = [self.data_root]
        else:
            data_roots = self.data_root
            
        data_list = []
        for root in data_roots:
            if not os.path.exists(root): continue
            for dirpath, _, filenames in os.walk(root):
                if "coord.npy" in filenames:
                    # S3DIS Standard Split Logic
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
            coord = np.load(os.path.join(room_dir, "coord.npy")).astype(np.float32)
            color = np.load(os.path.join(room_dir, "color.npy")).astype(np.float32)
            segment = np.load(os.path.join(room_dir, "segment.npy")).astype(np.int64).reshape(-1)
        except Exception as e:
            print(f"Error loading {room_dir}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

        # ==================================================================
        # 1. Sparse Label Masking (Coordinate Hashing)
        # ==================================================================
        # CRITICAL: We mask BEFORE any geometric transformation (crop/rotate).
        # This ensures the mask is locked to the physical world coordinates.
        if self.split == 'train' and self.labeled_ratio < 1.0:
            h1 = np.abs(coord[:, 0] * self.h1_k).astype(np.int64)
            h2 = np.abs(coord[:, 1] * self.h2_k).astype(np.int64)
            h3 = np.abs(coord[:, 2] * self.h3_k).astype(np.int64)
            seed_hash = h1 ^ h2 ^ h3
            
            threshold = int(self.labeled_ratio * 100000)
            label_mask = (seed_hash % 100000) < threshold
            
            # Set Unlabeled to ignore_index (255)
            # 255 is the standard ignore_index in S3DIS config
            segment[~label_mask] = 255

        # ==================================================================
        # 2. Standard Transforms
        # ==================================================================
        # We construct the dict exactly as S3DISDataset does.
        # The 'transform' pipeline (SphereCrop, GridSample, ToTensor) will handle the rest.
        data_dict = dict(
            coord=coord,
            normal=color, # In S3DIS, color is often treated as feature. Explicitly mapped later.
            color=color,
            segment=segment
        )
        
        if self.transform is not None:
            data_dict = self.transform(data_dict)
            
        return data_dict