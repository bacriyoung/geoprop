import os
import glob
import numpy as np
import torch
import logging

# 引入基类
from geoprop.data.base_dataset import BaseDataset

class S3DISDataset(BaseDataset):
    """
    S3DIS 专用数据集实现。
    继承自 BaseDataset，自动获取 logger 和通用配置 (num_points, block_size, aug_cfg 等)。
    """
    def __init__(self, cfg, split='train'):
        # 1. 调用父类初始化 
        # (父类会自动设置 self.cfg, self.split, self.logger, self.num_points, self.block_size, self.aug_cfg)
        super().__init__(cfg, split)
        
        root_dir = cfg['dataset']['root_dir']
        
        # 2. 获取全量文件
        all_files = sorted(glob.glob(os.path.join(root_dir, '*.npy')))
        if len(all_files) == 0: 
            raise ValueError(f"No files found in {root_dir}")
        
        # 3. 根据 Split 获取目标区域列表
        # split 参数对应 yaml 中的 dataset.split.xxx_areas
        target_areas = []
        if split == 'train':
            target_areas = cfg['dataset']['split']['train_areas']
        elif split == 'val':
            target_areas = cfg['dataset']['split']['val_areas']
        elif split == 'inference':
            target_areas = cfg['dataset']['split']['inference_areas']
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # 4. 过滤文件
        self.files = []
        for f in all_files:
            # 检查文件名是否包含目标区域字符串 (e.g. "Area_1_")
            for area_idx in target_areas:
                if f"Area_{area_idx}" in os.path.basename(f):
                    self.files.append(f)
                    break
        
        self.logger.info(f"[{split.upper()}] S3DIS Dataset: Loaded {len(self.files)} rooms from Areas {target_areas}")

    def __len__(self):
        # 训练时扩充 Epoch 长度，验证/推理时只跑一遍
        return len(self.files) * (5 if self.split == 'train' else 1)

    def __getitem__(self, idx):
        # 验证/推理时固定随机性，确保结果可复现
        if self.split != 'train':
            np.random.seed(idx)
            
        file_path = self.files[idx % len(self.files)]
        data = np.load(file_path)
        
        # 随机中心点采样 block
        center_idx = np.random.choice(len(data))
        center = data[center_idx, :3]
        
        mask = np.all((data[:, :3] >= center - self.block_size/2) & 
                      (data[:, :3] <= center + self.block_size/2), axis=1)
        block_data = data[mask]
        
        # 点数采样
        if len(block_data) < self.num_points: 
            choice = np.random.choice(len(block_data), self.num_points, replace=True)
        else: 
            choice = np.random.choice(len(block_data), self.num_points, replace=False)
            
        sample = block_data[choice]
        xyz = sample[:, :3].astype(np.float32)
        rgb = sample[:, 3:6].astype(np.float32)
        lbl = sample[:, 6].astype(int)
        
        if rgb.max() > 1.1: rgb /= 255.0
        
        # 数据增强 (仅在训练且配置开启时应用)
        # self.aug_cfg 已经在父类 BaseDataset 中解析好了
        if self.split == 'train' and self.aug_cfg and self.aug_cfg.get('enabled', False):
            np.random.seed() # 恢复随机状态
            sigma = self.aug_cfg.get('jitter_sigma', 0.001)
            clip = self.aug_cfg.get('jitter_clip', 0.005)
            xyz += np.clip(sigma * np.random.randn(*xyz.shape), -clip, clip)
            
            # RGB 颜色扰动
            if np.random.random() > 0.5: 
                rgb = np.clip(0.5 + np.random.uniform(0.9, 1.1) * (rgb - 0.5) + np.random.uniform(-0.05, 0.05), 0, 1)
        
        # 归一化与 Tensor 转换
        xyz_t = torch.from_numpy(xyz).float()
        rgb_t = torch.from_numpy(rgb).float()
        lbl_t = torch.from_numpy(lbl).long()
        
        xyz_min = xyz_t.min(0)[0]
        xyz_max = xyz_t.max(0)[0]
        xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min + 1e-6)
        
        # 特征拼接: RGB + Normalized XYZ
        sft_feat = torch.cat([rgb_t, xyz_norm], dim=1)
        xyz_centered = xyz_t - xyz_t.mean(0)
        
        return xyz_centered, sft_feat, rgb_t, lbl_t