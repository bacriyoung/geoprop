import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoupledPointJAFAR(nn.Module):
    def __init__(self, qk_dim=64, k=16, input_geo_dim=18): 
        super().__init__()
        self.qk_dim = qk_dim
        self.k = k
        
        # 1. Guidance Encoder (The "Relative" Eye)
        # Input: 9 (Geo Cov) + 9 (RGB Cov) = 18 dim
        self.geom_encoder = nn.Sequential(
            nn.Conv1d(input_geo_dim, qk_dim, 1),
            nn.BatchNorm1d(qk_dim),
            nn.ReLU(),
            nn.Conv1d(qk_dim, qk_dim, 1),
            nn.BatchNorm1d(qk_dim),
            nn.ReLU()
        )
        
        # 2. Modulators (The "Absolute" Adjuster)
        # Training Target Value: XYZ(3) + RGB(3) = 6 dim
        self.scale_conv = nn.Conv1d(6, qk_dim, 1)
        self.shift_conv = nn.Conv1d(6, qk_dim, 1)
        
        # 3. Attention
        self.sem_query = nn.Conv1d(qk_dim, qk_dim, 1)
        self.sem_key = nn.Conv1d(qk_dim, qk_dim, 1)
        
        # 4. Boundary Head (BFANet integration)
        self.bdy_head = nn.Sequential(
            nn.Conv1d(qk_dim, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Relative Position Encoding (Essential for spatial awareness without abs XYZ input)
        self.rel_pos_mlp = nn.Sequential(
            nn.Conv2d(3, qk_dim, 1),
            nn.BatchNorm2d(qk_dim),
            nn.ReLU(),
            nn.Conv2d(qk_dim, qk_dim, 1)
        )
        self.softmax = nn.Softmax(dim=-1)

    def _gather_val_efficient(self, tensor, idx):
        B, C, M = tensor.shape
        _, N, k = idx.shape
        tensor_flat = tensor.transpose(1, 2).contiguous().view(B * M, C)
        batch_offset = torch.arange(B, device=tensor.device).view(B, 1, 1) * M
        flat_idx = (idx + batch_offset).view(-1)
        return tensor_flat[flat_idx].view(B, N, k, C).permute(0, 3, 1, 2)

    def forward(self, xyz_hr, xyz_lr, val_lr, geo_blobs_hr, geo_blobs_lr, rgb_blobs_hr, rgb_blobs_lr):
        """
        xyz: ONLY for relative position encoding. NOT input feature.
        val_lr: Values to propagate (Training: 6-dim Abs XYZ+RGB, Inference: One-Hot)
        geo_blobs: [B, 9, N] Geometric Covariance
        rgb_blobs: [B, 9, N] Color Covariance
        """
        B, _, M = xyz_lr.shape
        curr_k = min(self.k, M)
        
        # [V31 Input] Concatenate Relative Features: 9 + 9 = 18
        feat_hr = torch.cat([geo_blobs_hr, rgb_blobs_hr], dim=1) # [B, 18, N]
        feat_lr = torch.cat([geo_blobs_lr, rgb_blobs_lr], dim=1) # [B, 18, M]

        # Encode Guidance
        geom_hr = self.geom_encoder(feat_hr) 
        geom_lr = self.geom_encoder(feat_lr)
        
        # Modulation (Only during training when val is 6-dim absolute info)
        if val_lr.shape[1] == 6: 
            scale = self.scale_conv(val_lr)
            shift = self.shift_conv(val_lr)
            geom_lr = geom_lr * (scale + 1) + shift

        bdy_prob = self.bdy_head(geom_hr)
        
        # Attention
        Q = self.sem_query(geom_hr)
        K = self.sem_key(geom_lr) 
        
        # KNN (Physical Distance)
        with torch.no_grad(): 
            dists = torch.cdist(xyz_hr.transpose(1, 2), xyz_lr.transpose(1, 2))
            _, k_idx = torch.topk(dists, curr_k, dim=-1, largest=False)
            
        K_g = self._gather_val_efficient(K, k_idx)
        xyz_lr_g = self._gather_val_efficient(xyz_lr, k_idx)
        
        # Relative Position Encoding
        rel_pos = xyz_hr.unsqueeze(-1) - xyz_lr_g
        pos_enc = self.rel_pos_mlp(rel_pos)
        
        # Softmax Attention
        attn_logits = torch.sum(Q.unsqueeze(-1) * (K_g + pos_enc), dim=1) / (self.qk_dim ** 0.5)
        attn = self.softmax(attn_logits)
        
        # Propagation
        val_g = self._gather_val_efficient(val_lr, k_idx)
        rec = torch.sum(attn.unsqueeze(1) * val_g, dim=-1)
        
        return rec, bdy_prob