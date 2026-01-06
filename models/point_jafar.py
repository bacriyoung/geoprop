import torch
import torch.nn as nn

class DecoupledPointJAFAR(nn.Module):
    def __init__(self, qk_dim=64, k=16, input_mode='gblobs'): 
        super().__init__()
        self.qk_dim = qk_dim
        self.k = k
        self.input_mode = input_mode
        
        if input_mode == 'absolute':
            self.input_geo_dim = 6  
        elif input_mode == 'gblobs':
            self.input_geo_dim = 18
        elif input_mode == 'mix':
            # 9 dims from Geometric GBlobs + 3 dims from RGB = 12 dims
            self.input_geo_dim = 12
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.geom_encoder = nn.Sequential(
            nn.Conv1d(self.input_geo_dim, qk_dim, 1),
            nn.BatchNorm1d(qk_dim), nn.ReLU(),
            nn.Conv1d(qk_dim, qk_dim, 1),
            nn.BatchNorm1d(qk_dim), nn.ReLU()
        )
        
        self.scale_conv = nn.Conv1d(6, qk_dim, 1)
        self.shift_conv = nn.Conv1d(6, qk_dim, 1)
        
        self.sem_query = nn.Conv1d(qk_dim, qk_dim, 1)
        self.sem_key = nn.Conv1d(qk_dim, qk_dim, 1)
        
        self.bdy_head = nn.Sequential(
            nn.Conv1d(qk_dim, 32, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 1), nn.Sigmoid()
        )
        
        self.rel_pos_mlp = nn.Sequential(
            nn.Conv2d(3, qk_dim, 1), nn.BatchNorm2d(qk_dim), nn.ReLU(),
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

    def forward(self, xyz_hr, xyz_lr, val_lr, feat_hr, feat_lr):
        B, _, M = xyz_lr.shape
        curr_k = min(self.k, M)
        
        geom_hr = self.geom_encoder(feat_hr) 
        geom_lr = self.geom_encoder(feat_lr)
        
        if val_lr.shape[1] == 6: 
            scale = self.scale_conv(val_lr)
            shift = self.shift_conv(val_lr)
            geom_lr = geom_lr * (scale + 1) + shift

        bdy_prob = self.bdy_head(geom_hr)
        Q = self.sem_query(geom_hr)
        K = self.sem_key(geom_lr) 
        
        with torch.no_grad(): 
            dists = torch.cdist(xyz_hr.transpose(1, 2), xyz_lr.transpose(1, 2))
            _, k_idx = torch.topk(dists, curr_k, dim=-1, largest=False)
            
        K_g = self._gather_val_efficient(K, k_idx)
        xyz_lr_g = self._gather_val_efficient(xyz_lr, k_idx)
        
        rel_pos = xyz_hr.unsqueeze(-1) - xyz_lr_g
        pos_enc = self.rel_pos_mlp(rel_pos)
        attn = self.softmax(torch.sum(Q.unsqueeze(-1) * (K_g + pos_enc), dim=1) / (self.qk_dim ** 0.5))
        val_g = self._gather_val_efficient(val_lr, k_idx)
        
        return torch.sum(attn.unsqueeze(1) * val_g, dim=-1), bdy_prob