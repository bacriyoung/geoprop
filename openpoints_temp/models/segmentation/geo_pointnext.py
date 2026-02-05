import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.models.build import MODELS
from openpoints.models.backbone.pointnext import PointNextEncoder, PointNextDecoder

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def knn_point_chunked(k, query, support):
    """
    分块计算 KNN 以避免 OOM (Out of Memory)。
    验证集点数可能 >50k，直接 cdist 会产生 (B, 50k, 50k) 的矩阵导致显存爆炸。
    """
    B, N, C = query.shape
    M = support.shape[1]
    
    # 分块大小，可根据显存调整，2048 通常安全
    chunk_size = 2048 
    idx_list = []
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        q_chunk = query[:, i:end, :] # (B, Chunk, 3)
        
        # 计算距离矩阵 (B, Chunk, M)
        dist = torch.cdist(q_chunk, support)
        
        # 取前 k 个最近邻
        _, idx = dist.topk(k, dim=-1, largest=False) # (B, Chunk, k)
        idx_list.append(idx)
        
    return torch.cat(idx_list, dim=1) # (B, N, k)

def dense_gather(features, idx):
    """
    Gather: (B, C, N) -> (B, C, N, K) using (B, N, K) indices
    """
    B, C, N = features.shape
    K = idx.shape[2]
    
    idx_base = torch.arange(0, B, device=features.device).view(B, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    features_flat = features.transpose(1, 2).contiguous().view(B * N, C)
    out = features_flat[idx, :]
    out = out.view(B, N, K, C).permute(0, 3, 1, 2).contiguous()
    return out

def compute_lean_gblobs_dense(xyz, k=16, knn_idx=None, scale=1.0):
    B, N, _ = xyz.shape
    
    # 1. KNN Query (Chunked to prevent OOM)
    if knn_idx is None:
        knn_idx = knn_point_chunked(k, xyz, xyz)
        knn_idx = knn_idx.long()
    
    # 2. Gather Neighbors
    xyz_trans = xyz.transpose(1, 2).contiguous() # (B, 3, N)
    neighbors = dense_gather(xyz_trans, knn_idx) # (B, 3, N, K)
    
    # 3. Compute Covariance
    neighbors = neighbors.permute(0, 2, 3, 1) # (B, N, K, 3)
    local_mean = neighbors.mean(dim=2, keepdim=True)
    centered = (neighbors - local_mean) * scale
    
    centered_t = centered.transpose(2, 3)
    cov = torch.matmul(centered_t, centered) / (k - 1 + 1e-6)
    
    geo_blobs = cov.view(B, N, 9).transpose(1, 2).contiguous()
    geo_blobs = torch.sign(geo_blobs) * torch.pow(torch.abs(geo_blobs) + 1e-8, 0.25)
    
    return geo_blobs, knn_idx

# -------------------------------------------------------------------------
# JAFAR Module
# -------------------------------------------------------------------------

class DecoupledPointJAFAR(nn.Module):
    def __init__(self, qk_dim=64, k=16, input_geo_dim=12, sem_dim=192, num_classes=13): 
        super().__init__()
        self.qk_dim = qk_dim
        self.k = k
        self.input_geo_dim = input_geo_dim 
        self.sem_dim = sem_dim 

        self.geom_encoder = nn.Sequential(
            nn.Conv1d(self.input_geo_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), nn.ReLU(),
            nn.Conv1d(qk_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), nn.ReLU()
        )
        self.val_proj = nn.Sequential(
            nn.Conv1d(sem_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), nn.ReLU()
        )
        self.geo_query = nn.Conv1d(qk_dim, qk_dim, 1)
        self.geo_key = nn.Conv1d(qk_dim, qk_dim, 1)
        self.rel_pos_mlp = nn.Sequential(
            nn.Conv2d(7, qk_dim, 1), 
            nn.GroupNorm(8, qk_dim), nn.ReLU(),
            nn.Conv2d(qk_dim, qk_dim, 1)
        )
        self.cls_head = nn.Conv1d(qk_dim, num_classes, 1)
        
        # [FIX] 重建头输出维度 = 3(Coord) + 3(RGB) = 6
        self.rec_head = nn.Sequential(
            nn.Conv1d(qk_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), nn.ReLU(),
            nn.Conv1d(qk_dim, 6, 1) 
        )

    def forward(self, xyz, jafar_feat, sem_feat, knn_idx):
        B, N, _ = xyz.shape
        geom_emb = self.geom_encoder(jafar_feat)
        Q = self.geo_query(geom_emb)
        K = self.geo_key(geom_emb)
        V = self.val_proj(sem_feat)

        K_g = dense_gather(K, knn_idx)
        V_g = dense_gather(V, knn_idx)
        
        xyz_trans = xyz.transpose(1, 2).contiguous()
        xyz_g = dense_gather(xyz_trans, knn_idx)
        
        rel_diff = xyz_trans.unsqueeze(-1) - xyz_g
        sq_sum = torch.sum(rel_diff ** 2, dim=1, keepdim=True)
        rel_dist = torch.sqrt(sq_sum + 1e-10)
        rel_direction = rel_diff / (rel_dist + 1e-5)
        
        rel_geo_feat = torch.cat([rel_diff, rel_dist, rel_direction], dim=1)
        pos_enc = self.rel_pos_mlp(rel_geo_feat)
        
        attn_logits = torch.sum(Q.unsqueeze(-1) * (K_g + pos_enc), dim=1) / (self.qk_dim ** 0.5)
        affinity = torch.softmax(attn_logits, dim=-1)
        
        refined_feat = torch.sum(affinity.unsqueeze(1) * V_g, dim=-1)
        refined_feat = refined_feat + V
        
        logits = self.cls_head(refined_feat)
        rec_phys = self.rec_head(refined_feat)
        return logits, affinity, refined_feat, rec_phys

# -------------------------------------------------------------------------
# GeoPointNeXt (Model)
# -------------------------------------------------------------------------

@MODELS.register_module()
class GeoPointNeXt(nn.Module):
    def __init__(self, encoder_args, decoder_args, geo_args, num_classes, 
                 in_channels=None, 
                 geo_scale=0.04, criteria=None, 
                 **kwargs):
        super().__init__()
        
        if in_channels is not None:
            encoder_args['in_channels'] = in_channels

        self.encoder = PointNextEncoder(**encoder_args)
        
        if hasattr(self.encoder, 'channel_list'):
            decoder_args['encoder_channel_list'] = self.encoder.channel_list
            
        self.decoder = PointNextDecoder(**decoder_args)
        self.sem_feat_dim = decoder_args.get('width', 32)
        self.aux_head = nn.Conv1d(self.sem_feat_dim, num_classes, 1)
        
        self.geo_scale = geo_scale
        input_geo_dim = 9 + encoder_args.get('in_channels', 4)
        
        self.geo_stream = DecoupledPointJAFAR(
            qk_dim=geo_args.get('qk_dim', 64),
            k=geo_args.get('k', 16),
            input_geo_dim=input_geo_dim,
            sem_dim=self.sem_feat_dim, 
            num_classes=num_classes
        )
        
        self.criterion = criteria

    def isotropic_normalize_batch(self, pos):
        B, N, _ = pos.shape
        xyz_min = pos.min(dim=1, keepdim=True)[0]
        xyz_max = pos.max(dim=1, keepdim=True)[0]
        scale = (xyz_max - xyz_min).max(dim=-1, keepdim=True)[0] + 1e-6
        iso_coord = (pos - xyz_min) / scale
        return iso_coord

    def forward(self, data):
        p, x = data['pos'], data['x']
        
        p_list, f_list = self.encoder(p, x)
        sem_feat = self.decoder(p_list, f_list)
        aux_logits = self.aux_head(sem_feat)
        
        iso_coord = self.isotropic_normalize_batch(p)
        geo_blobs, knn_idx = compute_lean_gblobs_dense(iso_coord, k=16, scale=self.geo_scale)
        
        jafar_input = torch.cat([geo_blobs, x], dim=1)
        
        refined_logits, affinity, refined_feat, rec_phys = self.geo_stream(
            xyz=iso_coord,
            jafar_feat=jafar_input, 
            sem_feat=sem_feat,
            knn_idx=knn_idx 
        )
        
        # [FIX] 恢复重建目标：IsoCoord + RGB
        iso_coord_trans = iso_coord.transpose(1, 2).contiguous()
        rgb = x[:, :3, :] # 假设前3通道是RGB
        target_phys = torch.cat([iso_coord_trans, rgb], dim=1)
        
        output_dict = {
            "seg_logits": refined_logits,
            "aux_logits": aux_logits,
            "affinity": affinity,
            "refined_feat": refined_feat,
            "rec_phys": rec_phys,
            "knn_idx": knn_idx,
            "target_phys": target_phys
        }
        
        return output_dict