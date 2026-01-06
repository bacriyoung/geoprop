import torch
import torch.nn as nn
import torch.nn.functional as F
import pointops
import inspect
from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

# =========================================================================
# 0. GBlobs Calculation Utilities
# =========================================================================
def compute_covariance_features(features, knn_indices, k=16):
    """
    Compute Covariance Matrix (GBlobs) efficiently.
    features: [B, N, C]
    knn_indices: [B, N, k]
    """
    B, N, C = features.shape
    batch_idx = torch.arange(B, device=features.device).view(B, 1, 1).expand(-1, N, k)
    
    feat_flat = features.view(B*N, C)
    idx_flat = knn_indices.view(B, N, k) + (batch_idx * N)
    idx_flat = idx_flat.view(-1)
    
    neighbors = feat_flat[idx_flat].view(B, N, k, C)
    
    # Centering
    local_mean = neighbors.mean(dim=2, keepdim=True) 
    centered = neighbors - local_mean 
    
    # Covariance: (X^T * X) / (k-1)
    centered_t = centered.transpose(2, 3)
    cov = torch.matmul(centered_t, centered) / (k - 1 + 1e-6)
    
    cov_flat = cov.view(B, N, C*C)
    return cov_flat

def compute_lean_gblobs(xyz, k=16):
    """
    Compute ONLY Geometric GBlobs (9-dim).
    """
    B, N, _ = xyz.shape
    xyz_flat = xyz.view(-1, 3).contiguous()
    offset = torch.arange(1, B + 1, dtype=torch.int32, device=xyz.device) * N
    idx_flat = pointops.knn_query(k, xyz_flat, offset)[0].long()
    
    batch_start = (torch.arange(B, device=xyz.device) * N).view(B, 1, 1)
    idx = idx_flat.view(B, N, k) - batch_start
    
    # Scaling by 10.0 for stability
    geo_blobs = compute_covariance_features(xyz * 10.0, idx, k)
    return geo_blobs # [B, N, 9]

# =========================================================================
# 1. DecoupledPointJAFAR (Option A: Semantic Refinement)
# =========================================================================
class DecoupledPointJAFAR(nn.Module):
    def __init__(self, qk_dim=64, k=16, input_geo_dim=12, sem_dim=192, num_classes=13): 
        super().__init__()
        self.qk_dim = qk_dim
        self.k = k
        self.input_geo_dim = input_geo_dim 
        self.sem_dim = sem_dim 

        # A. Geometry Encoder (GBlobs+RGB)
        self.geom_encoder = nn.Sequential(
            nn.Conv1d(self.input_geo_dim, qk_dim, 1),
            nn.BatchNorm1d(qk_dim), nn.ReLU(),
            nn.Conv1d(qk_dim, qk_dim, 1),
            nn.BatchNorm1d(qk_dim), nn.ReLU()
        )
        
        # B. Semantic Projector (PTv3 Features)
        self.val_proj = nn.Sequential(
            nn.Conv1d(sem_dim, qk_dim, 1),
            nn.BatchNorm1d(qk_dim), nn.ReLU()
        )

        self.geo_query = nn.Conv1d(qk_dim, qk_dim, 1)
        self.geo_key = nn.Conv1d(qk_dim, qk_dim, 1)
        
        self.rel_pos_mlp = nn.Sequential(
            nn.Conv2d(3, qk_dim, 1), nn.BatchNorm2d(qk_dim), nn.ReLU(),
            nn.Conv2d(qk_dim, qk_dim, 1)
        )
        
        self.bdy_head = nn.Sequential(
            nn.Conv1d(qk_dim, 32, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
        
        self.softmax = nn.Softmax(dim=-1)
        self.cls_head = nn.Linear(qk_dim, num_classes)

    def _gather_val_efficient(self, tensor, idx):
        B, C, N = tensor.shape
        _, _, K = idx.shape
        tensor_flat = tensor.transpose(1, 2).contiguous().view(B * N, C)
        batch_offset = torch.arange(B, device=tensor.device).view(B, 1, 1) * N
        flat_idx = (idx + batch_offset).view(-1)
        val = tensor_flat[flat_idx].view(B, N, K, C).permute(0, 3, 1, 2)
        return val

    def forward(self, xyz, jafar_feat, sem_feat):
        B, N, _ = jafar_feat.shape
        jafar_feat_t = jafar_feat.transpose(1, 2).contiguous()
        sem_feat_t = sem_feat.transpose(1, 2).contiguous()
        xyz_t = xyz.transpose(1, 2).contiguous()
        
        # 1. Encode
        geom_emb = self.geom_encoder(jafar_feat_t)
        bdy_logits = self.bdy_head(geom_emb) 
        Q = self.geo_query(geom_emb)
        K = self.geo_key(geom_emb)
        V = self.val_proj(sem_feat_t)

        # 2. KNN
        xyz_flat = xyz.view(-1, 3).contiguous()
        offset = torch.arange(1, B + 1, dtype=torch.int32, device=xyz.device) * N
        k_idx_flat = pointops.knn_query(self.k, xyz_flat, offset)[0].long()
        batch_start = (torch.arange(B, device=xyz.device) * N).view(B, 1, 1)
        k_idx = k_idx_flat.view(B, N, self.k) - batch_start
        
        # 3. Attention
        K_g = self._gather_val_efficient(K, k_idx)
        xyz_g = self._gather_val_efficient(xyz_t, k_idx)
        V_g = self._gather_val_efficient(V, k_idx)
        
        rel_pos = xyz_t.unsqueeze(-1) - xyz_g
        pos_enc = self.rel_pos_mlp(rel_pos)
        
        attn_logits = torch.sum(Q.unsqueeze(-1) * (K_g + pos_enc), dim=1) / (self.qk_dim ** 0.5)
        affinity = self.softmax(attn_logits)
        
        # 4. Refine
        refined_feat = torch.sum(affinity.unsqueeze(1) * V_g, dim=-1)
        refined_feat = refined_feat + V 
        
        refined_feat_flat = refined_feat.transpose(1, 2).contiguous().view(-1, self.qk_dim)
        logits = self.cls_head(refined_feat_flat)
        
        return logits, affinity, k_idx, refined_feat_flat, bdy_logits

# =========================================================================
# 2. GeoPTV3 Main Model
# =========================================================================
@MODELS.register_module()
class GeoPTV3(nn.Module):
    def __init__(self, 
                 backbone_ptv3_cfg, 
                 geo_input_dim=6, 
                 num_classes=13,
                 num_points=80000, 
                 criteria=None):
        super().__init__()
        
        # 1. Semantic Stream: PTv3
        valid_params = inspect.signature(PointTransformerV3.__init__).parameters
        clean_cfg = {k: v for k, v in backbone_ptv3_cfg.items() if k in valid_params}
        self.sem_stream = PointTransformerV3(**clean_cfg)
        
        # [Fix] Read from config dictionary instead of object attribute
        self.ptv3_in_channels = backbone_ptv3_cfg.get("in_channels", 6)
        
        dec_channels = backbone_ptv3_cfg.get('dec_channels', [48, 96, 192, 384])
        self.sem_feat_dim = dec_channels[0]
        self.aux_head = nn.Linear(self.sem_feat_dim, num_classes)
        
        # 2. Geometric Stream: JAFAR (Lean Mode: 12 dim)
        self.num_points = num_points
        self.real_geo_dim = 12 
        print(f"🚀 [GeoPTV3] Lean Mode: JAFAR Input Dim = {self.real_geo_dim} (9 GeoGBlobs + 3 RGB)")
        
        self.geo_stream = DecoupledPointJAFAR(
            qk_dim=64, 
            k=16, 
            input_geo_dim=self.real_geo_dim, 
            sem_dim=self.sem_feat_dim, 
            num_classes=num_classes
        )
        
        self.register_buffer("prototypes", torch.zeros(num_classes, 64))
        self.register_buffer("proto_count", torch.zeros(num_classes))
        self.momentum = 0.99

        if criteria is not None:
            self.criteria = LOSSES.build(criteria)
        else:
            self.criteria = None

    def update_prototypes(self, features, labels):
        with torch.no_grad():
            for c in range(self.aux_head.out_features):
                mask = (labels == c)
                if mask.sum() > 0:
                    curr_proto = features[mask].mean(0)
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * curr_proto
                    self.proto_count[c] += 1

    def forward(self, input_dict):
        # -----------------------------------------------------------
        # A. PTv3 Prep
        # -----------------------------------------------------------
        if "jafar_coord" in input_dict:
            j_coord = input_dict['jafar_coord']
            j_feat_raw = input_dict['jafar_feat']
        else:
            j_coord = input_dict['coord'].clone()
            j_feat_raw = input_dict['feat'].clone()
            
        if j_coord.dim() == 2:
            total_points = j_coord.shape[0]
            if "batch" in input_dict:
                B_size = input_dict["batch"].max().item() + 1
            else:
                B_size = total_points // self.num_points
            valid_len = B_size * self.num_points
            j_coord = j_coord[:valid_len].view(B_size, self.num_points, -1)
            j_feat_raw = j_feat_raw[:valid_len].view(B_size, self.num_points, -1)
        else:
            B_size = j_coord.shape[0]

        ptv3_input = {}
        raw_coord = input_dict["coord"]
        raw_feat = input_dict.get("ptv3_feat", input_dict.get("feat"))
        raw_grid = input_dict.get("grid_coord")
        
        # 1. Flatten for PTv3
        if raw_coord.dim() == 3: 
            flat_coord = raw_coord.reshape(-1, 3).contiguous()
            flat_feat = raw_feat.reshape(-1, raw_feat.shape[-1]).contiguous()
            if raw_grid is not None:
                flat_grid = raw_grid.reshape(-1, 3).contiguous().int()
            N_total = raw_coord.shape[1]
            ptv3_input["batch"] = torch.arange(B_size, device=raw_coord.device).repeat_interleave(N_total)
        else:
            flat_coord = raw_coord
            flat_feat = raw_feat
            flat_grid = raw_grid
            if "batch" in input_dict:
                ptv3_input["batch"] = input_dict["batch"]
            else:
                ptv3_input["batch"] = torch.arange(B_size, device=raw_coord.device).repeat_interleave(self.num_points)

        ptv3_input["coord"] = flat_coord
        ptv3_input["feat"] = flat_feat
        
        # Input Channel Adaptation
        # If PTv3 expects 6 channels (RGB+XYZ) but 'feat' is 3 (RGB), append coords.
        if self.ptv3_in_channels == 6 and flat_feat.shape[1] == 3:
            ptv3_input["feat"] = torch.cat([flat_feat, flat_coord], dim=1)
        
        if flat_grid is None:
            ptv3_input["grid_coord"] = (flat_coord / 0.02).int()
        else:
            ptv3_input["grid_coord"] = flat_grid

        # -----------------------------------------------------------
        # B. Stage I: PTv3 Forward
        # -----------------------------------------------------------
        sem_feat_sparse = self.sem_stream(ptv3_input).feat 
        aux_logits = self.aux_head(sem_feat_sparse) 
        
        # -----------------------------------------------------------
        # C. Feature Assembly for JAFAR (Lean 12-Dim)
        # -----------------------------------------------------------
        sem_feat_dense = sem_feat_sparse.view(B_size, self.num_points, -1)
        
        # 1. GBlobs (9-dim)
        geo_blobs = compute_lean_gblobs(j_coord, k=16) 
        
        # 2. RGB (3-dim)
        rgb_feat = j_feat_raw[:, :, :3]
        
        # 3. Concatenate (12-dim)
        jafar_input = torch.cat([geo_blobs, rgb_feat], dim=-1)
        
        # -----------------------------------------------------------
        # D. PointJAFAR Refinement
        # -----------------------------------------------------------
        refined_logits, affinity, k_idx, refined_feat, bdy_logits = self.geo_stream(
            xyz=j_coord,
            jafar_feat=jafar_input,
            sem_feat=sem_feat_dense
        )
        
        # -----------------------------------------------------------
        # E. Outputs
        # -----------------------------------------------------------
        targets = input_dict['segment'].view(-1)
        
        if self.training:
            valid_mask = (targets != 255)
            if valid_mask.sum() > 0:
                self.update_prototypes(refined_feat[valid_mask].detach(), targets[valid_mask])

        output_dict = {
            "seg_logits": refined_logits,
            "refined_logits": refined_logits,
            "aux_logits": aux_logits,
            "bdy_logits": bdy_logits,
            "refined_feat": refined_feat,
            "affinity": affinity, 
            "k_idx": k_idx,
            "input_jafar_feat": jafar_input, 
            "target": targets,
            "prototypes": self.prototypes
        }

        if self.criteria is not None:
            output_dict['loss'] = self.criteria(output_dict)
            
        return output_dict