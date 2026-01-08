import torch
import torch.nn as nn
import torch.nn.functional as F
import pointops
import inspect
from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

# =========================================================================
# 0. GBlobs Calculation Utilities (Optimized)
# =========================================================================
def compute_covariance_features(features, knn_indices, k=16):
    """
    Compute Covariance Matrix (GBlobs) efficiently.
    features: [B, N, C]
    knn_indices: [B, N, k]
    """
    B, N, C = features.shape
    
    # Create batch indices broadcasting to neighbor dimension
    batch_idx = torch.arange(B, device=features.device).view(B, 1, 1).expand(-1, N, k)
    
    # Flatten features for gathering
    feat_flat = features.view(B*N, C)
    
    # Calculate flattened indices: neighbor_idx + batch_offset
    idx_flat = knn_indices.view(B, N, k) + (batch_idx * N)
    idx_flat = idx_flat.view(-1)
    
    # Gather neighbors: [B, N, k, C]
    neighbors = feat_flat[idx_flat].view(B, N, k, C)

    # Force Float32 calculation to prevent FP16 overflow during x*x
    dtype_backup = features.dtype
    neighbors = neighbors.float()
    
    # Centering: subtract local mean
    local_mean = neighbors.mean(dim=2, keepdim=True) 
    centered = neighbors - local_mean 
    
    # Covariance: (X^T * X) / (k-1)
    # Transpose for matrix multiplication: [B, N, C, k] @ [B, N, k, C] -> [B, N, C, C]
    centered_t = centered.transpose(2, 3)
    cov = torch.matmul(centered_t, centered) / (k - 1 + 1e-6)
    
    # Flatten covariance matrix: [B, N, C*C]
    cov_flat = cov.view(B, N, C*C)
    return cov_flat

def compute_lean_gblobs(xyz, k=16, knn_idx=None, scale=10.0):
    """
    Compute ONLY Geometric GBlobs (9-dim).
    
    Args:
        xyz: [B, N, 3] Coordinates.
        k: Neighbor count.
        knn_idx: [B, N, k] Pre-computed KNN indices (local index 0..N-1).
        scale: Scaling factor for numerical stability.
    """
    B, N, _ = xyz.shape
    
    # Use pre-computed KNN indices if provided to save computation
    if knn_idx is None:
        xyz_flat = xyz.view(-1, 3).contiguous()
        offset = torch.arange(1, B + 1, dtype=torch.int32, device=xyz.device) * N
        idx_flat = pointops.knn_query(k, xyz_flat, offset)[0].long()
        batch_start = (torch.arange(B, device=xyz.device) * N).view(B, 1, 1)
        knn_idx = idx_flat.view(B, N, k) - batch_start

    # Compute covariance on SCALED coordinates for stability
    # Note: knn_idx is valid for both scaled and unscaled xyz as relative order is preserved
    geo_blobs = compute_covariance_features(xyz * scale, knn_idx, k)
    return geo_blobs # [B, N, 9]

# =========================================================================
# 1. DecoupledPointJAFAR (Optimized: Accepts pre-computed KNN)
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
        """
        Gather features using local indices.
        tensor: [B, C, N]
        idx: [B, N, K] (values in 0..N-1)
        """
        B, C, N = tensor.shape
        _, _, K = idx.shape
        tensor_flat = tensor.transpose(1, 2).contiguous().view(B * N, C)
        # Add batch offsets to convert local indices to global flattened indices
        batch_offset = torch.arange(B, device=tensor.device).view(B, 1, 1) * N
        flat_idx = (idx + batch_offset).view(-1)
        val = tensor_flat[flat_idx].view(B, N, K, C).permute(0, 3, 1, 2)
        return val

    def forward(self, xyz, jafar_feat, sem_feat, knn_idx=None):
        """
        Args:
            xyz: [B, N, 3]
            jafar_feat: [B, N, C_geo]
            sem_feat: [B, N, C_sem]
            knn_idx: [B, N, k] Pre-computed KNN indices (optional)
        """
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

        # 2. KNN (Compute if not provided)
        if knn_idx is None:
            xyz_flat = xyz.view(-1, 3).contiguous()
            offset = torch.arange(1, B + 1, dtype=torch.int32, device=xyz.device) * N
            k_idx_flat = pointops.knn_query(self.k, xyz_flat, offset)[0].long()
            batch_start = (torch.arange(B, device=xyz.device) * N).view(B, 1, 1)
            knn_idx = k_idx_flat.view(B, N, self.k) - batch_start
        
        # 3. Attention
        K_g = self._gather_val_efficient(K, knn_idx)
        xyz_g = self._gather_val_efficient(xyz_t, knn_idx)
        V_g = self._gather_val_efficient(V, knn_idx)
        
        rel_pos = xyz_t.unsqueeze(-1) - xyz_g
        pos_enc = self.rel_pos_mlp(rel_pos)
        
        attn_logits = torch.sum(Q.unsqueeze(-1) * (K_g + pos_enc), dim=1) / (self.qk_dim ** 0.5)
        affinity = self.softmax(attn_logits)
        
        # 4. Refine
        refined_feat = torch.sum(affinity.unsqueeze(1) * V_g, dim=-1)
        refined_feat = refined_feat + V 
        
        refined_feat_flat = refined_feat.transpose(1, 2).contiguous().view(-1, self.qk_dim)
        logits = self.cls_head(refined_feat_flat)
        
        return logits, affinity, knn_idx, refined_feat_flat, bdy_logits

# =========================================================================
# 2. GeoPTV3 Main Model (Optimized)
# =========================================================================
@MODELS.register_module()
class GeoPTV3(nn.Module):
    def __init__(self, 
                 backbone_ptv3_cfg, 
                 geo_input_dim=6,  # e.g., 6 for XYZ+RGB, 4 for XYZ+Density
                 num_classes=13,
                 num_points=80000,
                 geo_scale=10.0, 
                 criteria=None):
        super().__init__()
        
        # 1. Semantic Stream: PTv3
        valid_params = inspect.signature(PointTransformerV3.__init__).parameters
        clean_cfg = {k: v for k, v in backbone_ptv3_cfg.items() if k in valid_params}
        self.sem_stream = PointTransformerV3(**clean_cfg)
        
        self.ptv3_in_channels = backbone_ptv3_cfg.get("in_channels", 6)
        
        dec_channels = backbone_ptv3_cfg.get('dec_channels', [48, 96, 192, 384])
        self.sem_feat_dim = dec_channels[0]
        self.aux_head = nn.Linear(self.sem_feat_dim, num_classes)
        
        # 2. Geometric Stream: JAFAR
        self.num_points = num_points
        self.geo_scale = geo_scale 
        
        # Dynamic Dimension Logic:
        # Base Geometry: XYZ -> GBlobs (9 dims)
        # Extra Features: Input Dim - 3 (XYZ) -> Extra Feats (e.g., RGB=3, Density=1)
        self.extra_feat_dim = max(0, geo_input_dim - 3)
        self.real_geo_dim = 9 + self.extra_feat_dim
        
        print(f"[GeoPTV3] Lean Mode: JAFAR Input Dim = {self.real_geo_dim} "
              f"(9 GeoGBlobs + {self.extra_feat_dim} Extra Feats)")
        print(f"[GeoPTV3] Geometry Scale Factor = {self.geo_scale}")
        
        self.geo_stream = DecoupledPointJAFAR(
            qk_dim=64, 
            k=16, 
            input_geo_dim=self.real_geo_dim, # Pass the dynamic dimension
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
        import torch.distributed as dist
        
        with torch.no_grad():
            for c in range(self.aux_head.out_features):
                mask = (labels == c)
                
                # Calculate local sum and count
                if mask.sum() > 0:
                    local_sum = features[mask].sum(0)
                    local_count = mask.sum().float()
                else:
                    local_sum = torch.zeros(features.shape[1], device=features.device)
                    local_count = torch.tensor(0.0, device=features.device)
                
                # Synchronize across all GPUs
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
                
                # Update global prototypes
                if local_count > 0:
                    global_mean = local_sum / local_count
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * global_mean
                    self.proto_count[c] += 1

    def forward(self, input_dict):
        # -----------------------------------------------------------
        # A. PTv3 Prep & Batch Handling
        # -----------------------------------------------------------
        if "jafar_coord" in input_dict:
            j_coord = input_dict['jafar_coord']
            j_feat_raw = input_dict['jafar_feat']
        else:
            j_coord = input_dict['coord'].clone()
            j_feat_raw = input_dict['feat'].clone()
            
        if j_coord.dim() == 2:
            total_points = j_coord.shape[0]
            
            if self.training:
                # [Training] Strict Fixed Size Logic
                if "batch" in input_dict:
                    B_size = input_dict["batch"].max().item() + 1
                else:
                    B_size = total_points // self.num_points
                
                valid_len = B_size * self.num_points
                j_coord = j_coord[:valid_len].view(B_size, self.num_points, -1)
                j_feat_raw = j_feat_raw[:valid_len].view(B_size, self.num_points, -1)
            
            else:
                # [Test/Inference] Dynamic Size Handling
                # Support both Test (B=1) and Validation (B>1)
                if "batch" in input_dict:
                    B_size = input_dict["batch"].max().item() + 1
                else:
                    B_size = 1
                
                # Attempt Reshape to [B, N, C]
                # Only valid if total points are divisible by Batch Size
                if total_points % B_size == 0:
                    points_per_batch = total_points // B_size
                    j_coord = j_coord.view(B_size, points_per_batch, -1)
                    j_feat_raw = j_feat_raw.view(B_size, points_per_batch, -1)
                else:
                    # Fallback for irregular batch sizes (force B=1 to avoid crash)
                    # Note: This might cause cross-room KNN issues if multiple rooms are packed
                    B_size = 1
                    j_coord = j_coord.view(1, total_points, -1)
                    j_feat_raw = j_feat_raw.view(1, total_points, -1)

                # Ensure PTv3 input has batch index
                if "batch" not in input_dict:
                    input_dict["batch"] = torch.zeros(total_points, device=j_coord.device, dtype=torch.long)
        else:
            B_size = j_coord.shape[0]

        ptv3_input = {}
        raw_coord = input_dict["coord"]
        raw_feat = input_dict.get("ptv3_feat", input_dict.get("feat"))
        raw_grid = input_dict.get("grid_coord")
        
        # Flatten for PTv3
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
                # Dynamic handling: Calculate N based on actual flat_coord size
                # This fixes the mismatch between 80000 and 149393 during test
                current_N = flat_coord.shape[0] // B_size
                ptv3_input["batch"] = torch.arange(B_size, device=raw_coord.device).repeat_interleave(current_N)

        ptv3_input["coord"] = flat_coord
        ptv3_input["feat"] = flat_feat
        
        if self.ptv3_in_channels == 6 and flat_feat.shape[1] == 3:
            ptv3_input["feat"] = torch.cat([flat_feat, flat_coord], dim=1)
        
        if flat_grid is None:
            ptv3_input["grid_coord"] = (flat_coord / 0.02).int()
        else:
            ptv3_input["grid_coord"] = flat_grid

        # -----------------------------------------------------------
        # B. Stage I: PTv3 Forward (Optimized: No Restore needed)
        # -----------------------------------------------------------
        sem_feat_sparse = self.sem_stream(ptv3_input).feat 
        aux_logits = self.aux_head(sem_feat_sparse) 
        
        # -----------------------------------------------------------
        # C. Feature Assembly (Shared KNN Optimization)
        # -----------------------------------------------------------
        # Use dynamic shape [-1] instead of self.num_points
        sem_feat_dense = sem_feat_sparse.view(B_size, -1, sem_feat_sparse.shape[-1])
        
        # Get actual point count (80000 during train, dynamic during test)
        N_current = j_coord.shape[1]

        # [Optimization] Compute KNN once and reuse
        j_coord_flat = j_coord.view(-1, 3).contiguous()
        
        # Calculate offset using ACTUAL N_current
        j_offset = torch.arange(1, B_size + 1, dtype=torch.int32, device=j_coord.device) * N_current
        
        k_neighbors = 16
        idx_flat = pointops.knn_query(k_neighbors, j_coord_flat, j_offset)[0].long()
        
        # Calculate batch start using ACTUAL N_current
        batch_start = (torch.arange(B_size, device=j_coord.device) * N_current).view(B_size, 1, 1)
        
        # Reshape using ACTUAL N_current
        shared_knn_idx = idx_flat.view(B_size, N_current, k_neighbors) - batch_start
        
        # Compute GBlobs (9 dims)
        geo_blobs = compute_lean_gblobs(j_coord, k=k_neighbors, knn_idx=shared_knn_idx, scale=self.geo_scale) 
        
        # Concatenate Extra Features dynamically (e.g., RGB or Density)
        if self.extra_feat_dim > 0:
            # Safely slice the required number of feature channels
            extra_feat = j_feat_raw[:, :, :self.extra_feat_dim]
            jafar_input = torch.cat([geo_blobs, extra_feat], dim=-1)
        else:
            jafar_input = geo_blobs
        
        # -----------------------------------------------------------
        # D. PointJAFAR Refinement (Reusing shared_knn_idx)
        # -----------------------------------------------------------
        refined_logits, affinity, k_idx, refined_feat, bdy_logits = self.geo_stream(
            xyz=j_coord,
            jafar_feat=jafar_input,
            sem_feat=sem_feat_dense,
            knn_idx=shared_knn_idx 
        )
        
        # -----------------------------------------------------------
        # E. Outputs
        # -----------------------------------------------------------
        if "segment" in input_dict:
            targets = input_dict['segment'].view(-1)
        else:
            targets = None
        
        # Only update prototypes if we have targets and are training
        if self.training and targets is not None:
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

        # Calculate loss only if targets exist and criteria is set
        if self.criteria is not None and targets is not None:
            output_dict['loss'] = self.criteria(output_dict)
        elif self.criteria is not None:
            # For logging purposes in case loss is expected but not computable
            output_dict['loss'] = torch.tensor(0.0, device=refined_logits.device)
            
        return output_dict