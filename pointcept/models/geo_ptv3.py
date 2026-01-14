import torch
import torch.nn as nn
import torch.nn.functional as F
import pointops
import inspect
from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

# =========================================================================
# 0. GBlobs Utilities
# =========================================================================
def compute_covariance_features(features, knn_indices, k=16):
    b_dim, n_dim, c_dim = features.shape
    batch_idx = torch.arange(b_dim, device=features.device).view(b_dim, 1, 1).expand(-1, n_dim, k)
    feat_flat = features.view(b_dim*n_dim, c_dim)
    idx_flat = knn_indices.view(b_dim, n_dim, k) + (batch_idx * n_dim)
    idx_flat = idx_flat.view(-1)
    neighbors = feat_flat[idx_flat].view(b_dim, n_dim, k, c_dim)

    dtype_backup = features.dtype
    neighbors = neighbors.float()
    local_mean = neighbors.mean(dim=2, keepdim=True) 
    centered = neighbors - local_mean 
    centered_t = centered.transpose(2, 3)
    cov = torch.matmul(centered_t, centered) / (k - 1 + 1e-6)
    cov_flat = cov.view(b_dim, n_dim, c_dim*c_dim)
    return cov_flat.to(dtype_backup)

def compute_lean_gblobs(xyz, k=16, knn_idx=None, scale=10.0):
    b_dim, n_dim, _ = xyz.shape
    if knn_idx is None:
        xyz_flat = xyz.view(-1, 3).contiguous()
        offset = torch.arange(1, b_dim + 1, dtype=torch.int32, device=xyz.device) * n_dim
        idx_flat = pointops.knn_query(k, xyz_flat, offset)[0].long()
        batch_start = (torch.arange(b_dim, device=xyz.device) * n_dim).view(b_dim, 1, 1)
        knn_idx = idx_flat.view(b_dim, n_dim, k) - batch_start
    geo_blobs = compute_covariance_features(xyz * scale, knn_idx, k)
    return geo_blobs 

# =========================================================================
# 1. DecoupledPointJAFAR
# =========================================================================
class DecoupledPointJAFAR(nn.Module):
    def __init__(self, qk_dim=64, k=16, input_geo_dim=12, sem_dim=192, num_classes=13): 
        super().__init__()
        self.qk_dim = qk_dim
        self.k = k
        self.input_geo_dim = input_geo_dim 
        self.sem_dim = sem_dim 

        self.geom_encoder = nn.Sequential(
            nn.Conv1d(self.input_geo_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), 
            nn.ReLU(),
            nn.Conv1d(qk_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), 
            nn.ReLU()
        )
        self.val_proj = nn.Sequential(
            nn.Conv1d(sem_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), 
            nn.ReLU()
        )
        self.geo_query = nn.Conv1d(qk_dim, qk_dim, 1)
        self.geo_key = nn.Conv1d(qk_dim, qk_dim, 1)
        self.rel_pos_mlp = nn.Sequential(
            nn.Conv2d(7, qk_dim, 1), 
            nn.GroupNorm(8, qk_dim), 
            nn.ReLU(),
            nn.Conv2d(qk_dim, qk_dim, 1)
        )
        self.bdy_head = nn.Sequential(
            nn.Conv1d(qk_dim, 32, 1), 
            nn.GroupNorm(4, 32),     
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.cls_head = nn.Linear(qk_dim, num_classes)

        self.rec_head = nn.Sequential(
            nn.Linear(qk_dim, qk_dim),
            nn.LayerNorm(qk_dim),
            nn.ReLU(),
            nn.Linear(qk_dim, 6) # Output: 3 dims (XYZ) + 3 dims (RGB)
        )

    def _gather_val_efficient(self, tensor, idx):
        b_dim, c_dim, n_dim = tensor.shape
        _, _, k_dim = idx.shape
        tensor_flat = tensor.transpose(1, 2).contiguous().view(b_dim * n_dim, c_dim)
        batch_offset = torch.arange(b_dim, device=tensor.device).view(b_dim, 1, 1) * n_dim
        flat_idx = (idx + batch_offset).view(-1)
        val = tensor_flat[flat_idx].view(b_dim, n_dim, k_dim, c_dim).permute(0, 3, 1, 2)
        return val

    def forward(self, xyz, jafar_feat, sem_feat, knn_idx=None):
        b_dim, n_dim, _ = jafar_feat.shape
        jafar_feat_t = jafar_feat.transpose(1, 2).contiguous()
        sem_feat_t = sem_feat.transpose(1, 2).contiguous()
        xyz_t = xyz.transpose(1, 2).contiguous()
        
        geom_emb = self.geom_encoder(jafar_feat_t)
        bdy_logits = self.bdy_head(geom_emb) 
        Q = self.geo_query(geom_emb)
        K = self.geo_key(geom_emb)
        V = self.val_proj(sem_feat_t)

        if knn_idx is None:
            xyz_flat = xyz.view(-1, 3).contiguous()
            offset = torch.arange(1, b_dim + 1, dtype=torch.int32, device=xyz.device) * n_dim
            k_idx_flat = pointops.knn_query(self.k, xyz_flat, offset)[0].long()
            batch_start = (torch.arange(b_dim, device=xyz.device) * n_dim).view(b_dim, 1, 1)
            knn_idx = k_idx_flat.view(b_dim, n_dim, self.k) - batch_start
        
        K_g = self._gather_val_efficient(K, knn_idx)
        xyz_g = self._gather_val_efficient(xyz_t, knn_idx)
        V_g = self._gather_val_efficient(V, knn_idx)
        
        # [MODIFIED & SECURED] Full Explicit Geometric Encoding
        # ------------------------------------------------------------------
        # 1. Force FP32: Crucial for AMP stability to prevent overflow/underflow
        xyz_t_f32 = xyz_t.float()
        xyz_g_f32 = xyz_g.float()
        
        # 2. Relative Coordinates
        rel_diff = xyz_t_f32.unsqueeze(-1) - xyz_g_f32
        
        # 3. Euclidean Distance
        # Calculation in FP32 preventing underflow
        sq_sum = torch.sum(rel_diff ** 2, dim=1, keepdim=True)
        
        # [OPTIMIZATION] Lower epsilon to 1e-10.
        # sqrt(1e-10) = 1e-5. This allows distinguishing very close points
        # while keeping self-loop gradients safe.
        rel_dist = torch.sqrt(sq_sum + 1e-10)
        
        # 4. Direction Cosines / Angles
        # [SAFETY] Clamp ensures denominator >= 1e-5. Max gradient ~ 100,000 (Safe for FP32).
        rel_dist_safe = torch.clamp(rel_dist, min=1e-5)
        rel_direction = rel_diff / rel_dist_safe
        
        # 5. Concatenate & Cast back
        rel_geo_feat = torch.cat([rel_diff, rel_dist, rel_direction], dim=1)
        
        # [SAFETY CHECK] Replace any accidental NaNs/Infs with 0 before casting back
        if torch.isnan(rel_geo_feat).any() or torch.isinf(rel_geo_feat).any():
             rel_geo_feat = torch.nan_to_num(rel_geo_feat, nan=0.0, posinf=1.0, neginf=-1.0)

        rel_geo_feat = rel_geo_feat.type_as(xyz)
        # ------------------------------------------------------------------
        
        # 6. Feed to MLP
        pos_enc = self.rel_pos_mlp(rel_geo_feat)
        
        attn_logits = torch.sum(Q.unsqueeze(-1) * (K_g + pos_enc), dim=1) / (self.qk_dim ** 0.5)

        affinity = torch.softmax(attn_logits.float(), dim=-1).type_as(attn_logits)
        
        refined_feat = torch.sum(affinity.unsqueeze(1) * V_g, dim=-1)
        refined_feat = refined_feat + V 
        refined_feat_flat = refined_feat.transpose(1, 2).contiguous().view(-1, self.qk_dim)
        logits = self.cls_head(refined_feat_flat)
        rec_phys = self.rec_head(refined_feat_flat)
        return logits, affinity, knn_idx, refined_feat_flat, bdy_logits, rec_phys

# =========================================================================
# 2. GeoPTV3 Main Model
# =========================================================================
@MODELS.register_module()
class GeoPTV3(nn.Module):
    def __init__(self, backbone_ptv3_cfg, geo_input_dim=6, num_classes=13,
                 num_points=80000, geo_scale=10.0, criteria=None):
        super().__init__()
        
        valid_params = inspect.signature(PointTransformerV3.__init__).parameters
        clean_cfg = {k: v for k, v in backbone_ptv3_cfg.items() if k in valid_params}
        self.sem_stream = PointTransformerV3(**clean_cfg)
        self.ptv3_in_channels = backbone_ptv3_cfg.get("in_channels", 6)
        
        dec_channels = backbone_ptv3_cfg.get('dec_channels', [48, 96, 192, 384])
        self.sem_feat_dim = dec_channels[0]
        self.aux_head = nn.Linear(self.sem_feat_dim, num_classes)
        
        self.num_points = num_points
        self.geo_scale = geo_scale 
        self.extra_feat_dim = max(0, geo_input_dim - 3)
        self.real_geo_dim = 9 + self.extra_feat_dim
        
        print(f"[GeoPTV3] Lean Mode: JAFAR Input Dim = {self.real_geo_dim}")
        
        self.geo_stream = DecoupledPointJAFAR(
            qk_dim=64, k=16, 
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
        import torch.distributed as dist
        with torch.no_grad():
            for c in range(self.aux_head.out_features):
                mask = (labels == c)
                if mask.sum() > 0:
                    local_sum = features[mask].sum(0)
                    local_count = mask.sum().float()
                else:
                    local_sum = torch.zeros(features.shape[1], device=features.device)
                    local_count = torch.tensor(0.0, device=features.device)
                
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
                
                if local_count > 0:
                    global_mean = local_sum / local_count
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * global_mean
                    self.proto_count[c] += 1

    def forward(self, input_dict):
        # =========================================================================
        # 1. Sliding Window Validation (Fixed for 2-GPU DDP Shape Mismatch)
        # =========================================================================
        if "fragment_list" in input_dict:
            # Unpack Batch (B=1)
            fragment_list = input_dict["fragment_list"][0]
            
            # Force flatten Target to (N,) for internal Loss calculation
            full_segment = input_dict["segment"].view(-1)
            
            # [CRITICAL CHANGE] Do NOT overwrite input_dict["segment"] in place.
            # In DDP, the Evaluator holds a reference to the original batch (1, N),
            # modifying it here creates a mismatch if we flatten it to (N,).
            # input_dict["segment"] = full_segment  <-- REMOVED
            
            num_points_total = full_segment.shape[0]
            num_classes = self.aux_head.out_features
            
            # Standard GPU Device
            device = torch.cuda.current_device()

            # Initialize container on GPU
            full_logits = torch.zeros((num_points_total, num_classes), device=device)
            full_counts = torch.zeros((num_points_total, 1), device=device)

            for fragment in fragment_list:
                # Move fragment data to GPU
                for key in fragment.keys():
                    if isinstance(fragment[key], torch.Tensor):
                        fragment[key] = fragment[key].to(device)
                
                # Inference
                chunk_output = self.forward(fragment)
                
                # Probabilities
                chunk_logits = torch.softmax(chunk_output["seg_logits"], dim=-1)
                global_idx = fragment["index"].long()
                
                # Accumulate
                full_logits.index_add_(0, global_idx, chunk_logits)
                full_counts.index_add_(0, global_idx, torch.ones_like(chunk_logits[:, :1]))

            # Normalize
            full_logits /= full_counts.clamp(min=1.0)
            
            # Ensure target is on GPU for Loss
            if full_segment.device != device:
                full_segment = full_segment.to(device)

            val_loss = F.nll_loss(
                torch.log(full_logits.clamp(min=1e-6)), 
                full_segment.long(), 
                ignore_index=255
            )
            
            # [CRITICAL FIX] Reshape logits to match Evaluator's expected input shape
            # Evaluator has target shape (1, N). It does pred = logits.max(1)[1].
            # We need logits to be (1, C, N) so that max(1) yields pred of shape (1, N).
            # Current full_logits is (N, C).
            # Permute (N, C) -> (C, N) -> Unsqueeze (1, C, N)
            logits_eval = full_logits.permute(1, 0).unsqueeze(0)
            
            output_dict = {
                "seg_logits": logits_eval,  # (1, C, N) matches Evaluator
                "target": full_segment, 
                "loss": val_loss
            }
            return output_dict

        # =========================================================================
        # 2. Training / Single Chunk Forward (Standard v5.0)
        # =========================================================================
        if "jafar_coord" in input_dict:
            j_coord = input_dict['jafar_coord']
            j_feat_raw = input_dict['jafar_feat']
        else:
            j_coord = input_dict['coord'].clone()
            j_feat_raw = input_dict['feat'].clone()
            
        if j_coord.dim() == 2:
            total_points = j_coord.shape[0]
            if self.training:
                if "batch" in input_dict:
                    batch_size_val = input_dict["batch"].max().item() + 1
                else:
                    batch_size_val = total_points // self.num_points
                valid_len = batch_size_val * self.num_points
                j_coord = j_coord[:valid_len].view(batch_size_val, self.num_points, -1)
                j_feat_raw = j_feat_raw[:valid_len].view(batch_size_val, self.num_points, -1)
            else:
                if "batch" in input_dict:
                    batch_size_val = input_dict["batch"].max().item() + 1
                else:
                    batch_size_val = 1
                
                if total_points % batch_size_val == 0:
                    points_per_batch = total_points // batch_size_val
                    j_coord = j_coord.view(batch_size_val, points_per_batch, -1)
                    j_feat_raw = j_feat_raw.view(batch_size_val, points_per_batch, -1)
                else:
                    batch_size_val = 1
                    j_coord = j_coord.view(1, total_points, -1)
                    j_feat_raw = j_feat_raw.view(1, total_points, -1)

                if "batch" not in input_dict:
                    input_dict["batch"] = torch.zeros(total_points, device=j_coord.device, dtype=torch.long)
        else:
            batch_size_val = j_coord.shape[0]

        ptv3_input = {}
        raw_coord = input_dict["coord"]
        raw_feat = input_dict.get("ptv3_feat", input_dict.get("feat"))
        raw_grid = input_dict.get("grid_coord")
        
        if raw_coord.dim() == 3: 
            flat_coord = raw_coord.reshape(-1, 3).contiguous()
            flat_feat = raw_feat.reshape(-1, raw_feat.shape[-1]).contiguous()
            if raw_grid is not None:
                flat_grid = raw_grid.reshape(-1, 3).contiguous().int()
            N_total = raw_coord.shape[1]
            ptv3_input["batch"] = torch.arange(batch_size_val, device=raw_coord.device).repeat_interleave(N_total)
        else:
            flat_coord = raw_coord
            flat_feat = raw_feat
            flat_grid = raw_grid
            if "batch" in input_dict:
                ptv3_input["batch"] = input_dict["batch"]
            else:
                current_N = flat_coord.shape[0] // batch_size_val
                ptv3_input["batch"] = torch.arange(batch_size_val, device=raw_coord.device).repeat_interleave(current_N)

        ptv3_input["coord"] = flat_coord
        ptv3_input["feat"] = flat_feat
        
        if self.ptv3_in_channels == 6 and flat_feat.shape[1] == 3:
            ptv3_input["feat"] = torch.cat([flat_feat, flat_coord], dim=1)
        
        if flat_grid is None:
            ptv3_input["grid_coord"] = (flat_coord / 0.02).int()
        else:
            ptv3_input["grid_coord"] = flat_grid

        sem_feat_sparse = self.sem_stream(ptv3_input).feat 
        aux_logits = self.aux_head(sem_feat_sparse) 
        
        sem_feat_dense = sem_feat_sparse.view(batch_size_val, -1, sem_feat_sparse.shape[-1])
        N_current = j_coord.shape[1]

        j_coord_flat = j_coord.view(-1, 3).contiguous()
        j_offset = torch.arange(1, batch_size_val + 1, dtype=torch.int32, device=j_coord.device) * N_current
        idx_flat = pointops.knn_query(16, j_coord_flat, j_offset)[0].long()
        batch_start = (torch.arange(batch_size_val, device=j_coord.device) * N_current).view(batch_size_val, 1, 1)
        shared_knn_idx = idx_flat.view(batch_size_val, N_current, 16) - batch_start
        
        geo_blobs = compute_lean_gblobs(j_coord, k=16, knn_idx=shared_knn_idx, scale=self.geo_scale) 
        
        if self.extra_feat_dim > 0:
            extra_feat = j_feat_raw[:, :, :self.extra_feat_dim]
            jafar_input = torch.cat([geo_blobs, extra_feat], dim=-1)
        else:
            jafar_input = geo_blobs
        
        target_phys = j_feat_raw.contiguous().view(-1, j_feat_raw.shape[-1])
        
        refined_logits, affinity, k_idx, refined_feat, bdy_logits, rec_phys = self.geo_stream(
            xyz=j_coord,
            jafar_feat=jafar_input,
            sem_feat=sem_feat_dense,
            knn_idx=shared_knn_idx 
        )
        
        if "segment" in input_dict:
            targets = input_dict['segment'].view(-1)
        else:
            targets = None
        
        if self.training and targets is not None:
            valid_mask = (targets != 255)
            if valid_mask.sum() > 0:
                self.update_prototypes(refined_feat[valid_mask].detach(), targets[valid_mask])
            else:
                # Create dummy empty tensors on the same device to keep DDP sync happy
                dummy_feat = torch.zeros((0, refined_feat.shape[-1]), device=refined_feat.device)
                dummy_label = torch.zeros((0,), dtype=torch.long, device=targets.device)
                self.update_prototypes(dummy_feat, dummy_label)

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
            "prototypes": self.prototypes,
            "rec_phys": rec_phys,  
            "target_phys": target_phys  
        }

        if self.criteria is not None and targets is not None:
            output_dict['loss'] = self.criteria(output_dict)
        elif self.criteria is not None:
            output_dict['loss'] = torch.tensor(0.0, device=refined_logits.device)
            
        return output_dict