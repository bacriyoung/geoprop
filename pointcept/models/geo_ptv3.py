import torch
import torch.nn as nn
import torch.nn.functional as F
import pointops
import inspect
from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

# =========================================================================
# 1. DecoupledPointJAFAR (V9.2: AMP Safe & Bug Fixed)
# =========================================================================
class DecoupledPointJAFAR(nn.Module):
    def __init__(self, qk_dim=64, k=16, input_geo_dim=6, num_classes=13): 
        super().__init__()
        self.qk_dim = qk_dim
        self.k = k
        self.input_geo_dim = input_geo_dim 

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
        
        # 🟢 [AMP FIX] 移除了最后的 Sigmoid，改为输出 Logits
        # 以配合 Loss 中的 BCEWithLogitsLoss，防止混合精度报错
        self.bdy_head = nn.Sequential(
            nn.Conv1d(qk_dim, 32, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 1) # No Sigmoid here!
        )
        
        self.rel_pos_mlp = nn.Sequential(
            nn.Conv2d(3, qk_dim, 1), nn.BatchNorm2d(qk_dim), nn.ReLU(),
            nn.Conv2d(qk_dim, qk_dim, 1)
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

    def forward(self, xyz, feat):
        B, N, C = feat.shape
        feat_t = feat.transpose(1, 2).contiguous()
        xyz_t = xyz.transpose(1, 2).contiguous()
        
        # 1. Encode
        geom_feat = self.geom_encoder(feat_t)
        
        # 2. Modulate
        if C == 6: 
            scale = self.scale_conv(feat_t)
            shift = self.shift_conv(feat_t)
            geom_feat = geom_feat * (scale + 1) + shift

        # 3. Boundary Logits (Pre-Sigmoid)
        bdy_logits = self.bdy_head(geom_feat) 
        
        # 4. Attention Prep
        Q = self.sem_query(geom_feat)
        K = self.sem_key(geom_feat)
        
        # 5. KNN Search 
        xyz_flat = xyz.view(-1, 3).contiguous()
        offset = torch.arange(1, B + 1, dtype=torch.int32, device=xyz.device) * N
        
        # 🟢 [BUG FIX] pointops 返回 (idx, dist) 元组，取 [0]
        k_idx_flat = pointops.knn_query(self.k, xyz_flat, offset)[0].long()
        
        batch_start = (torch.arange(B, device=xyz.device) * N).view(B, 1, 1)
        k_idx = k_idx_flat.view(B, N, self.k) - batch_start
        
        # 6. Gather
        K_g = self._gather_val_efficient(K, k_idx)
        xyz_g = self._gather_val_efficient(xyz_t, k_idx)
        val_g = self._gather_val_efficient(geom_feat, k_idx)
        
        # 7. RPE
        rel_pos = xyz_t.unsqueeze(-1) - xyz_g
        pos_enc = self.rel_pos_mlp(rel_pos)
        
        # 8. Attention
        attn_logits = torch.sum(Q.unsqueeze(-1) * (K_g + pos_enc), dim=1) / (self.qk_dim ** 0.5)
        affinity = self.softmax(attn_logits)
        
        # 9. Propagate
        out_feat = torch.sum(affinity.unsqueeze(1) * val_g, dim=-1)
        out_feat_flat = out_feat.transpose(1, 2).contiguous().view(-1, self.qk_dim)
        logits = self.cls_head(out_feat_flat)
        
        return logits, affinity, k_idx, out_feat_flat, bdy_logits

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
        print("\n" + "="*60)
        print(f"🚀🚀 [[ GeoPTV3 V9.2 (AMP Safe) | Points: {num_points} ]] 🚀🚀")
        print("="*60 + "\n")
        
        valid_params = inspect.signature(PointTransformerV3.__init__).parameters
        clean_cfg = {k: v for k, v in backbone_ptv3_cfg.items() if k in valid_params}
        self.sem_stream = PointTransformerV3(**clean_cfg)
        
        dec_channels = backbone_ptv3_cfg.get('dec_channels', [48, 96, 192, 384])
        self.sem_feat_dim = dec_channels[0]
        self.sem_head = nn.Linear(self.sem_feat_dim, num_classes)
        
        self.num_points = num_points
        self.geo_stream = DecoupledPointJAFAR(
            qk_dim=64, 
            k=16, 
            input_geo_dim=geo_input_dim,
            num_classes=num_classes
        )
        
        self.register_buffer("prototypes", torch.zeros(num_classes, self.sem_feat_dim))
        self.register_buffer("proto_count", torch.zeros(num_classes))
        self.momentum = 0.99
        self.geo_proj = nn.Linear(64, self.sem_feat_dim)

        if criteria is not None:
            self.criteria = LOSSES.build(criteria)
        else:
            self.criteria = None

    def update_prototypes(self, features, labels):
        with torch.no_grad():
            for c in range(self.sem_head.out_features):
                mask = (labels == c)
                if mask.sum() > 0:
                    curr_proto = features[mask].mean(0)
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * curr_proto
                    self.proto_count[c] += 1

    def forward(self, input_dict):
        # A. Data Pre-processing
        if "jafar_coord" in input_dict:
            j_coord = input_dict['jafar_coord']
            j_feat = input_dict['jafar_feat']
        else:
            j_coord = input_dict['coord'].clone()
            j_feat = input_dict['feat'].clone()
        
        if j_coord.dim() == 2:
            total_points = j_coord.shape[0]
            if "batch" in input_dict:
                B_size = input_dict["batch"].max().item() + 1
            else:
                B_size = total_points // self.num_points
            valid_len = B_size * self.num_points
            j_coord = j_coord[:valid_len].view(B_size, self.num_points, -1)
            j_feat = j_feat[:valid_len].view(B_size, self.num_points, -1)
        else:
            B_size = j_coord.shape[0]

        # B. PTv3 Input Construction
        ptv3_input = {}
        raw_coord = input_dict["coord"]
        raw_feat = input_dict.get("ptv3_feat", input_dict.get("feat"))
        raw_grid = input_dict.get("grid_coord")
        
        if raw_coord.dim() == 3: 
            ptv3_input["coord"] = raw_coord.reshape(-1, 3).contiguous()
            ptv3_input["feat"] = raw_feat.reshape(-1, raw_feat.shape[-1]).contiguous()
            if raw_grid is not None:
                ptv3_input["grid_coord"] = raw_grid.reshape(-1, 3).contiguous().int()
            N = raw_coord.shape[1]
            ptv3_input["batch"] = torch.arange(B_size, device=raw_coord.device).repeat_interleave(N)
        else:
            ptv3_input["coord"] = raw_coord
            ptv3_input["feat"] = raw_feat
            ptv3_input["grid_coord"] = raw_grid
            if "batch" in input_dict:
                ptv3_input["batch"] = input_dict["batch"]
            else:
                ptv3_input["batch"] = torch.arange(B_size, device=raw_coord.device).repeat_interleave(self.num_points)

        if ptv3_input["feat"].shape[-1] == 3:
            ptv3_input["feat"] = torch.cat([ptv3_input["coord"], ptv3_input["feat"]], dim=1)
        if ptv3_input.get("grid_coord") is None:
            ptv3_input["grid_coord"] = (ptv3_input["coord"] / 0.02).int()

        # C. Forward Sem
        sem_feat_sparse = self.sem_stream(ptv3_input).feat 
        sem_logits = self.sem_head(sem_feat_sparse)
        
        if not self.training and "segment" not in input_dict:
            return dict(seg_logits=sem_logits)

        # D. Forward Geo
        geo_logits, affinity, k_idx, geo_feat_raw, bdy_logits = self.geo_stream(j_coord, j_feat)
        
        sem_feat_dense = sem_feat_sparse.view(B_size, self.num_points, -1)
        geo_feat_proj = self.geo_proj(geo_feat_raw)
        
        targets = input_dict['segment'].view(-1)
        
        if self.training:
            valid_mask = (targets != 255)
            if valid_mask.sum() > 0:
                self.update_prototypes(sem_feat_sparse[valid_mask].detach(), targets[valid_mask])

        output_dict = {
            "seg_logits": sem_logits,
            "sem_logits": sem_logits,
            "geo_logits": geo_logits.view(-1, 13),
            "sem_feat_dense": sem_feat_dense,
            "geo_feat_dense": geo_feat_proj,
            "affinity": affinity, 
            "k_idx": k_idx,
            "bdy_logits": bdy_logits, # 🟢 Logits, not Prob
            "input_jafar_feat": j_feat, 
            "target": targets,
            "prototypes": self.prototypes,
            "epoch": input_dict.get("epoch", 0)
        }

        if self.criteria is not None:
            output_dict['loss'] = self.criteria(output_dict)
            
        return output_dict