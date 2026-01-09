import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch.distributed as dist
import pointops

from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point

# =========================================================================
# 1. VoxelJAFAR Module (The "True Reuse" Version)
# =========================================================================
class VoxelJAFAR(nn.Module):
    def __init__(self, 
                 geo_input_channels, # This matches PTv3 Stem Output Channels (e.g., 32)
                 sem_input_channels,
                 attn_dim=64, 
                 kernel_size=5, # Deprecated
                 search_radius=1, 
                 num_classes=13, 
                 use_rel_pos=True,
                 indice_key=None): # Deprecated
        super().__init__()
        self.use_rel_pos = use_rel_pos
        self.search_radius = search_radius
        self.attn_dim = attn_dim
        
        self.K_seq = (2 * search_radius + 1) ** 3
        
        # [TRUE REUSE]
        # We NO LONGER build a spconv layer here.
        # We NO LONGER use an MLP on coordinates.
        # We accept the PRE-COMPUTED features from PTv3 Backbone's Stem.
        # These features already contain RGB + Geometry info processed by Sparse Conv.
        self.geo_proj = nn.Linear(geo_input_channels, attn_dim, bias=False)
        
        self.geo_norm = nn.LayerNorm(attn_dim)
        self.geo_act = nn.ReLU(inplace=True)

        self.bdy_head = nn.Linear(attn_dim, 1)

        self.to_q = nn.Linear(attn_dim, attn_dim, bias=False)
        self.to_k = nn.Linear(attn_dim, attn_dim, bias=False)
        self.to_v = nn.Linear(sem_input_channels, attn_dim, bias=False) 
        
        if self.use_rel_pos:
            self.diameter = 2 * self.search_radius + 1
            self.pos_embedding = nn.Embedding(self.diameter ** 3, attn_dim)
        
        self.out_proj = nn.Linear(attn_dim, attn_dim)
        self.cls_head = nn.Linear(attn_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _cuda_grid_query(self, coords, batch_idx, device):
        # [Standard CUDA KNN - No changes]
        batch_size = batch_idx.max().item() + 1
        batch_counts = torch.bincount(batch_idx.int(), minlength=batch_size)
        offset = torch.cumsum(batch_counts, dim=0).int().to(device)
        neighbor_idx = pointops.knn_query(self.K_seq, coords.float(), offset)[0].long()
        
        M = coords.shape[0]
        neighbor_coords = coords[neighbor_idx.view(-1)].view(M, self.K_seq, 3)
        center_coords = coords.unsqueeze(1)
        rel_coords = neighbor_coords - center_coords 
        valid_mask = (rel_coords.abs().max(dim=-1)[0] <= self.search_radius + 0.1)
        
        if self.use_rel_pos:
            rel_int = rel_coords.long() + self.search_radius
            rel_int = rel_int.clamp(0, 2 * self.search_radius)
            pos_indices = rel_int[:, :, 0] * (self.diameter**2) + \
                          rel_int[:, :, 1] * (self.diameter) + \
                          rel_int[:, :, 2]
        else:
            pos_indices = None
        return neighbor_idx, valid_mask, pos_indices

    def forward(self, sp_structure, geo_feat_M, sem_feat_M):
        # geo_feat_M: This comes DIRECTLY from PTv3 Stem output [M, 32]
        
        # 1. Projection (Channel Alignment)
        # 32 -> 64
        Q_geo = self.geo_proj(geo_feat_M)
        Q_geo = self.geo_act(self.geo_norm(Q_geo))

        # 2. Boundary Prediction
        bdy_logits = self.bdy_head(Q_geo)

        # 3. Neighbor Search (CUDA)
        # Reuse coordinates from structure
        batch_idx = sp_structure.indices[:, 0]
        coords = sp_structure.indices[:, 1:]
        neighbor_idx, valid_mask, pos_indices = self._cuda_grid_query(coords, batch_idx, coords.device)
        K = self.K_seq

        # 4. Attention
        flat_idx = neighbor_idx.view(-1)
        
        K_feat = Q_geo[flat_idx].view(-1, K, self.attn_dim)
        V_feat = sem_feat_M[flat_idx].view(-1, K, sem_feat_M.shape[-1])
        
        Q_proj = self.to_q(Q_geo).unsqueeze(1)
        K_proj = self.to_k(K_feat)
        V_proj = self.to_v(V_feat)
        
        if self.use_rel_pos:
            P_pos = self.pos_embedding(pos_indices)
            K_proj = K_proj + P_pos
            
        attn_logits = torch.matmul(Q_proj, K_proj.transpose(-1, -2)) / (self.attn_dim ** 0.5)
        attn_logits = attn_logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)
        
        affinity = self.softmax(attn_logits)
        
        refined = torch.matmul(affinity, V_proj).squeeze(1)
        refined = refined + self.to_v(sem_feat_M)
        
        refined_feat = self.out_proj(refined)
        logits = self.cls_head(refined_feat)
        
        return logits, bdy_logits, affinity, refined_feat, neighbor_idx, valid_mask


# =========================================================================
# 2. GeoPTV3 (Capture Stem Features Logic)
# =========================================================================
@MODELS.register_module("GeoPTV3")
class GeoPTV3(PointTransformerV3):
    def __init__(self, 
                 geo_input_channels=6, # Placeholder
                 jafar_kernel_size=5, 
                 attn_search_radius=1, 
                 attn_dim=64,
                 use_rel_pos=True,
                 criteria=None,
                 num_classes=13,
                 **kwargs):

        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        self.dec_channels = kwargs.get('dec_channels', (64, 64, 128, 256))
        self.sem_feat_dim = self.dec_channels[0]
        self.aux_head = nn.Linear(self.sem_feat_dim, self.num_classes)
        
        # [AUTO DETECT] Stem Output Channels
        # Typically self.embedding.embed_channels holds the stem output dim
        stem_out_dim = self.embedding.embed_channels
        
        self.geo_stream = VoxelJAFAR(
            geo_input_channels=stem_out_dim, # Dynamic matching (e.g. 32)
            sem_input_channels=self.sem_feat_dim,
            attn_dim=attn_dim,
            kernel_size=jafar_kernel_size,
            search_radius=attn_search_radius,
            num_classes=self.num_classes,
            use_rel_pos=use_rel_pos,
            indice_key=None 
        )
        
        self.register_buffer("prototypes", torch.zeros(self.num_classes, attn_dim))
        self.register_buffer("proto_count", torch.zeros(self.num_classes))
        self.momentum = 0.99
        
        if criteria is not None:
            self.criteria = LOSSES.build(criteria)
        else:
            self.criteria = None

    def update_prototypes(self, features, labels):
        # ... (Keep unchanged) ...
        with torch.no_grad():
            for c in range(self.num_classes):
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
        point = Point(input_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        
        # --- 1. PTv3 Backbone ---
        point = self.embedding(point) 
        
        # [TRUE REUSE: CAPTURE STEM FEATURES]
        # At this exact moment, point.sparse_conv_feat holds the output of the Stem Sparse Conv.
        # It contains aggregated RGB + Geometry info.
        # We CLONE it because subsequent layers will overwrite point.sparse_conv_feat.features
        ptv3_sparse_structure = point.sparse_conv_feat
        geo_feat_M = ptv3_sparse_structure.features.clone() # [M, 32] -> To JAFAR
        
        point = self.enc(point)
        point = self.dec(point)
        
        sem_feat_N = point.feat 
        aux_logits = self.aux_head(sem_feat_N)
        
        # --- 2. Efficient N-to-M Mapping (Using Inverse Indices) ---
        if "inverse_indices" in point.keys():
            inverse_indices = point.inverse_indices
        else:
            raise RuntimeError("inverse_indices not found! Did you modify Embedding.forward?")

        M = ptv3_sparse_structure.features.shape[0]
        
        # Aggregate Semantic Features (N -> M)
        sem_feat_M = torch.zeros((M, sem_feat_N.shape[1]), device=sem_feat_N.device)
        count = torch.zeros((M, 1), device=sem_feat_N.device)
        
        sem_feat_M.index_add_(0, inverse_indices, sem_feat_N)
        count.index_add_(0, inverse_indices, torch.ones_like(count))
        
        count = count.clamp(min=1.0)
        sem_feat_M = sem_feat_M / count
        
        # --- 3. Run VoxelJAFAR ---
        # Pass the CAPTURED stem features + Aggregated semantic features
        refined_logits_M, bdy_logits_M, affinity_M, refined_feat_M, k_idx_M, valid_mask_M = self.geo_stream(
            sp_structure=ptv3_sparse_structure,
            geo_feat_M=geo_feat_M, # Contains RGB info from Stem!
            sem_feat_M=sem_feat_M
        )
        
        # --- 4. Broadcast Back (M -> N) ---
        refined_logits = refined_logits_M[inverse_indices]
        bdy_logits = bdy_logits_M[inverse_indices]
        refined_feat = refined_feat_M[inverse_indices]
        
        if "segment" in input_dict:
            targets = input_dict['segment'].view(-1)
        else:
            targets = None

        if self.training and targets is not None:
            valid_mask = (targets != 255)
            if valid_mask.sum() > 0:
                self.update_prototypes(refined_feat[valid_mask].detach(), targets[valid_mask])

        # Prepare Voxel Center for Loss
        voxel_center = ptv3_sparse_structure.indices[:, 1:].float() * 0.02

        output_dict = {
            "seg_logits": refined_logits,
            "refined_logits": refined_logits,
            "aux_logits": aux_logits,
            "target": targets,
            "prototypes": self.prototypes,
            "refined_feat": refined_feat, 
            
            "voxel_feat": refined_feat_M,      
            "voxel_k_idx": k_idx_M,
            "voxel_valid_mask": valid_mask_M,
            "voxel_affinity": affinity_M,     
            "voxel_bdy_logits": bdy_logits_M, 
            "voxel_input_feat": voxel_center 
        }

        if self.criteria is not None and targets is not None:
            output_dict['loss'] = self.criteria(output_dict)
        elif self.criteria is not None:
            output_dict['loss'] = torch.tensor(0.0, device=refined_logits.device)
            
        return output_dict