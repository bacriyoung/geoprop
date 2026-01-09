import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch.distributed as dist

from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
from pointcept.models.utils.structure import Point

# =========================================================================
# 1. VoxelJAFAR Module (Structure-Aware Refiner)
# =========================================================================
class VoxelJAFAR(nn.Module):
    def __init__(self, 
                 geo_input_channels, 
                 sem_input_channels,
                 attn_dim=64, 
                 kernel_size=5, # [CORRECTION] Aligned with PTv3 Stem
                 search_radius=3, 
                 num_classes=13, 
                 use_rel_pos=True,
                 indice_key=None): # [NEW] Reuse PTv3 structure
        super().__init__()
        self.use_rel_pos = use_rel_pos
        self.search_radius = search_radius
        self.attn_dim = attn_dim
        
        # 1. Geometry Encoder (Sparse Convolution)
        # Input: [M, C_geo] -> Voxel Features
        self.sparse_conv = spconv.SubMConv3d(
            geo_input_channels, 
            attn_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            bias=False, 
            indice_key=indice_key # [CRITICAL] Reuse backbone hash table
        )
        self.geo_norm = nn.LayerNorm(attn_dim)
        self.geo_act = nn.ReLU(inplace=True)

        # 2. Boundary Head
        self.bdy_head = nn.Linear(attn_dim, 1)

        # 3. Attention Projections
        self.to_q = nn.Linear(attn_dim, attn_dim, bias=False)
        self.to_k = nn.Linear(attn_dim, attn_dim, bias=False)
        self.to_v = nn.Linear(sem_input_channels, attn_dim, bias=False) 
        
        # 4. Positional Embedding
        if self.use_rel_pos:
            diameter = 2 * self.search_radius + 1
            num_positions = diameter ** 3
            self.pos_embedding = nn.Embedding(num_positions, attn_dim)
        
        self.out_proj = nn.Linear(attn_dim, attn_dim)
        self.cls_head = nn.Linear(attn_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def _hash_query_voxels(self, indices, batch_idx, device):
        """
        Hash-based neighbor search on M unique voxels.
        """
        M = indices.shape[0]
        r = self.search_radius
        
        # 1. Offsets (Relative Voxel Coordinate Offset Definition)
        range_ = torch.arange(-r, r + 1, device=device)
        offsets = torch.stack(torch.meshgrid(range_, range_, range_, indexing='ij'), dim=-1).reshape(-1, 3)
        K = offsets.shape[0]

        # 2. Build Hash Map
        scale = 4096 
        keys = batch_idx * (scale**3) + indices[:, 2] * (scale**2) + indices[:, 1] * scale + indices[:, 0]
        sorted_keys, sort_idx = torch.sort(keys)
        
        # 3. Query
        neighbor_coords = indices.unsqueeze(1) + offsets.unsqueeze(0)
        neighbor_batch = batch_idx.unsqueeze(1).repeat(1, K)
        
        query_keys = neighbor_batch * (scale**3) + neighbor_coords[:, :, 2] * (scale**2) + neighbor_coords[:, :, 1] * scale + neighbor_coords[:, :, 0]
        query_keys_flat = query_keys.view(-1)
        
        # 4. Search
        idx_in_sorted = torch.searchsorted(sorted_keys, query_keys_flat).clamp(max=M-1)
        found_keys = sorted_keys[idx_in_sorted]
        
        mask = (found_keys == query_keys_flat)
        neighbor_indices = sort_idx[idx_in_sorted].view(M, K)
        
        return neighbor_indices, mask.view(M, K), K

    def forward(self, sp_structure, geo_feat_M, sem_feat_M):
        # --- A. Sparse Convolution ---
        input_sp = sp_structure.replace_feature(geo_feat_M)
        output_sp = self.sparse_conv(input_sp)
        Q_geo = self.geo_act(self.geo_norm(output_sp.features))

        # --- B. Boundary ---
        bdy_logits = self.bdy_head(Q_geo)

        # --- C. Neighbor Search ---
        # indices format: [Batch, Z, Y, X] (Standard spconv order)
        batch_idx = sp_structure.indices[:, 0]
        coords = sp_structure.indices[:, 1:]
        neighbor_idx, valid_mask, K = self._hash_query_voxels(coords, batch_idx, coords.device)

        # --- D. Attention ---
        flat_idx = neighbor_idx.view(-1)
        K_feat = Q_geo[flat_idx].view(-1, K, self.attn_dim)
        V_feat = sem_feat_M[flat_idx].view(-1, K, sem_feat_M.shape[-1])
        
        Q_proj = self.to_q(Q_geo).unsqueeze(1)
        K_proj = self.to_k(K_feat)
        V_proj = self.to_v(V_feat)
        
        # Relative Position Embedding (Your "Relative Voxel Coordinate Offset" logic)
        if self.use_rel_pos:
            pos_ids = torch.arange(K, device=coords.device).unsqueeze(0).expand(Q_geo.shape[0], -1)
            P_pos = self.pos_embedding(pos_ids)
            K_proj = K_proj + P_pos
            
        attn_logits = torch.matmul(Q_proj, K_proj.transpose(-1, -2)) / (self.attn_dim ** 0.5)
        attn_logits = attn_logits.masked_fill(~valid_mask.unsqueeze(1), -1e9)
        affinity = self.softmax(attn_logits)
        
        refined = torch.matmul(affinity, V_proj).squeeze(1)
        refined = refined + self.to_v(sem_feat_M) # Residual
        
        refined_feat = self.out_proj(refined)
        logits = self.cls_head(refined_feat)
        
        return logits, bdy_logits, affinity, refined_feat, neighbor_idx


# =========================================================================
# 2. GeoPTV3 (Restored Full Interface)
# =========================================================================
@MODELS.register_module("GeoPTV3")
class GeoPTV3(PointTransformerV3):
    def __init__(self, 
                 geo_input_channels=6, 
                 jafar_kernel_size=5, # [CORRECTION] Aligned with PTv3 Stem
                 attn_search_radius=3,
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
        
        # [NEW] Default key for PTv3 Stem structure reuse
        backbone_indice_key = "subm1" 

        self.geo_stream = VoxelJAFAR(
            geo_input_channels=geo_input_channels,
            sem_input_channels=self.sem_feat_dim,
            attn_dim=attn_dim,
            kernel_size=jafar_kernel_size,
            search_radius=attn_search_radius,
            num_classes=self.num_classes,
            use_rel_pos=use_rel_pos,
            indice_key=backbone_indice_key 
        )
        
        self.register_buffer("prototypes", torch.zeros(self.num_classes, attn_dim))
        self.register_buffer("proto_count", torch.zeros(self.num_classes))
        self.momentum = 0.99
        
        if criteria is not None:
            self.criteria = LOSSES.build(criteria)
        else:
            self.criteria = None

    def update_prototypes(self, features, labels):
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
        ptv3_sparse_structure = point.sparse_conv_feat
        
        point = self.enc(point)
        point = self.dec(point)
        
        sem_feat_N = point.feat 
        aux_logits = self.aux_head(sem_feat_N)
        
        # --- 2. Prepare JAFAR Inputs (N -> M) [FIXED] ---
        # Use Voxel Center Coord + RGB
        
        grid_coord = point.grid_coord 
        raw_feat = input_dict["feat"]
        
        # [CRITICAL FIX] Geometric Input: [RGB, Voxel_Center_Absolute]
        # We DELETE point-level offsets. We use Voxel Center Coordinates.
        # This preserves the Voxel's Geometry. Relative offsets are implicit A-B.
        grid_size = 0.02 
        voxel_center = grid_coord.float() * grid_size # [N, 3] Absolute Voxel Coord
        rgb = raw_feat[:, :3] 
        geo_input_N = torch.cat([rgb, voxel_center], dim=1) # [N, 6]

        # [ALIGNMENT LOGIC]
        # Align N points to M voxels based on PTv3 structure
        
        M = ptv3_sparse_structure.features.shape[0]
        sp_indices = ptv3_sparse_structure.indices # [M, 4] (batch, z, y, x)
        
        scale = 4096 
        def make_keys(batch, coords):
            return batch * (scale**3) + coords[:, 0] * (scale**2) + coords[:, 1] * scale + coords[:, 2]

        # Reference Keys (M Voxels)
        m_batch = sp_indices[:, 0]
        m_coords = sp_indices[:, 1:]
        m_keys = make_keys(m_batch, m_coords)
        
        # Query Keys (N Points)
        n_batch = point.batch
        n_coords = grid_coord
        n_keys = make_keys(n_batch, n_coords)
        
        # Sort and Search
        m_keys_sorted, m_sort_idx = torch.sort(m_keys)
        idx_in_sorted = torch.searchsorted(m_keys_sorted, n_keys).clamp(max=M-1)
        point_to_voxel_idx = m_sort_idx[idx_in_sorted] # [N] -> [0...M-1]
        
        # Aggregate Features (Mean)
        geo_feat_M = torch.zeros((M, geo_input_N.shape[1]), device=rgb.device)
        sem_feat_M = torch.zeros((M, sem_feat_N.shape[1]), device=rgb.device)
        count = torch.zeros((M, 1), device=rgb.device)
        
        geo_feat_M.index_add_(0, point_to_voxel_idx, geo_input_N)
        sem_feat_M.index_add_(0, point_to_voxel_idx, sem_feat_N)
        count.index_add_(0, point_to_voxel_idx, torch.ones_like(count))
        
        geo_feat_M = geo_feat_M / count.clamp(min=1.0)
        sem_feat_M = sem_feat_M / count.clamp(min=1.0)
        
        inverse_indices = point_to_voxel_idx
        
        # --- 3. Run VoxelJAFAR (on M) ---
        refined_logits_M, bdy_logits_M, affinity_M, refined_feat_M, k_idx_M = self.geo_stream(
            sp_structure=ptv3_sparse_structure,
            geo_feat_M=geo_feat_M,
            sem_feat_M=sem_feat_M
        )
        
        # --- 4. Broadcast Back (M -> N) ---
        refined_logits = refined_logits_M[inverse_indices]
        bdy_logits = bdy_logits_M[inverse_indices]
        refined_feat = refined_feat_M[inverse_indices]
        
        # Target handling
        if "segment" in input_dict:
            targets = input_dict['segment'].view(-1)
        else:
            targets = None

        # --- 5. Update Prototypes ---
        if self.training and targets is not None:
            valid_mask = (targets != 255)
            if valid_mask.sum() > 0:
                self.update_prototypes(refined_feat[valid_mask].detach(), targets[valid_mask])

        # --- 6. Final Output Dict ---
        output_dict = {
            "seg_logits": refined_logits,
            "refined_logits": refined_logits,
            "aux_logits": aux_logits,
            "target": targets,
            "prototypes": self.prototypes,
            
            "refined_feat": refined_feat, 

            # [M-Voxel Level]
            "voxel_feat": refined_feat_M,      
            "voxel_k_idx": k_idx_M,           
            "voxel_affinity": affinity_M,     
            "voxel_bdy_logits": bdy_logits_M, 
            "voxel_input_feat": geo_feat_M     # Contains [RGB, Voxel_Coord]
        }

        if self.criteria is not None and targets is not None:
            output_dict['loss'] = self.criteria(output_dict)
        elif self.criteria is not None:
            output_dict['loss'] = torch.tensor(0.0, device=refined_logits.device)
            
        return output_dict