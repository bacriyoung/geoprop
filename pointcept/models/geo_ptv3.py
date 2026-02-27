import torch
import torch.nn as nn
import torch.nn.functional as F
import pointops
import inspect
from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

# =================================================================================
# Sparse Anchor JAFAR: 单流稀疏几何修正模块
# =================================================================================
class SparseAnchorJAFAR(nn.Module):
    def __init__(self, qk_dim=64, input_feat_dim=6, sem_dim=192, num_classes=13): 
        super().__init__()
        self.qk_dim = qk_dim
        self.input_feat_dim = input_feat_dim 
        self.sem_dim = sem_dim 

        # 几何编码器：直接处理 (XYZ, RGB)
        self.geom_encoder = nn.Sequential(
            nn.Conv1d(self.input_feat_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), 
            nn.ReLU(),
            nn.Conv1d(qk_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), 
            nn.ReLU()
        )
        
        # 语义 Value 投影
        self.val_proj = nn.Sequential(
            nn.Conv1d(sem_dim, qk_dim, 1),
            nn.GroupNorm(8, qk_dim), 
            nn.ReLU()
        )
        
        self.geo_query = nn.Conv1d(qk_dim, qk_dim, 1)
        self.geo_key = nn.Conv1d(qk_dim, qk_dim, 1)
        
        # 相对位置编码 (Geometric Awareness)
        # 输入: dx, dy, dz, dist
        self.rel_pos_mlp = nn.Sequential(
            nn.Conv2d(4, qk_dim, 1), 
            nn.GroupNorm(8, qk_dim), 
            nn.ReLU(),
            nn.Conv2d(qk_dim, qk_dim, 1)
        )
        
        self.cls_head = nn.Linear(qk_dim, num_classes)

        # 重建头 (自监督：尝试重建输入的 Raw Feature)
        self.rec_head = nn.Sequential(
            nn.Linear(qk_dim, qk_dim),
            nn.LayerNorm(qk_dim),
            nn.ReLU(),
            nn.Linear(qk_dim, input_feat_dim) 
        )

    def _gather_val_efficient(self, tensor, idx):
        # tensor: (B, C, N_anchor)
        # idx: (B, N_query, k) -> 指向 Anchor 的索引
        b_dim, c_dim, n_anchor = tensor.shape
        _, n_query, k_dim = idx.shape
        
        tensor_flat = tensor.transpose(1, 2).contiguous().view(b_dim * n_anchor, c_dim)
        batch_offset = torch.arange(b_dim, device=tensor.device).view(b_dim, 1, 1) * n_anchor
        flat_idx = (idx + batch_offset).view(-1)
        
        val = tensor_flat[flat_idx].view(b_dim, n_query, k_dim, c_dim).permute(0, 3, 1, 2)
        return val

    def forward(self, 
                feat_q, sem_q, 
                feat_k, sem_k, 
                knn_idx):
        """
        feat_q: (B, N_q, 6) - XYZ+RGB of Query
        sem_q: (B, N_q, C_sem) - PTV3 feat of Query
        feat_k: (B, N_k, 6) - XYZ+RGB of Anchor
        sem_k: (B, N_k, C_sem) - PTV3 feat of Anchor
        knn_idx: (B, N_q, k) - Indices into Anchor
        """
        
        # 1. 编码 Query 几何
        feat_q_t = feat_q.transpose(1, 2).contiguous()
        geom_emb_q = self.geom_encoder(feat_q_t)
        Q = self.geo_query(geom_emb_q) # (B, Dim, N_q)

        # 2. 编码 Key 几何 (Anchors)
        feat_k_t = feat_k.transpose(1, 2).contiguous()
        geom_emb_k = self.geom_encoder(feat_k_t)
        K_all = self.geo_key(geom_emb_k)
        K_g = self._gather_val_efficient(K_all, knn_idx) # (B, Dim, N_q, k)

        # 3. 编码 Value 语义 (Anchors)
        sem_k_t = sem_k.transpose(1, 2).contiguous()
        V_all = self.val_proj(sem_k_t)
        V_g = self._gather_val_efficient(V_all, knn_idx) # (B, Dim, N_q, k)

        # 4. 相对位置编码
        # 分离出 XYZ (前3维)
        xyz_q = feat_q[:, :, :3]
        xyz_k = feat_k[:, :, :3]
        
        xyz_k_t = xyz_k.transpose(1, 2).contiguous()
        xyz_g = self._gather_val_efficient(xyz_k_t, knn_idx) # (B, 3, N_q, k)
        
        xyz_q_t = xyz_q.transpose(1, 2).contiguous()
        xyz_q_expanded = xyz_q_t.unsqueeze(-1)
        
        rel_diff = xyz_q_expanded - xyz_g 
        rel_dist = torch.sqrt(torch.sum(rel_diff ** 2, dim=1, keepdim=True) + 1e-10)
        
        # (B, 4, N_q, k)
        rel_geo_feat = torch.cat([rel_diff, rel_dist], dim=1) 
        pos_enc = self.rel_pos_mlp(rel_geo_feat)

        # 5. Attention Interaction
        # 利用几何相似度 (Q * K) 指导 语义聚合 (V)
        attn_logits = torch.sum(Q.unsqueeze(-1) * (K_g + pos_enc), dim=1) / (self.qk_dim ** 0.5)
        affinity = torch.softmax(attn_logits.float(), dim=-1).type_as(attn_logits)
        
        # 6. Aggregation
        refined_feat = torch.sum(affinity.unsqueeze(1) * V_g, dim=-1) # (B, Dim, N_q)
        
        # Residual: 加上 Query 自己的 PTV3 特征 (做过投影)
        sem_q_t = sem_q.transpose(1, 2).contiguous()
        V_q = self.val_proj(sem_q_t)
        refined_feat = refined_feat + V_q
        
        # 7. Prediction Heads
        refined_feat_flat = refined_feat.transpose(1, 2).contiguous().view(-1, self.qk_dim)
        logits = self.cls_head(refined_feat_flat)
        rec = self.rec_head(refined_feat_flat) 
        
        return logits, rec

@MODELS.register_module()
class GeoPTV3(nn.Module):
    def __init__(self, backbone_ptv3_cfg, geo_input_dim=6, num_classes=13,
                 criteria=None, query_ratio=0.25, anchor_ratio=0.5): 
        super().__init__()
        
        valid_params = inspect.signature(PointTransformerV3.__init__).parameters
        clean_cfg = {k: v for k, v in backbone_ptv3_cfg.items() if k in valid_params}
        self.sem_stream = PointTransformerV3(**clean_cfg)
        
        # PTV3 Decoder Channels
        dec_channels = backbone_ptv3_cfg.get('dec_channels', [48, 96, 192, 384])
        self.sem_feat_dim = dec_channels[0]
        self.aux_head = nn.Linear(self.sem_feat_dim, num_classes)
        
        self.ptv3_in_channels = backbone_ptv3_cfg.get("in_channels", 6)
        self.query_ratio = query_ratio   
        self.anchor_ratio = anchor_ratio 
        self.jafar_in_dim = 6 # Fixed to XYZ+RGB
        
        print(f"[GeoPTV3] Anchor Mode. Query Top {query_ratio*100}%, Anchor Bottom {anchor_ratio*100}%")
        
        self.geo_stream = SparseAnchorJAFAR(
            qk_dim=64,
            input_feat_dim=self.jafar_in_dim,
            sem_dim=self.sem_feat_dim, 
            num_classes=num_classes
        )
        
        if criteria is not None:
            self.criteria = LOSSES.build(criteria)
        else:
            self.criteria = None

    def forward(self, input_dict):
        # ------------------------------------------------------------------
        # Fragment Handling (Validation/Test)
        # ------------------------------------------------------------------
        if "fragment_list" in input_dict:
            fragment_list = input_dict["fragment_list"][0]
            full_segment = input_dict["segment"].view(-1)
            num_points_total = full_segment.shape[0]
            num_classes = self.aux_head.out_features
            device = torch.cuda.current_device()
            full_logits = torch.zeros((num_points_total, num_classes), device=device)
            full_counts = torch.zeros((num_points_total, 1), device=device)
            for fragment in fragment_list:
                for key in fragment.keys():
                    if isinstance(fragment[key], torch.Tensor):
                        fragment[key] = fragment[key].to(device)
                chunk_output = self.forward(fragment)
                chunk_logits = torch.softmax(chunk_output["seg_logits"], dim=-1)
                global_idx = fragment["index"].long()
                full_logits.index_add_(0, global_idx, chunk_logits)
                full_counts.index_add_(0, global_idx, torch.ones_like(chunk_logits[:, :1]))
            full_logits /= full_counts.clamp(min=1.0)
            if full_segment.device != device:
                full_segment = full_segment.to(device)
            val_loss = F.nll_loss(torch.log(full_logits.clamp(min=1e-6)), full_segment.long(), ignore_index=255)
            logits_eval = full_logits.permute(1, 0).unsqueeze(0)
            output_dict = {"seg_logits": logits_eval, "target": full_segment, "loss": val_loss}
            return output_dict

        # ------------------------------------------------------------------
        # 1. PTV3 前向传播
        # ------------------------------------------------------------------
        coord = input_dict['coord'] # (N, 3)
        feat = input_dict.get("ptv3_feat", input_dict.get("feat")) # Usually (N, 6) or (N, 3)
        
        # 构造统一的 raw_input (N, 6) -> XYZ + RGB
        if feat.shape[1] == 3:
            raw_input = torch.cat([coord, feat], dim=1)
        else:
            raw_input = feat # Assume feat already contains XYZ+RGB if dim=6
            
        ptv3_input = {}
        ptv3_input["coord"] = coord
        ptv3_input["feat"] = feat # PTV3内部处理维度
        ptv3_input["grid_coord"] = input_dict.get("grid_coord", (coord / 0.02).int())
        ptv3_input["batch"] = input_dict.get("batch", torch.zeros(coord.shape[0], device=coord.device).long())
        
        # Offset Generation
        batch_idx = ptv3_input["batch"]
        if "batch" in input_dict:
             batch_size_val = input_dict["batch"].max().item() + 1
        else:
             batch_size_val = 1
        _, counts = torch.unique(batch_idx, return_counts=True)
        offset = torch.cumsum(counts, dim=0).int()

        # Forward Backbone
        sem_feat = self.sem_stream(ptv3_input).feat 
        aux_logits = self.aux_head(sem_feat) 

        # ------------------------------------------------------------------
        # 2. 锚点与难点挖掘 (Global Sorting)
        # ------------------------------------------------------------------
        with torch.no_grad():
            probs = torch.softmax(aux_logits, dim=-1)
            confidence, _ = torch.max(probs, dim=-1) 
            
            # Global Sort by Confidence
            sorted_indices = torch.argsort(confidence) 
            
            num_total = confidence.shape[0]
            k_query = int(num_total * self.query_ratio)   
            k_anchor = int(num_total * self.anchor_ratio) 
            
            # Slice Indices
            query_indices = sorted_indices[:k_query]      # Hardest
            anchor_indices = sorted_indices[-k_anchor:]   # Easiest
            
            # Re-sort indices to match batch order (Critical for KNN Offset)
            query_indices, _ = torch.sort(query_indices)
            anchor_indices, _ = torch.sort(anchor_indices)
            
            # Generate Subset Offsets
            def get_subset_offset(indices, original_batch_idx, batch_size):
                sub_batch = original_batch_idx[indices]
                counts_full = torch.zeros(batch_size, device=indices.device, dtype=torch.int32)
                present_b, present_c = torch.unique(sub_batch, return_counts=True)
                counts_full[present_b] = present_c.int()
                return torch.cumsum(counts_full, dim=0).int()

            query_offset = get_subset_offset(query_indices, batch_idx, batch_size_val)
            anchor_offset = get_subset_offset(anchor_indices, batch_idx, batch_size_val)

        # ------------------------------------------------------------------
        # 3. 单次稀疏 KNN (Query -> Anchor)
        # ------------------------------------------------------------------
        xyz_q = coord[query_indices]
        xyz_k = coord[anchor_indices]
        
        # idx_in_anchor: (N_query, 16) - Index relative to xyz_k
        idx_in_anchor = pointops.knn_query(16, xyz_k, anchor_offset, xyz_q, query_offset)[0].long()
        
        # ------------------------------------------------------------------
        # 4. 执行 JAFAR
        # ------------------------------------------------------------------
        # Prepare Data
        raw_q = raw_input[query_indices]
        sem_q_sub = sem_feat[query_indices]
        
        raw_k = raw_input[anchor_indices]
        sem_k_sub = sem_feat[anchor_indices]
        
        # Fake Batching (B=1)
        logits_sp, rec_sp = self.geo_stream(
            feat_q=raw_q.unsqueeze(0), 
            sem_q=sem_q_sub.unsqueeze(0),
            feat_k=raw_k.unsqueeze(0), 
            sem_k=sem_k_sub.unsqueeze(0),
            knn_idx=idx_in_anchor.unsqueeze(0)
        )
        
        # ------------------------------------------------------------------
        # 5. 结果回填与输出
        # ------------------------------------------------------------------
        logits_sp = logits_sp.view(-1, self.aux_head.out_features)
        rec_sp = rec_sp.view(-1, self.jafar_in_dim)
        
        final_logits = aux_logits.clone()
        final_logits[query_indices] = logits_sp
        
        output_dict = {
            "seg_logits": final_logits,
            "aux_logits": aux_logits,
            "rec_pred": rec_sp,          
            "rec_target": raw_q,    # Reconstruction Target is Raw Input
            "target": input_dict.get('segment')
        }

        if self.criteria is not None:
             output_dict['loss'] = self.criteria(output_dict)
             
        return output_dict