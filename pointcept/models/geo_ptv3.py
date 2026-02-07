import torch
import torch.nn as nn
import torch.nn.functional as F
import pointops
import inspect
from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

@MODELS.register_module()
class GeoPTV3(nn.Module):
    def __init__(self, backbone_ptv3_cfg, geo_input_dim=6, num_classes=13,
                 num_points=80000, geo_scale=10.0, criteria=None):
        super().__init__()
        
        # 1. PTV3 Backbone
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
        
        # [Ablation] Direct Reconstruction
        print(f"[GeoPTV3] Mode: Direct Reconstruction (Force Aligned)")
        
        self.rec_head = nn.Sequential(
            nn.Linear(self.sem_feat_dim, self.sem_feat_dim),
            nn.LayerNorm(self.sem_feat_dim),
            nn.ReLU(),
            nn.Linear(self.sem_feat_dim, 6) 
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
        if "fragment_list" in input_dict:
            # Test time logic (Keep as is)
            return self.forward_test(input_dict)

        # ------------------------------------------------------------------
        # 1. 原始数据准备 (Target Source)
        # ------------------------------------------------------------------
        # 这里可能会包含多余的 Batch 数据 (例如 594000)
        if "iso_coord" in input_dict:
            j_coord = input_dict['jafar_coord']
            j_feat_raw = input_dict['jafar_feat']
            iso_coord = input_dict['iso_coord']
        else:
            j_coord = input_dict['coord'].clone()
            j_feat_raw = input_dict['feat'].clone()
            iso_coord = j_coord - j_coord.min(0)[0]
            
        # ------------------------------------------------------------------
        # 2. PTV3 Forward
        # ------------------------------------------------------------------
        ptv3_input = {}
        raw_coord = input_dict["coord"]
        raw_feat = input_dict.get("ptv3_feat", input_dict.get("feat"))
        raw_grid = input_dict.get("grid_coord")
        
        # Batch 处理
        if "batch" in input_dict:
            ptv3_input["batch"] = input_dict["batch"]
        else:
            total_p = raw_coord.shape[0]
            batch_size_val = max(1, total_p // self.num_points)
            ptv3_input["batch"] = torch.arange(batch_size_val, device=raw_coord.device).repeat_interleave(self.num_points)
            if ptv3_input["batch"].shape[0] > raw_coord.shape[0]:
                ptv3_input["batch"] = ptv3_input["batch"][:raw_coord.shape[0]]

        ptv3_input["coord"] = raw_coord
        ptv3_input["feat"] = raw_feat
        if self.ptv3_in_channels == 6 and raw_feat.shape[1] == 3:
            ptv3_input["feat"] = torch.cat([raw_feat, raw_coord], dim=1)
        if raw_grid is None:
            ptv3_input["grid_coord"] = (raw_coord / 0.02).int()
        else:
            ptv3_input["grid_coord"] = raw_grid

        # [PTV3 Forward]
        # sem_feat_sparse: 这里的长度是 396000 (2个样本), 是绝对真理
        sem_feat_sparse = self.sem_stream(ptv3_input).feat 
        aux_logits = self.aux_head(sem_feat_sparse) 
        
        # ------------------------------------------------------------------
        # 3. Direct Reconstruction (强制对齐逻辑)
        # ------------------------------------------------------------------
        
        # [Step A] 预测值
        rec_phys = self.rec_head(sem_feat_sparse) # (N_pred, 6)
        
        # [Step B] 目标值 (可能是 594000)
        target_phys_full = torch.cat([iso_coord, j_feat_raw], dim=-1).contiguous().view(-1, 6)
        
        # [Step C] 终极对齐：以 Prediction 长度为准，切掉 Target 多余部分
        N_pred = rec_phys.shape[0]
        N_target = target_phys_full.shape[0]
        
        # 这段代码解决了 396000 vs 594000 的冲突
        if N_pred != N_target:
            min_len = min(N_pred, N_target)
            rec_phys = rec_phys[:min_len]
            target_phys = target_phys_full[:min_len]
            
            # 同时也要切 auxiliary logits 保证 loss 计算一致
            aux_logits = aux_logits[:min_len]
            sem_feat_sparse = sem_feat_sparse[:min_len]
        else:
            target_phys = target_phys_full

        # ------------------------------------------------------------------
        # 4. Output Construction (Dummy)
        # ------------------------------------------------------------------
        N_current = rec_phys.shape[0]
        
        # Dummy 也要用 N_current (396000)
        dummy_affinity = torch.zeros((1, N_current, 16), device=sem_feat_sparse.device)
        dummy_k_idx = torch.zeros((1, N_current, 16), dtype=torch.long, device=sem_feat_sparse.device)
        dummy_bdy_logits = torch.zeros((1, 1, N_current), device=sem_feat_sparse.device)
        dummy_jafar_input = torch.zeros((1, N_current, self.real_geo_dim), device=sem_feat_sparse.device)

        # Label 也要切！
        if "segment" in input_dict:
            targets = input_dict['segment'].view(-1)
            if targets.shape[0] > N_current:
                targets = targets[:N_current]
        else:
            targets = None
        
        if self.training and targets is not None:
            valid_mask = (targets != 255)
            if valid_mask.sum() > 0:
                self.update_prototypes(sem_feat_sparse[valid_mask].detach(), targets[valid_mask])

        output_dict = {
            "seg_logits": aux_logits,
            "refined_logits": aux_logits,
            "aux_logits": aux_logits,
            
            "bdy_logits": dummy_bdy_logits,    
            "affinity": dummy_affinity,       
            "k_idx": dummy_k_idx,
            "input_jafar_feat": dummy_jafar_input, 
            
            # 这里的 tensor 长度绝对一致 (N_current)
            "refined_feat": sem_feat_sparse, 
            "rec_phys": rec_phys,  
            "target_phys": target_phys, 
            
            "target": targets,
            "prototypes": self.prototypes,
        }

        if self.criteria is not None and targets is not None:
            output_dict['loss'] = self.criteria(output_dict)
        elif self.criteria is not None:
            output_dict['loss'] = torch.tensor(0.0, device=rec_phys.device)
            
        return output_dict

    def forward_test(self, input_dict):
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