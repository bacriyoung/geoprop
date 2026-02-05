import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import LOSS

@LOSS.register_module()
class GeoCoTrainLoss(nn.Module):
    def __init__(self, 
                 lambda_main=1.0, 
                 lambda_aux=1.0, 
                 lambda_aff=0.5, 
                 lambda_rec=1.0, 
                 ignore_index=255,
                 warmup_epochs=0,
                 label_smoothing=0.0, # [新增] 接收 label_smoothing
                 **kwargs):           # [新增] 接收所有其他未定义的参数，防止报错
        super().__init__()
        
        self.lambda_main = lambda_main 
        self.lambda_aux = lambda_aux   
        self.lambda_aff = lambda_aff
        self.lambda_rec = lambda_rec
        self.ignore_index = ignore_index
        
        # [修改] 将 label_smoothing 传给 CE Loss
        # 注意：PyTorch 1.10+ 的 CrossEntropyLoss 支持 label_smoothing 参数
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.mse = nn.MSELoss()

        self.register_buffer('iter_step', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = warmup_epochs * 500

    def dense_gather(self, features, idx):
        B, C, N = features.shape
        K = idx.shape[2]
        idx_base = torch.arange(0, B, device=features.device).view(B, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)
        features_flat = features.transpose(1, 2).contiguous().view(B * N, C)
        out = features_flat[idx, :].view(B, N, K, C).permute(0, 3, 1, 2).contiguous()
        return out

    def forward(self, output_dict, target=None):
        # 兼容处理：有时 target 直接传入，有时在 output_dict 里
        if target is None:
            target = output_dict.get('target')
            
        if self.training:
            self.iter_step += 1
            
        if self.warmup_steps > 0:
            step_ratio = self.iter_step.float() / float(self.warmup_steps)
            alpha = torch.clamp(step_ratio, max=1.0)
        else:
            alpha = torch.tensor(1.0, device=target.device)

        # -----------------------------------------------------------
        # 1. Supervision Loss (Main + Aux)
        # -----------------------------------------------------------
        loss_main = self.ce(output_dict['seg_logits'], target)
        loss_aux = self.ce(output_dict['aux_logits'], target)
        loss_sup = self.lambda_main * loss_main + self.lambda_aux * loss_aux
        
        # -----------------------------------------------------------
        # 2. Affinity Loss (Cosine Guidance)
        # -----------------------------------------------------------
        feat = output_dict['refined_feat'].float()
        affinity = output_dict['affinity'].float()
        knn_idx = output_dict['knn_idx']
        
        # Normalize
        feat_norm = F.normalize(feat, p=2, dim=1)
        center_feat = feat_norm.unsqueeze(-1)
        neighbor_feat = self.dense_gather(feat_norm, knn_idx)
        
        # Cosine Distance
        cos_sim = torch.sum(center_feat * neighbor_feat, dim=1)
        cos_dist = 1.0 - cos_sim
        
        loss_aff = torch.sum(affinity * cos_dist) / (torch.sum(affinity) + 1e-6)

        # -----------------------------------------------------------
        # 3. Reconstruction Loss (Self-Supervision)
        # -----------------------------------------------------------
        loss_rec = torch.tensor(0.0, device=target.device)
        if 'rec_phys' in output_dict and 'target_phys' in output_dict:
            rec_pred = output_dict['rec_phys']
            rec_target = output_dict['target_phys']
            
            # Channel Alignment
            min_c = min(rec_pred.shape[1], rec_target.shape[1])
            loss_rec = self.mse(rec_pred[:, :min_c, :], rec_target[:, :min_c, :])

        total_loss = loss_sup + \
                     loss_aff * (self.lambda_aff * alpha) + \
                     loss_rec * self.lambda_rec
                     
        return total_loss