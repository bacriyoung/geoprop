import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

@LOSSES.register_module()
class GeoCoTrainLoss(nn.Module):
    def __init__(self, 
                 lambda_sup=10.0, 
                 lambda_con=1.0, 
                 lambda_aff=0.1, 
                 lambda_dist=0.1,
                 lambda_bdy=0.5, 
                 warmup_epochs=15,
                 ignore_index=255):
        super().__init__()
        self.lambda_sup = lambda_sup
        self.lambda_con = lambda_con
        self.lambda_aff = lambda_aff
        self.lambda_dist = lambda_dist
        self.lambda_bdy = lambda_bdy
        self.warmup_epochs = warmup_epochs
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # 🟢 [AMP FIX] 使用 BCEWithLogitsLoss，数值稳定且支持自动混合精度
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, output_dict):
        target = output_dict['target']
        curr_epoch = output_dict.get('epoch', 0)
        
        # 1. Supervised
        loss_sup = self.ce(output_dict['sem_logits'], target) + \
                   self.ce(output_dict['geo_logits'], target)
        
        # 2. Consistency
        progress = min(1.0, max(0.0, (curr_epoch - 1) / self.warmup_epochs))
        p_sem = F.softmax(output_dict['sem_logits'], dim=1) + 1e-6
        p_geo = F.softmax(output_dict['geo_logits'], dim=1) + 1e-6
        
        if curr_epoch < self.warmup_epochs:
            current_lambda_con = self.lambda_con * progress * 0.1
            loss_con = F.kl_div(p_sem.log(), p_geo.detach(), reduction='batchmean')
        else:
            current_lambda_con = self.lambda_con
            loss_con = (F.kl_div(p_sem.log(), p_geo.detach(), reduction='batchmean') + 
                        F.kl_div(p_geo.log(), p_sem.detach(), reduction='batchmean')) * 0.5

        # 3. Affinity
        sem_feat = output_dict['sem_feat_dense']
        affinity = output_dict['affinity'] 
        k_idx = output_dict['k_idx']
        
        B, N, K = k_idx.shape
        C = sem_feat.shape[-1]
        sem_feat_flat = sem_feat.view(B*N, C)
        batch_offset = torch.arange(B, device=k_idx.device).unsqueeze(1).unsqueeze(2) * N
        k_idx_flat = (k_idx + batch_offset).view(-1)
        neighbor_feat = sem_feat_flat[k_idx_flat].view(B, N, K, C)
        center_feat = sem_feat.unsqueeze(2).expand(-1, -1, K, -1)
        
        feat_dist = torch.sum((center_feat - neighbor_feat) ** 2, dim=-1) / (C ** 0.5)
        valid_aff_mask = (affinity > 0.8).float()
        loss_aff = torch.sum(affinity.detach() * feat_dist * valid_aff_mask) / (valid_aff_mask.sum() + 1e-6)

        # 4. Distribution
        loss_dist = torch.tensor(0.0, device=target.device)
        if 'prototypes' in output_dict and curr_epoch > 0:
            prototypes = output_dict['prototypes']
            sem_feat_norm = F.normalize(sem_feat_flat, p=2, dim=1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            sim_matrix = torch.mm(sem_feat_norm, prototypes_norm.t())
            valid_mask = (target != self.ignore_index)
            if valid_mask.sum() > 0:
                valid_sim = sim_matrix[valid_mask]
                valid_target = target[valid_mask]
                target_sim = valid_sim.gather(1, valid_target.unsqueeze(1)).squeeze()
                loss_dist = torch.mean(1.0 - target_sim)

        # 5. Self-Supervised Boundary
        feat_inp = output_dict['input_jafar_feat']
        feat_inp_flat = feat_inp.view(B*N, -1)
        neighbor_inp = feat_inp_flat[k_idx_flat].view(B, N, K, -1)
        center_inp = feat_inp.unsqueeze(2).expand(-1, -1, K, -1)
        
        joint_diff = torch.norm(center_inp - neighbor_inp, dim=-1)
        edge_score_pseudo = joint_diff.mean(dim=-1)
        
        # Soft Target [0-1]
        target_bdy = torch.sigmoid((edge_score_pseudo - 0.15) * 20)
        
        # Logits [(-inf, inf)]
        pred_bdy_logits = output_dict['bdy_logits'].squeeze(1) 
        
        # BCEWithLogitsLoss computes Sigmoid internally
        loss_bdy = self.bce(pred_bdy_logits, target_bdy.detach())

        return loss_sup * self.lambda_sup + \
               loss_con * current_lambda_con + \
               loss_aff * self.lambda_aff + \
               loss_dist * self.lambda_dist + \
               loss_bdy * self.lambda_bdy