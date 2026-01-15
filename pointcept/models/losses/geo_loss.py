import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

@LOSSES.register_module()
class GeoCoTrainLoss(nn.Module):
    def __init__(self, 
                 lambda_main=1.0,   
                 lambda_aux=1.0,    
                 lambda_aff=0.5,
                 lambda_rec=1.0,    
                 lambda_dist=0.1,   
                 lambda_bdy=0.1,    
                 warmup_epochs=0,   
                 ignore_index=255,
                 class_weights=None):
        super().__init__()
        
        self.lambda_main = lambda_main 
        self.lambda_aux = lambda_aux   
        self.lambda_aff = lambda_aff
        self.lambda_rec = lambda_rec
        self.lambda_dist = lambda_dist
        self.lambda_bdy = lambda_bdy
        self.ignore_index = ignore_index
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        self.register_buffer('iter_step', torch.tensor(0, dtype=torch.long))
        self.warmup_steps = warmup_epochs * 500

    def forward(self, output_dict):
        
        if torch.isnan(output_dict['refined_logits']).any():
            print("💀 NaN detected in output_dict['refined_logits'] entering Loss function!")

        target = output_dict['target']
        
        # Update Iteration Step
        if self.training:
            self.iter_step += 1
        
        if self.warmup_steps > 0:
            step_ratio = self.iter_step.float() / float(self.warmup_steps)
            alpha = torch.clamp(step_ratio, max=1.0)
        else:
            alpha = torch.tensor(1.0, device=target.device)

        # 1. Dual Supervision (Refined + Aux)
        # -----------------------------------------------------------
        loss_main = self.ce(output_dict['refined_logits'], target)
        loss_aux = self.ce(output_dict['aux_logits'], target)
        loss_sup = self.lambda_main * loss_main + self.lambda_aux * loss_aux
        
        # 2. Affinity Loss (Weighted Cosine with NaN Protection)
        # -----------------------------------------------------------
        feat = output_dict['refined_feat'].float() 
        affinity = output_dict['affinity'].float()
        k_idx = output_dict['k_idx']
        
        B, N, K = k_idx.shape
        C = feat.shape[-1]
        
        # [FIX] Stronger eps (1e-6) to prevent NaN if feat is zero-vector
        feat_norm = F.normalize(feat, p=2, dim=-1, eps=1e-6)
        
        batch_offset = torch.arange(B, device=k_idx.device).view(B, 1, 1) * N
        k_idx_flat = (k_idx + batch_offset).view(-1)
        
        feat_flat = feat_norm.view(B*N, C)
        neighbor_feat = feat_flat[k_idx_flat].view(B, N, K, C)
        center_feat = feat_norm.view(B, N, C).unsqueeze(2) # (B, N, 1, C)
        
        # [FIX] Clamp cosine similarity to [-1, 1] before subtracting
        # Floating point errors can cause sum > 1.0, leading to negative distance if using acos, 
        # or instability in gradients.
        cos_sim = torch.sum(center_feat * neighbor_feat, dim=-1)
        cos_dist = 1.0 - torch.clamp(cos_sim, min=-1.0, max=1.0)
        
        aff_weight = affinity.detach()
        loss_aff = torch.sum(aff_weight * cos_dist) / (torch.sum(aff_weight) + 1e-6)

        # 3. Reconstruction Loss (Upper Strategy)
        # -----------------------------------------------------------
        loss_rec = torch.tensor(0.0, device=target.device)
        if 'rec_phys' in output_dict and 'target_phys' in output_dict:
            rec_pred = output_dict['rec_phys'].float()
            rec_target = output_dict['target_phys'].float()
            
            # [FIX] Clamp predictions to prevent huge MSE gradients in early training
            # Target is normalized [0, 1] or close to it. +/- 10 is a safe bound.
            rec_pred = torch.clamp(rec_pred, min=-10.0, max=10.0)
            
            loss_rec = self.mse(rec_pred, rec_target)

        # 4. Distribution Loss (Prototypes)
        # -----------------------------------------------------------
        loss_dist = torch.tensor(0.0, device=target.device)
        if 'prototypes' in output_dict:
            prototypes = output_dict['prototypes'].float()
            feat_flat_raw = output_dict['refined_feat'].view(B*N, C).float()
            
            # [FIX] Stronger eps
            feat_norm_p = F.normalize(feat_flat_raw, p=2, dim=1, eps=1e-6)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1, eps=1e-6)
            
            sim_matrix = torch.mm(feat_norm_p, prototypes_norm.t())
            
            with torch.no_grad():
                probs = torch.softmax(output_dict['refined_logits'].float(), dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                pseudo_mask = (target == self.ignore_index) & (max_probs > 0.9)

            gt_mask = (target != self.ignore_index)
            valid_mask = gt_mask | pseudo_mask
            
            if valid_mask.sum() > 0:
                valid_sim = sim_matrix[valid_mask]
                mixed_target = target.clone()
                mixed_target[pseudo_mask] = pseudo_labels[pseudo_mask]
                valid_target_indices = mixed_target[valid_mask]
                target_sim = valid_sim.gather(1, valid_target_indices.unsqueeze(1)).squeeze()
                loss_dist = torch.mean(1.0 - target_sim)

        # 5. Boundary Loss (Robust Normalization)
        # -----------------------------------------------------------
        feat_inp = output_dict['input_jafar_feat'].float()
        
        # [CRITICAL FIX] Use eps=1e-6.
        # Even without Color Drop, flat geometry + dark textures can yield near-zero vectors.
        feat_inp = F.normalize(feat_inp, p=2, dim=-1, eps=1e-6)
        
        feat_inp_flat = feat_inp.view(B*N, -1)
        neighbor_inp = feat_inp_flat[k_idx_flat].view(B, N, K, -1)
        center_inp = feat_inp.view(B, N, -1).unsqueeze(2).expand(-1, -1, K, -1)
        
        diff_sq = (center_inp - neighbor_inp) ** 2
        
        # [FIX] Increase eps inside sqrt. 1e-8 is too small for stability in some cases.
        joint_diff = torch.sqrt(diff_sq.sum(dim=-1) + 1e-6)
        
        edge_score_pseudo = joint_diff.mean(dim=-1)
        
        target_bdy = torch.sigmoid((edge_score_pseudo - 0.15) * 20)
        pred_bdy_logits = output_dict['bdy_logits'].squeeze(1).float()
        loss_bdy = self.bce(pred_bdy_logits, target_bdy.detach())

        return loss_sup + \
               loss_aff * (self.lambda_aff * alpha) + \
               loss_rec * self.lambda_rec + \
               loss_dist * self.lambda_dist + \
               loss_bdy * self.lambda_bdy