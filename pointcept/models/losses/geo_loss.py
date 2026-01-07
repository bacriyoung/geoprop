import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

@LOSSES.register_module()
class GeoCoTrainLoss(nn.Module):
    def __init__(self, 
                 lambda_main=1.0,   # Renamed from lambda_sup: Weight for Refined Logits (JAFAR/Final)
                 lambda_aux=1.0,    # Renamed from lambda_con: Weight for Aux Logits (PTv3/Backbone)
                 lambda_aff=0.1,    
                 lambda_dist=0.1,   
                 lambda_bdy=0.5,    
                 warmup_epochs=0,   # Reserved parameter for potential future scheduling
                 ignore_index=255):
        super().__init__()
        
        self.lambda_main = lambda_main 
        self.lambda_aux = lambda_aux   
        
        self.lambda_aff = lambda_aff
        self.lambda_dist = lambda_dist
        self.lambda_bdy = lambda_bdy
        self.ignore_index = ignore_index
        
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, output_dict):
        target = output_dict['target']
        
        # 1. Dual Supervision (Refined + Aux)
        # -----------------------------------------------------------
        # Refined: JAFAR output (Final Prediction)
        loss_main = self.ce(output_dict['refined_logits'], target)
        
        # Aux: PTv3 output (Backbone Supervision)
        # Force PTv3 to learn robust features independently, 
        # ensuring it doesn't rely solely on JAFAR refinement.
        loss_aux = self.ce(output_dict['aux_logits'], target)
        
        loss_sup = self.lambda_main * loss_main + self.lambda_aux * loss_aux
        
        # 2. Affinity Loss (Soft-Thresholding)
        # -----------------------------------------------------------
        # Objective: If JAFAR predicts high affinity between points, 
        # their Refined features must be close in embedding space.
        feat_to_constrain = output_dict['refined_feat'] # [B*N, C]
        affinity = output_dict['affinity'] # [B, N, K]
        k_idx = output_dict['k_idx']
        
        B, N, K = k_idx.shape
        C = feat_to_constrain.shape[-1]
        
        # Gather Neighbor Features
        batch_offset = torch.arange(B, device=k_idx.device).view(B, 1, 1) * N
        k_idx_flat = (k_idx + batch_offset).view(-1)
        feat_flat = feat_to_constrain.view(B*N, C)
        
        neighbor_feat = feat_flat[k_idx_flat].view(B, N, K, C)
        center_feat = feat_to_constrain.view(B, N, C).unsqueeze(2).expand(-1, -1, K, -1)
        
        # L2 Distance
        feat_dist = torch.sum((center_feat - neighbor_feat) ** 2, dim=-1) / (C ** 0.5)
        
        # [Strategy] Soft-Thresholding
        # Only points with affinity > 0.5 generate pull force.
        # Higher affinity results in stronger pull force.
        aff_weight = F.relu(affinity - 0.5) 
        
        # Normalize by sum of weights to avoid scaling issues
        loss_aff = torch.sum(aff_weight * feat_dist) / (aff_weight.sum() + 1e-4)

        # 3. Distribution Loss (Prototype Alignment)
        # -----------------------------------------------------------
        loss_dist = torch.tensor(0.0, device=target.device)
        if 'prototypes' in output_dict:
            prototypes = output_dict['prototypes']
            feat_norm = F.normalize(feat_flat, p=2, dim=1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            
            # Maximize similarity between feature and its class prototype
            sim_matrix = torch.mm(feat_norm, prototypes_norm.t())
            
            valid_mask = (target != self.ignore_index)
            if valid_mask.sum() > 0:
                valid_sim = sim_matrix[valid_mask]
                valid_target = target[valid_mask]
                # Gather score of the correct class
                target_sim = valid_sim.gather(1, valid_target.unsqueeze(1)).squeeze()
                loss_dist = torch.mean(1.0 - target_sim)

        # 4. Boundary Loss (Geometry Self-Supervision)
        # -----------------------------------------------------------
        # Use the 12-dim input to determine boundaries
        feat_inp = output_dict['input_jafar_feat'] 
        feat_inp_flat = feat_inp.view(B*N, -1)
        neighbor_inp = feat_inp_flat[k_idx_flat].view(B, N, K, -1)
        center_inp = feat_inp.unsqueeze(2).expand(-1, -1, K, -1)
        
        # Compute "Pseudo Edge" based on input difference
        joint_diff = torch.norm(center_inp - neighbor_inp, dim=-1)
        edge_score_pseudo = joint_diff.mean(dim=-1)
        
        # Sigmoid to create soft target (0~1)
        target_bdy = torch.sigmoid((edge_score_pseudo - 0.15) * 20)
        
        pred_bdy_logits = output_dict['bdy_logits'].squeeze(1)
        loss_bdy = self.bce(pred_bdy_logits, target_bdy.detach())

        # Total Loss
        return loss_sup + \
               loss_aff * self.lambda_aff + \
               loss_dist * self.lambda_dist + \
               loss_bdy * self.lambda_bdy