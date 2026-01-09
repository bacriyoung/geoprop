import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

@LOSSES.register_module()
class GeoCoTrainLoss(nn.Module):
    def __init__(self, 
                 lambda_main=1.0,   
                 lambda_aux=1.0,    
                 lambda_aff=0.1,    
                 lambda_dist=0.1,   
                 lambda_bdy=0.5,    
                 warmup_epochs=0,   
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
        
        # ===========================================================
        # 1. Point-Level Supervision (N points)
        # ===========================================================
        # Main Semantic Loss
        loss_main = self.ce(output_dict['refined_logits'], target)
        
        # Aux Backbone Loss
        loss_aux = self.ce(output_dict['aux_logits'], target)
        
        loss_sup = self.lambda_main * loss_main + self.lambda_aux * loss_aux
        
        # ===========================================================
        # 2. Voxel-Level Affinity Loss (M voxels)
        # Optimized: Operates on sparse voxels directly
        # ===========================================================
        # Check if we have voxel-level outputs (New Architecture)
        if 'voxel_feat' in output_dict:
            feat_M = output_dict['voxel_feat']       # [M, C]
            affinity_M = output_dict['voxel_affinity'] # [M, K] or [M, 1, K]
            k_idx_M = output_dict['voxel_k_idx']       # [M, K] (Indices are 0..M-1)
            
            if affinity_M.dim() == 3: affinity_M = affinity_M.squeeze(1)
            
            M, K = k_idx_M.shape
            C = feat_M.shape[-1]
            
            # 1. Gather Neighbor Features
            # Note: k_idx_M already contains absolute indices into feat_M (Packed)
            # No batch_offset needed because M implies flattened unique voxels
            neighbor_feat = feat_M[k_idx_M.view(-1)].view(M, K, C) # [M, K, C]
            center_feat = feat_M.unsqueeze(1).expand(-1, K, -1)    # [M, K, C]
            
            # 2. L2 Distance between voxel and its neighbors
            feat_dist = torch.sum((center_feat - neighbor_feat) ** 2, dim=-1) / (C ** 0.5)
            
            # 3. Soft-Thresholding Weighting
            # Only pull features if affinity is high (>0.5)
            aff_weight = F.relu(affinity_M - 0.5)
            
            loss_aff = torch.sum(aff_weight * feat_dist) / (aff_weight.sum() + 1e-4)
            
        else:
            loss_aff = torch.tensor(0.0, device=target.device) # Fallback

        # ===========================================================
        # 3. Point-Level Distribution Loss (Prototypes)
        # ===========================================================
        loss_dist = torch.tensor(0.0, device=target.device)
        if 'prototypes' in output_dict:
            prototypes = output_dict['prototypes']
            feat_flat = output_dict['refined_feat'] # Use Point features [N, C] for alignment with GT
            
            # Normalize with eps to prevent NaN in FP16
            feat_norm = F.normalize(feat_flat, p=2, dim=1, eps=1e-6)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1, eps=1e-6)
            
            sim_matrix = torch.mm(feat_norm, prototypes_norm.t())
            
            valid_mask = (target != self.ignore_index)
            if valid_mask.sum() > 0:
                valid_sim = sim_matrix[valid_mask]
                valid_target = target[valid_mask]
                target_sim = valid_sim.gather(1, valid_target.unsqueeze(1)).squeeze()
                loss_dist = torch.mean(1.0 - target_sim)

        # ===========================================================
        # 4. Voxel-Level Boundary Loss
        # ===========================================================
        if 'voxel_input_feat' in output_dict:
            inp_M = output_dict['voxel_input_feat']   # [M, 6] (RGB+Offset)
            k_idx_M = output_dict['voxel_k_idx']      # [M, K]
            pred_bdy_M = output_dict['voxel_bdy_logits'].squeeze(-1) # [M]
            
            M, K = k_idx_M.shape
            
            # Gather Input Features for neighbors
            neighbor_inp = inp_M[k_idx_M.view(-1)].view(M, K, -1)
            center_inp = inp_M.unsqueeze(1).expand(-1, K, -1)
            
            # Compute geometric difference (Pseudo Edge)
            joint_diff = torch.norm(center_inp - neighbor_inp, dim=-1)
            edge_score_pseudo = joint_diff.mean(dim=-1) # [M]
            
            # Create Soft Target
            target_bdy = torch.sigmoid((edge_score_pseudo - 0.15) * 20)
            
            loss_bdy = self.bce(pred_bdy_M, target_bdy.detach())
        else:
            loss_bdy = torch.tensor(0.0, device=target.device)

        # ===========================================================
        # Total Loss
        # ===========================================================
        return loss_sup + \
               loss_aff * self.lambda_aff + \
               loss_dist * self.lambda_dist + \
               loss_bdy * self.lambda_bdy