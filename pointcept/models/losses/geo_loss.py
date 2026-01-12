import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

@LOSSES.register_module()
class GeoCoTrainLoss(nn.Module):
    def __init__(self, 
                 lambda_main=1.0,   #  Weight for Refined Logits (JAFAR/Final)
                 lambda_aux=1.0,    #  Weight for Aux Logits (PTv3/Backbone)
                 lambda_aff=0.1,    
                 lambda_dist=0.1,   
                 lambda_bdy=0.5,    
                 warmup_epochs=0,   # Reserved parameter for potential future scheduling
                 ignore_index=255,
                 class_weights=None):
        super().__init__()
        
        self.lambda_main = lambda_main 
        self.lambda_aux = lambda_aux   
        
        self.lambda_aff = lambda_aff
        self.lambda_dist = lambda_dist
        self.lambda_bdy = lambda_bdy
        self.ignore_index = ignore_index
        
        # [MODIFIED] Weighted Cross Entropy
        # Handle class imbalance for weak supervision
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.bce = nn.BCEWithLogitsLoss()

        # [New] Dynamic Weighting State
        # Register a buffer to track iterations (automatically saved/loaded with checkpoint)
        self.register_buffer('iter_step', torch.tensor(0, dtype=torch.long))
        # Estimate warmup steps: e.g., warmup_epochs * 500 steps per epoch
        # This controls how fast lambda_aff grows from 0 to its full value.
        self.warmup_steps = warmup_epochs * 500

    def forward(self, output_dict):
        target = output_dict['target']
        
        # [New] Update Iteration Step & Calculate Dynamic Alpha
        if self.training:
            self.iter_step += 1
        
        # Calculate alpha: grows linearly from 0.0 to 1.0 during warmup
        if self.warmup_steps > 0:
            alpha = min(1.0, self.iter_step.item() / float(self.warmup_steps))
        else:
            alpha = 1.0

        # 1. Dual Supervision (Refined + Aux)
        # -----------------------------------------------------------
        loss_main = self.ce(output_dict['refined_logits'], target)
        loss_aux = self.ce(output_dict['aux_logits'], target)
        loss_sup = self.lambda_main * loss_main + self.lambda_aux * loss_aux
        
        # 2. Affinity Loss (Soft-Thresholding)
        # -----------------------------------------------------------
        feat_to_constrain = output_dict['refined_feat']
        affinity = output_dict['affinity']
        k_idx = output_dict['k_idx']
        
        B, N, K = k_idx.shape
        C = feat_to_constrain.shape[-1]
        
        batch_offset = torch.arange(B, device=k_idx.device).view(B, 1, 1) * N
        k_idx_flat = (k_idx + batch_offset).view(-1)
        feat_flat = feat_to_constrain.view(B*N, C)
        
        neighbor_feat = feat_flat[k_idx_flat].view(B, N, K, C)
        center_feat = feat_to_constrain.view(B, N, C).unsqueeze(2).expand(-1, -1, K, -1)
        
        feat_dist = torch.sum((center_feat - neighbor_feat) ** 2, dim=-1) / (C ** 0.5)
        
        aff_weight = F.relu(affinity - 0.5) 
        loss_aff = torch.sum(aff_weight * feat_dist) / (aff_weight.sum() + 1e-4)

        # 3. Distribution Loss (Prototype Alignment with Pseudo-Labels)
        # -----------------------------------------------------------
        loss_dist = torch.tensor(0.0, device=target.device)
        if 'prototypes' in output_dict:
            prototypes = output_dict['prototypes']
            
            feat_norm = F.normalize(feat_flat, p=2, dim=1, eps=1e-6)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1, eps=1e-6)
            
            # Similarity Matrix [N, NumClasses]
            sim_matrix = torch.mm(feat_norm, prototypes_norm.t())
            
            # [New] Generate Pseudo-Labels for Unlabeled Points
            # -------------------------------------------------
            # 1. Get predictions confidence
            with torch.no_grad():
                # Use refined_logits for pseudo-labeling
                probs = torch.softmax(output_dict['refined_logits'], dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                
                # Criteria: Point is unlabeled (255) AND Confidence > 0.9
                pseudo_mask = (target == self.ignore_index) & (max_probs > 0.9)

            # 2. Combine GT and Pseudo-Labels mask
            gt_mask = (target != self.ignore_index)
            valid_mask = gt_mask | pseudo_mask
            
            if valid_mask.sum() > 0:
                # Select similarities for valid points
                valid_sim = sim_matrix[valid_mask]
                
                # Construct mixed target: Copy GT, then fill in Pseudo-Labels
                mixed_target = target.clone()
                mixed_target[pseudo_mask] = pseudo_labels[pseudo_mask]
                
                # Get the final target class indices for the valid subset
                valid_target_indices = mixed_target[valid_mask]
                
                # Maximize similarity to the target prototype (GT or Pseudo)
                # gather retrieves the score corresponding to the target class
                target_sim = valid_sim.gather(1, valid_target_indices.unsqueeze(1)).squeeze()
                loss_dist = torch.mean(1.0 - target_sim)

        # 4. Boundary Loss
        # -----------------------------------------------------------
        feat_inp = output_dict['input_jafar_feat'] 
        
        # [MODIFIED] De-coloring / Pure Geometric Boundary
        # feat_inp contains [GeoBlobs (9) | Color (3)].
        # We slice [:, :, :9] to use ONLY geometry for boundary detection.
        # This prevents the network from cheating by learning texture edges.
        feat_geo_only = feat_inp[:, :, :9] 
        
        feat_inp_flat = feat_geo_only.contiguous().view(B*N, -1)
        neighbor_inp = feat_inp_flat[k_idx_flat].view(B, N, K, -1)
        center_inp = feat_geo_only.view(B, N, -1).unsqueeze(2).expand(-1, -1, K, -1)
        
        joint_diff = torch.norm(center_inp - neighbor_inp, dim=-1)
        edge_score_pseudo = joint_diff.mean(dim=-1)
        
        target_bdy = torch.sigmoid((edge_score_pseudo - 0.15) * 20)
        pred_bdy_logits = output_dict['bdy_logits'].squeeze(1)
        loss_bdy = self.bce(pred_bdy_logits, target_bdy.detach())

        return loss_sup + \
               loss_aff * (self.lambda_aff * alpha) + \
               loss_dist * self.lambda_dist + \
               loss_bdy * self.lambda_bdy