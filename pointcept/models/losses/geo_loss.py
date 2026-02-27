import torch
import torch.nn as nn
from .builder import LOSSES

@LOSSES.register_module()
class GeoCoTrainLoss(nn.Module):
    def __init__(self, 
                 lambda_main=1.0, 
                 lambda_aux=1.0,    
                 lambda_rec=1.0,    
                 ignore_index=255,
                 class_weights=None):
        super().__init__()
        
        self.lambda_main = lambda_main 
        self.lambda_aux = lambda_aux    
        self.lambda_rec = lambda_rec
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.mse = nn.MSELoss()

    def forward(self, output_dict):
        target = output_dict['target']
        
        # 1. Main & Aux Supervision
        # -----------------------------------------------------------
        loss_main = self.ce(output_dict['seg_logits'], target)
        loss_aux = self.ce(output_dict['aux_logits'], target)
        
        loss_sup = self.lambda_main * loss_main + self.lambda_aux * loss_aux
        
        # 2. Reconstruction Loss (Only on Query Points)
        # -----------------------------------------------------------
        loss_rec = torch.tensor(0.0, device=target.device)
        if 'rec_pred' in output_dict and 'rec_target' in output_dict:
            rec_pred = output_dict['rec_pred']
            rec_target = output_dict['rec_target']
            loss_rec = self.mse(rec_pred, rec_target)

        return loss_sup + loss_rec * self.lambda_rec