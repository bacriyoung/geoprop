"""Objective function for AFA-Net."""

import torch
import torch.nn as nn

from .builder import LOSSES


@LOSSES.register_module()
class AFANetLoss(nn.Module):
    def __init__(
        self,
        lambda_main=1.0,
        lambda_aux=1.0,
        lambda_rec=10.0,
        ignore_index=255,
        class_weights=None,
    ):
        super().__init__()
        self.lambda_main = lambda_main
        self.lambda_aux = lambda_aux
        self.lambda_rec = lambda_rec
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights))
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=self.class_weights, ignore_index=ignore_index
            )
        else:
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, output_dict):
        target = output_dict["target"]
        main_loss = self.cross_entropy(output_dict["seg_logits"], target)
        auxiliary_loss = self.cross_entropy(output_dict["pre_logits"], target)
        reconstruction_loss = target.new_zeros((), dtype=torch.float32)
        if "reconstruction" in output_dict:
            reconstruction_loss = self.reconstruction_loss(
                output_dict["reconstruction"],
                output_dict["reconstruction_target"],
            )
        return (
            self.lambda_main * main_loss
            + self.lambda_aux * auxiliary_loss
            + self.lambda_rec * reconstruction_loss
        )
