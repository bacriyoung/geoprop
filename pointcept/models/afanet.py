"""AFA-Net: geometry-constrained asymmetric feature aggregation.

The implementation follows the paper terminology and computation graph:

* Confidence-Guided Query Selection (CQS) refines the least-confident points.
* Asymmetric Decoupled Attention (ADA) computes affinities only from
  coordinate--attribute inputs and relative geometry, while semantic backbone
  features are used only as values.
* Local Geometric Reconstruction (LGR) reconstructs the selected query input
  during training and is not evaluated at inference time.
"""

import inspect

import pointops
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS
from pointcept.models.losses import LOSSES
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)


class ConfidenceGuidedQuerySelection(nn.Module):
    """Partition each sample into low-confidence queries and non-query anchors."""

    def __init__(self, query_ratio=0.25):
        super().__init__()
        if not 0.0 < query_ratio < 1.0:
            raise ValueError("query_ratio must be in the open interval (0, 1)")
        self.query_ratio = query_ratio

    @staticmethod
    def _offset(indices, batch, batch_ids):
        subset_batch = batch[indices]
        counts = torch.stack([(subset_batch == batch_id).sum() for batch_id in batch_ids])
        return torch.cumsum(counts, dim=0).int()

    @torch.no_grad()
    def forward(self, pre_logits, batch):
        confidence = torch.softmax(pre_logits, dim=-1).amax(dim=-1)
        batch_ids = torch.unique(batch, sorted=True)
        query_parts = []
        anchor_parts = []

        for batch_id in batch_ids:
            sample_indices = torch.nonzero(batch == batch_id, as_tuple=False).flatten()
            sample_order = torch.argsort(confidence[sample_indices], descending=False)
            num_queries = max(1, int(sample_indices.numel() * self.query_ratio))
            num_queries = min(num_queries, sample_indices.numel() - 1)
            query_parts.append(torch.sort(sample_indices[sample_order[:num_queries]]).values)
            # The paper defines every non-query point as a candidate anchor.
            anchor_parts.append(torch.sort(sample_indices[sample_order[num_queries:]]).values)

        query_indices = torch.cat(query_parts)
        anchor_indices = torch.cat(anchor_parts)
        return {
            "query_indices": query_indices,
            "anchor_indices": anchor_indices,
            "query_offset": self._offset(query_indices, batch, batch_ids),
            "anchor_offset": self._offset(anchor_indices, batch, batch_ids),
        }


class AsymmetricDecoupledAttention(nn.Module):
    """ADA refinement with input-derived affinities and semantic values."""

    def __init__(self, input_dim, semantic_dim, attention_dim=64):
        super().__init__()
        if attention_dim % 8 != 0:
            raise ValueError("attention_dim must be divisible by 8 for GroupNorm")
        self.attention_dim = attention_dim

        self.input_encoder = nn.Sequential(
            nn.Conv1d(input_dim, attention_dim, 1),
            nn.GroupNorm(8, attention_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(attention_dim, attention_dim, 1),
            nn.GroupNorm(8, attention_dim),
            nn.ReLU(inplace=True),
        )
        self.query_projection = nn.Conv1d(attention_dim, attention_dim, 1)
        self.key_projection = nn.Conv1d(attention_dim, attention_dim, 1)
        self.semantic_value_projection = nn.Sequential(
            nn.Conv1d(semantic_dim, attention_dim, 1),
            nn.GroupNorm(8, attention_dim),
            nn.ReLU(inplace=True),
        )
        self.relative_position_encoder = nn.Sequential(
            nn.Conv2d(4, attention_dim, 1),
            nn.GroupNorm(8, attention_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, attention_dim, 1),
        )

    @staticmethod
    def _gather_neighbors(tensor, neighbor_indices):
        """Gather (B, C, A) anchor tensors into (B, C, Q, K)."""
        batch_size, channels, num_anchors = tensor.shape
        _, num_queries, num_neighbors = neighbor_indices.shape
        flattened = tensor.transpose(1, 2).contiguous().view(-1, channels)
        batch_offset = (
            torch.arange(batch_size, device=tensor.device).view(batch_size, 1, 1)
            * num_anchors
        )
        flattened_indices = (neighbor_indices + batch_offset).reshape(-1)
        return (
            flattened[flattened_indices]
            .view(batch_size, num_queries, num_neighbors, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(
        self,
        query_input,
        query_semantic,
        anchor_input,
        anchor_semantic,
        neighbor_indices,
    ):
        query_encoded = self.input_encoder(query_input.transpose(1, 2).contiguous())
        anchor_encoded = self.input_encoder(anchor_input.transpose(1, 2).contiguous())
        query_embedding = self.query_projection(query_encoded)
        anchor_keys = self.key_projection(anchor_encoded)
        neighbor_keys = self._gather_neighbors(anchor_keys, neighbor_indices)

        anchor_values = self.semantic_value_projection(
            anchor_semantic.transpose(1, 2).contiguous()
        )
        neighbor_values = self._gather_neighbors(anchor_values, neighbor_indices)

        query_xyz = query_input[..., :3].transpose(1, 2).contiguous().unsqueeze(-1)
        anchor_xyz = anchor_input[..., :3].transpose(1, 2).contiguous()
        neighbor_xyz = self._gather_neighbors(anchor_xyz, neighbor_indices)
        relative_xyz = query_xyz - neighbor_xyz
        relative_distance = torch.linalg.vector_norm(relative_xyz, dim=1, keepdim=True)
        position_encoding = self.relative_position_encoder(
            torch.cat([relative_xyz, relative_distance], dim=1)
        )

        affinity_logits = (
            query_embedding.unsqueeze(-1) * (neighbor_keys + position_encoding)
        ).sum(dim=1) / (self.attention_dim**0.5)
        affinity = torch.softmax(affinity_logits.float(), dim=-1).to(affinity_logits.dtype)

        aggregated_values = (affinity.unsqueeze(1) * neighbor_values).sum(dim=-1)
        query_residual = self.semantic_value_projection(
            query_semantic.transpose(1, 2).contiguous()
        )
        refined_query = aggregated_values + query_residual
        return refined_query.transpose(1, 2).contiguous(), affinity


class LocalGeometricReconstruction(nn.Module):
    """Training-only LGR decoder for coordinate--attribute reconstruction."""

    def __init__(self, feature_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, output_dim),
        )

    def forward(self, refined_query):
        return self.decoder(refined_query)


@MODELS.register_module()
class AFANet(nn.Module):
    """Attach AFA-Net once to input-resolution PTv3 output features."""

    def __init__(
        self,
        backbone_ptv3_cfg,
        input_dim=6,
        num_classes=13,
        criteria=None,
        query_ratio=0.25,
        neighborhood_size=16,
        attention_dim=64,
    ):
        super().__init__()
        if neighborhood_size < 1:
            raise ValueError("neighborhood_size must be positive")

        valid_params = inspect.signature(PointTransformerV3.__init__).parameters
        clean_cfg = {key: value for key, value in backbone_ptv3_cfg.items() if key in valid_params}
        self.backbone = PointTransformerV3(**clean_cfg)

        decoder_channels = backbone_ptv3_cfg.get("dec_channels", [48, 96, 192, 384])
        semantic_dim = decoder_channels[0]
        self.input_dim = input_dim
        self.neighborhood_size = neighborhood_size

        self.pre_refinement_head = nn.Linear(semantic_dim, num_classes)
        self.cqs = ConfidenceGuidedQuerySelection(query_ratio=query_ratio)
        self.ada = AsymmetricDecoupledAttention(
            input_dim=input_dim,
            semantic_dim=semantic_dim,
            attention_dim=attention_dim,
        )
        self.refined_segmentation_head = nn.Linear(attention_dim, num_classes)
        self.lgr = LocalGeometricReconstruction(attention_dim, input_dim)
        self.criteria = LOSSES.build(criteria) if criteria is not None else None

    def _forward_fragments(self, input_dict):
        fragment_list = input_dict["fragment_list"][0]
        full_segment = input_dict["segment"].view(-1)
        num_points = full_segment.shape[0]
        num_classes = self.pre_refinement_head.out_features
        device = next(self.parameters()).device
        accumulated = torch.zeros((num_points, num_classes), device=device)
        counts = torch.zeros((num_points, 1), device=device)

        for fragment in fragment_list:
            fragment = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in fragment.items()
            }
            probabilities = torch.softmax(self.forward(fragment)["seg_logits"], dim=-1)
            global_indices = fragment["index"].long()
            accumulated.index_add_(0, global_indices, probabilities)
            counts.index_add_(0, global_indices, torch.ones_like(probabilities[:, :1]))

        probabilities = accumulated / counts.clamp(min=1.0)
        full_segment = full_segment.to(device)
        loss = F.nll_loss(
            torch.log(probabilities.clamp(min=1e-6)),
            full_segment.long(),
            ignore_index=255,
        )
        return {
            "seg_logits": probabilities.permute(1, 0).unsqueeze(0),
            "target": full_segment,
            "loss": loss,
        }

    def forward(self, input_dict):
        if "fragment_list" in input_dict:
            return self._forward_fragments(input_dict)

        coordinates = input_dict["coord"]
        backbone_features = input_dict.get("ptv3_feat", input_dict.get("feat"))
        coordinate_attributes = input_dict.get("afanet_input")
        if coordinate_attributes is None:
            raise KeyError(
                "AFANet requires input_dict['afanet_input'] ordered as normalized XYZ "
                "followed by RGB or LiDAR intensity"
            )
        if coordinate_attributes.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected afanet_input with {self.input_dim} channels, got "
                f"{coordinate_attributes.shape[1]}"
            )

        batch = input_dict.get("batch")
        if batch is None and "offset" in input_dict:
            offset = input_dict["offset"].to(coordinates.device)
            sample_counts = offset.diff(prepend=offset.new_zeros(1)).long()
            batch = torch.repeat_interleave(
                torch.arange(offset.numel(), device=coordinates.device), sample_counts
            )
        if batch is None:
            batch = torch.zeros(
                coordinates.shape[0], device=coordinates.device, dtype=torch.long
            )
        grid_coordinates = input_dict.get("grid_coord", (coordinates / 0.02).int())
        semantic_features = self.backbone(
            {
                "coord": coordinates,
                "feat": backbone_features,
                "grid_coord": grid_coordinates,
                "batch": batch,
            }
        ).feat
        pre_logits = self.pre_refinement_head(semantic_features)

        selection = self.cqs(pre_logits, batch)
        query_indices = selection["query_indices"]
        anchor_indices = selection["anchor_indices"]
        anchor_counts = selection["anchor_offset"].diff(
            prepend=selection["anchor_offset"].new_zeros(1)
        )
        if torch.any(anchor_counts < self.neighborhood_size):
            raise ValueError("Each sample must contain at least neighborhood_size anchors")

        neighbor_indices = pointops.knn_query(
            self.neighborhood_size,
            coordinates[anchor_indices],
            selection["anchor_offset"],
            coordinates[query_indices],
            selection["query_offset"],
        )[0].long()

        refined_query, affinity = self.ada(
            query_input=coordinate_attributes[query_indices].unsqueeze(0),
            query_semantic=semantic_features[query_indices].unsqueeze(0),
            anchor_input=coordinate_attributes[anchor_indices].unsqueeze(0),
            anchor_semantic=semantic_features[anchor_indices].unsqueeze(0),
            neighbor_indices=neighbor_indices.unsqueeze(0),
        )
        refined_query = refined_query.squeeze(0)
        refined_logits = self.refined_segmentation_head(refined_query)
        final_logits = pre_logits.clone()
        final_logits[query_indices] = refined_logits

        output = {
            "seg_logits": final_logits,
            "pre_logits": pre_logits,
            "query_indices": query_indices,
            "affinity": affinity.squeeze(0),
            "target": input_dict.get("segment"),
        }
        # LGR is a training-only regularizer and incurs no inference computation.
        if self.training:
            output["reconstruction"] = self.lgr(refined_query)
            output["reconstruction_target"] = coordinate_attributes[query_indices]

        if self.criteria is not None and output["target"] is not None:
            output["loss"] = self.criteria(output)
        elif self.criteria is not None:
            output["loss"] = final_logits.new_zeros(())
        return output
