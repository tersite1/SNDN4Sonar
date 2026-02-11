"""
Multi-Scale Object-Aware TDP (Teacher's Detection Performance) Loss

Implements spatial weighting based on GT bounding boxes to focus on
object-relevant regions in feature maps, addressing the issue of
background features dominating the loss in MLO (underwater mine) detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class MultiScaleObjectAwareTDPLoss(nn.Module):
    """
    Multi-Scale Object-Aware TDP Loss for SR4IR.

    Extends standard feature matching with:
    1. Multi-scale FPN feature alignment across multiple levels
    2. Object-aware spatial weighting based on GT bounding boxes

    Args:
        layer_weights (dict): Weight for each FPN level (e.g., {'0': 1.0, '1': 0.5, '2': 0.25, '3': 0.1})
        loss_weight (float): Overall loss weight
        criterion (str): Distance metric ('l1' or 'l2')
        object_weight (float): Weight boost for object regions (default: 5.0)
        background_weight (float): Weight for background regions (default: 1.0)
        spatial_sigma (float): Gaussian sigma for smooth weighting (default: 3.0)
    """

    def __init__(
        self,
        layer_weights: Dict[str, float],
        loss_weight: float = 1.0,
        criterion: str = 'l1',
        object_weight: float = 5.0,
        background_weight: float = 1.0,
        spatial_sigma: float = 1.5,
        use_input_norm: bool = False,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights
        self.object_weight = object_weight
        self.background_weight = background_weight
        self.spatial_sigma = spatial_sigma
        self.use_input_norm = use_input_norm

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f'{criterion} criterion not supported.')

    def _generate_spatial_weight_map(
        self,
        feat_shape: tuple,  # (B, C, H_feat, W_feat)
        targets: List[Dict[str, torch.Tensor]],
        orig_size: tuple,  # (H_img, W_img)
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate spatial weight map based on GT bounding boxes.

        Args:
            feat_shape: Shape of feature map (B, C, H_feat, W_feat)
            targets: List of target dicts with 'boxes' key [N, 4] in (x1, y1, x2, y2) format
            orig_size: Original image size (H_img, W_img)
            device: Device

        Returns:
            weight_map: Tensor of shape (B, 1, H_feat, W_feat)
        """
        B, C, H_feat, W_feat = feat_shape
        H_img, W_img = orig_size

        # Initialize background weight
        weight_map = torch.full(
            (B, 1, H_feat, W_feat),
            self.background_weight,
            dtype=torch.float32,
            device=device
        )

        # Scale factors from image to feature map
        scale_h = H_feat / H_img
        scale_w = W_feat / W_img

        for batch_idx, target in enumerate(targets):
            if 'boxes' not in target or len(target['boxes']) == 0:
                continue

            boxes = target['boxes']  # [N, 4] in (x1, y1, x2, y2)

            # Convert boxes from image coords to feature coords
            boxes_feat = boxes.clone()
            boxes_feat[:, [0, 2]] *= scale_w  # x coords
            boxes_feat[:, [1, 3]] *= scale_h  # y coords

            # Clamp to feature map bounds
            boxes_feat[:, [0, 2]] = boxes_feat[:, [0, 2]].clamp(0, W_feat - 1)
            boxes_feat[:, [1, 3]] = boxes_feat[:, [1, 3]].clamp(0, H_feat - 1)

            # Create object mask for this batch
            for box in boxes_feat:
                x1, y1, x2, y2 = box.int()

                if x2 <= x1 or y2 <= y1:
                    continue

                # Option 1: Hard weighting (simple box mask)
                # weight_map[batch_idx, 0, y1:y2, x1:x2] = self.object_weight

                # Option 2: Gaussian-smoothed weighting (better gradient flow)
                # Create distance map from box center
                cy = (y1 + y2) / 2.0
                cx = (x1 + x2) / 2.0
                box_h = y2 - y1
                box_w = x2 - x1

                # Create coordinate grids
                y_coords = torch.arange(H_feat, device=device, dtype=torch.float32).view(-1, 1)
                x_coords = torch.arange(W_feat, device=device, dtype=torch.float32).view(1, -1)

                # Normalized distance from box center
                dy = (y_coords - cy) / (box_h / 2.0 + 1e-6)
                dx = (x_coords - cx) / (box_w / 2.0 + 1e-6)
                dist = torch.sqrt(dx**2 + dy**2)

                # Gaussian weight: higher near box center, decays smoothly
                gaussian = torch.exp(-dist**2 / (2 * self.spatial_sigma**2))

                # Blend: background_weight + (object_weight - background_weight) * gaussian
                box_weight = self.background_weight + \
                            (self.object_weight - self.background_weight) * gaussian

                # Take maximum (multiple boxes may overlap)
                weight_map[batch_idx, 0] = torch.maximum(
                    weight_map[batch_idx, 0],
                    box_weight
                )

        return weight_map

    def forward(
        self,
        feats_sr: Dict[str, torch.Tensor],
        feats_hr: Dict[str, torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        orig_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Forward pass with multi-scale object-aware feature matching.

        Args:
            feats_sr: SR features {'0': Tensor[B,C,H,W], '1': ..., }
            feats_hr: HR features {'0': Tensor[B,C,H,W], '1': ..., }
            targets: List of GT dicts with 'boxes' key (optional, for object-aware weighting)
            orig_size: Original image size (H, W) for coordinate mapping

        Returns:
            Weighted feature loss
        """
        if self.loss_weight <= 0:
            return torch.tensor(0.0, device=list(feats_sr.values())[0].device)

        feature_loss = 0
        use_object_aware = (targets is not None and orig_size is not None)

        for layer_key in self.layer_weights.keys():
            if layer_key not in feats_sr or layer_key not in feats_hr:
                continue

            feat_sr = feats_sr[layer_key]
            feat_hr = feats_hr[layer_key]

            # Handle different feature types
            if isinstance(feat_hr, list):
                # List of features (rare case)
                for idx in range(len(feat_sr)):
                    loss = self.criterion(feat_sr[idx], feat_hr[idx])
                    feature_loss += loss.mean() * self.layer_weights[layer_key]

            elif isinstance(feat_hr, dict):
                # Dict of features (nested FPN)
                for sub_key in feat_hr.keys():
                    loss = self.criterion(feat_sr[sub_key], feat_hr[sub_key])
                    feature_loss += loss.mean() * self.layer_weights[layer_key]

            else:
                # Standard case: Tensor [B, C, H, W]
                # Compute element-wise loss
                loss = self.criterion(feat_sr, feat_hr)  # [B, C, H, W]

                if use_object_aware:
                    # Generate spatial weight map
                    weight_map = self._generate_spatial_weight_map(
                        feat_shape=feat_sr.shape,
                        targets=targets,
                        orig_size=orig_size,
                        device=feat_sr.device
                    )  # [B, 1, H, W]

                    # Apply spatial weighting
                    weighted_loss = loss * weight_map  # Broadcasting: [B,C,H,W] * [B,1,H,W]
                    layer_loss = weighted_loss.mean()
                else:
                    # Uniform weighting (original behavior)
                    layer_loss = loss.mean()

                feature_loss += layer_loss * self.layer_weights[layer_key]

        return feature_loss * self.loss_weight


class FeatureLoss(nn.Module):
    """
    Legacy FeatureLoss for backward compatibility.

    Simple multi-scale feature matching without object-aware weighting.
    """

    def __init__(
        self,
        layer_weights: Dict[str, float],
        loss_weight: float = 1.0,
        criterion: str = 'l1',
        use_input_norm: bool = False,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError(f'{criterion} criterion not supported.')

    def forward(
        self,
        feats_sr: Dict[str, torch.Tensor],
        feats_hr: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Standard feature matching across multiple layers."""
        if self.loss_weight <= 0:
            return torch.tensor(0.0, device=list(feats_sr.values())[0].device)

        feature_loss = 0
        for layer_key in self.layer_weights.keys():
            if layer_key not in feats_sr or layer_key not in feats_hr:
                continue

            feat_sr = feats_sr[layer_key]
            feat_hr = feats_hr[layer_key]

            if isinstance(feat_hr, list):
                for idx in range(len(feat_sr)):
                    feature_loss += self.criterion(feat_sr[idx], feat_hr[idx]) * self.layer_weights[layer_key]
            elif isinstance(feat_hr, dict):
                for sub_key in feat_hr.keys():
                    feature_loss += self.criterion(feat_sr[sub_key], feat_hr[sub_key]) * self.layer_weights[layer_key]
            else:
                feature_loss += self.criterion(feat_sr, feat_hr) * self.layer_weights[layer_key]

        return feature_loss * self.loss_weight
