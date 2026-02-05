"""
Faster R-CNN detector for sonar mine detection using a MobileNetV3 backbone.

본 파일은 프로젝트 내부 모듈에 의존하지 않고, PyTorch/TorchVision만 사용해
독립적으로 동작하도록 작성되었습니다.
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import MultiScaleRoIAlign


class SonarFasterRCNNDetector(nn.Module):
    """
    TorchVision Faster R-CNN + MobileNetV3 백본.

    Forward 반환:
      - return_feats=False: (detections, loss_dict)
      - return_feats=True:  (detections, loss_dict, feat_dict)
        feat_dict['features']는 backbone/FPN 특징맵.
        FPN 사용 시 OrderedDict[str, Tensor] (각각 [B, C, H_l, W_l]).
        FPN 미사용 시 단일 Tensor [B, C, H, W].
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone_variant: str = "large",
        weights_backbone: Optional[
            Union[str, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights]
        ] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        trainable_backbone_layers: int = 6,
        freeze_backbone: bool = False,
        fpn: bool = True,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        min_size: int = 640,
        max_size: int = 640,
        anchor_sizes: Optional[Iterable[Tuple[int, ...]]] = None,
        aspect_ratios: Optional[Iterable[Tuple[float, ...]]] = None,
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 5000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 5000,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.001,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 1000,
        tdp_level: Optional[str] = None,
        rpn_only: bool = False,
    ) -> None:
        super().__init__()

        backbone_name = "mobilenet_v3_small" if backbone_variant.lower() == "small" else "mobilenet_v3_large"
        backbone = mobilenet_backbone(
            backbone_name,
            weights=weights_backbone,
            trainable_layers=trainable_backbone_layers,
            fpn=fpn,
        )

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, 3, min_size, min_size)
            feat_out = backbone(dummy)
        if isinstance(feat_out, dict):
            featmap_names = list(feat_out.keys())
        else:
            featmap_names = ["0"]
        num_levels = len(featmap_names)

        base_sizes = [(8,), (16,), (32,), (64, 128)]
        if anchor_sizes is None:
            sizes_norm = []
            for i in range(num_levels):
                if i < len(base_sizes):
                    sizes_norm.append(base_sizes[i])
                else:
                    sizes_norm.append(base_sizes[-1])
            anchor_sizes = tuple(sizes_norm)
        else:
            sizes_list = list(anchor_sizes)
            if len(sizes_list) < num_levels:
                sizes_list += [sizes_list[-1]] * (num_levels - len(sizes_list))
            anchor_sizes = tuple(sizes_list[:num_levels])

        if aspect_ratios is None:
            aspect_ratios = ((0.5, 1.0, 2.0),) * num_levels
        else:
            ratios_list = list(aspect_ratios)
            if len(ratios_list) < num_levels:
                ratios_list += [ratios_list[-1]] * (num_levels - len(ratios_list))
            aspect_ratios = tuple(ratios_list[:num_levels])

        anchor_generator = AnchorGenerator(
            sizes=tuple(anchor_sizes),
            aspect_ratios=tuple(aspect_ratios),
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=featmap_names if fpn else ["0"],
            output_size=7,
            sampling_ratio=2,
        )

        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=rpn_nms_thresh,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
        )
        self.tdp_level = tdp_level
        self.rpn_only = rpn_only

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
        return_feats: bool = False,
    ):
        """
        Args:
            images: List[[C,H,W]], float32, 0~1 (C=1 or 3; 1채널이면 내부에서 3채널로 변환)
            targets: 학습 시 GT dict 리스트
            return_feats: True면 feat_dict도 반환
        """
        # Convert 1-channel to 3-channel (MobileNetV3 expects 3-channel RGB)
        images = [img.repeat(3, 1, 1) if img.size(0) == 1 else img for img in images]

        original_image_sizes = [img.shape[-2:] for img in images]
        images_list: ImageList
        images_list, targets = self.detector.transform(images, targets)

        features = self.detector.backbone(images_list.tensors)
        if isinstance(features, torch.Tensor):
            features_for_heads = {"0": features}
        else:
            features_for_heads = features

        proposals, proposal_losses = self.detector.rpn(images_list, features_for_heads, targets)

        if self.rpn_only:
            detections = []
            for props in proposals:
                if props.numel() == 0:
                    detections.append(
                        {
                            "boxes": torch.empty((0, 4), device=props.device),
                            "scores": torch.empty((0,), device=props.device),
                            "labels": torch.empty((0,), dtype=torch.long, device=props.device),
                        },
                    )
                    continue
                # RPN objectness scores are stored in images_list.output ?: not exposed directly.
                # A lightweight workaround: assign a dummy descending score to preserve ordering.
                scores = torch.linspace(1.0, 0.0, steps=props.shape[0], device=props.device)
                labels = torch.ones((props.shape[0],), dtype=torch.long, device=props.device)
                detections.append({"boxes": props, "scores": scores, "labels": labels})
            detector_losses = {}
            # For rpn_only and fixed 640 size, skip postprocess to avoid redundant transforms.
        else:
            detections, detector_losses = self.detector.roi_heads(
                features_for_heads, proposals, images_list.image_sizes, targets
            )
            detections = self.detector.transform.postprocess(
                detections, images_list.image_sizes, original_image_sizes
            )

        if return_feats:
            if self.tdp_level is not None and self.tdp_level in features_for_heads:
                feat_out = {"0": features_for_heads[self.tdp_level]}
            else:
                feat_out = features_for_heads
            feat_dict = {"features": feat_out}
        else:
            feat_dict = None

        losses: Dict[str, Tensor] = {}
        if self.training and targets is not None:
            losses.update(detector_losses)
            losses.update(proposal_losses)
        else:
            losses = {}

        if return_feats:
            return detections, losses, feat_dict
        return detections, losses


def build_network(**kwargs) -> nn.Module:
    return SonarFasterRCNNDetector(**kwargs)
