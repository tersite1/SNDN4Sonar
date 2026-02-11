#!/usr/bin/env python3
"""
SonarSR 패치들에 대한 Faster R-CNN 디텍터 평가
- mAP(50-95), mAP50, mAP25 계산
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision.ops import box_iou
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.archs.det.sonarfasterrcnndetector_arch import SonarFasterRCNNDetector


def parse_yolo_label(label_path: Path, img_w: int, img_h: int):
    """
    YOLO 라벨 파싱 (normalized cx, cy, w, h → xyxy)
    Returns: [(class_id, x1, y1, x2, y2), ...]
    """
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            # Convert to absolute xyxy
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h

            boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def compute_ap(recalls, precisions):
    """Compute Average Precision using 101-point interpolation"""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Calculate area under PR curve
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap


def compute_map_at_iou(all_detections, all_groundtruths, iou_threshold=0.5, num_classes=3):
    """
    Compute mAP at specific IoU threshold

    Args:
        all_detections: List of detections per image [(boxes, scores, labels), ...]
        all_groundtruths: List of GT boxes per image [(boxes, labels), ...]
        iou_threshold: IoU threshold for TP
        num_classes: Number of classes (including background)
    """
    aps = []

    for class_id in range(1, num_classes):  # Skip background (class 0)
        # Collect all detections and GT for this class
        det_boxes = []
        det_scores = []
        det_image_ids = []
        gt_count = 0

        for img_id, (dets, gts) in enumerate(zip(all_detections, all_groundtruths)):
            det_box, det_score, det_label = dets
            gt_box, gt_label = gts

            # Filter by class
            if len(det_label) > 0:
                class_mask = (det_label == class_id)
                det_boxes.append(det_box[class_mask])
                det_scores.append(det_score[class_mask])
                det_image_ids.extend([img_id] * class_mask.sum().item())

            if len(gt_label) > 0:
                gt_count += (gt_label == class_id).sum().item()

        if gt_count == 0:
            continue

        # Concatenate all detections
        if len(det_boxes) == 0:
            aps.append(0.0)
            continue

        det_boxes = torch.cat(det_boxes, dim=0)
        det_scores = torch.cat(det_scores, dim=0)
        det_image_ids = torch.tensor(det_image_ids, device=det_boxes.device)

        # Sort by confidence
        sorted_indices = torch.argsort(det_scores, descending=True)
        det_boxes = det_boxes[sorted_indices]
        det_scores = det_scores[sorted_indices]
        det_image_ids = det_image_ids[sorted_indices]

        # Match detections to GT
        tp = torch.zeros(len(det_boxes))
        fp = torch.zeros(len(det_boxes))

        # Track matched GT boxes
        matched_gt = {}

        for det_idx, (det_box, img_id) in enumerate(zip(det_boxes, det_image_ids)):
            img_id = img_id.item()
            gt_box, gt_label = all_groundtruths[img_id]

            if len(gt_label) == 0:
                fp[det_idx] = 1
                continue

            # Filter GT by class
            class_mask = (gt_label == class_id)
            gt_box_class = gt_box[class_mask]

            if len(gt_box_class) == 0:
                fp[det_idx] = 1
                continue

            # Compute IoU
            ious = box_iou(det_box.unsqueeze(0), gt_box_class)[0]
            max_iou, max_idx = ious.max(dim=0)

            # Check if match
            if max_iou >= iou_threshold:
                # Check if GT already matched
                gt_global_idx = (img_id, class_mask.nonzero(as_tuple=True)[0][max_idx].item())
                if gt_global_idx not in matched_gt:
                    tp[det_idx] = 1
                    matched_gt[gt_global_idx] = True
                else:
                    fp[det_idx] = 1
            else:
                fp[det_idx] = 1

        # Compute precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        recalls = tp_cumsum / gt_count
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP
        ap = compute_ap(recalls.cpu().numpy(), precisions.cpu().numpy())
        aps.append(ap)

    # Return mean AP
    if len(aps) == 0:
        return 0.0
    return np.mean(aps)


@torch.no_grad()
def evaluate_detector(
    detector: nn.Module,
    sr_dir: Path,
    label_dir: Path,
    device: torch.device,
    conf_threshold: float = 0.001,
    suffix: str = "_sr",
    save_samples: int = 0,
    output_dir: Path = None,
):
    """
    Faster R-CNN 디텍터로 SR 패치들 평가

    Args:
        suffix: 파일 suffix (e.g., "_sr" for SR patches, "" for HR/noisy patches)
        save_samples: 저장할 샘플 개수 (0이면 저장 안 함)
        output_dir: 샘플 저장 디렉토리

    Returns:
        mAP(50-95), mAP50, mAP25
    """
    detector.eval()

    # Find all patches with given suffix
    pattern = f"*{suffix}.png" if suffix else "*.png"
    sr_files = sorted(sr_dir.glob(pattern))
    print(f"Found {len(sr_files)} patches (pattern: {pattern})")

    # Create output directory if saving samples
    if save_samples > 0 and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Will save {save_samples} sample images to {output_dir}")

    all_detections = []
    all_groundtruths = []
    saved_count = 0

    skipped = 0
    for idx, sr_path in enumerate(tqdm(sr_files, desc="Evaluating")):
        # Load SR image
        try:
            img = Image.open(sr_path).convert("L")  # Grayscale
            img_w, img_h = img.size
            # Force load to catch truncated images
            img.load()
        except Exception as e:
            skipped += 1
            continue

        # Convert to tensor (1-channel grayscale)
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W]
        img_tensor = img_tensor.to(device)

        # Get label path (remove suffix if present)
        stem = sr_path.stem
        if suffix and stem.endswith(suffix):
            stem = stem[:-len(suffix)]
        label_name = stem + ".txt"
        label_path = label_dir / label_name

        # Parse GT boxes
        gt_boxes_list = parse_yolo_label(label_path, img_w, img_h)

        if len(gt_boxes_list) > 0:
            gt_boxes = torch.tensor([[b[1], b[2], b[3], b[4]] for b in gt_boxes_list],
                                    dtype=torch.float32, device=device)
            gt_labels = torch.tensor([b[0] + 1 for b in gt_boxes_list],
                                     dtype=torch.int64, device=device)  # +1 for background
        else:
            gt_boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
            gt_labels = torch.empty((0,), dtype=torch.int64, device=device)

        # Run detector (expects List[Tensor])
        detections, losses = detector([img_tensor])
        det = detections[0]

        # Filter by confidence
        keep = det["scores"] > conf_threshold
        det_boxes = det["boxes"][keep]
        det_scores = det["scores"][keep]
        det_labels = det["labels"][keep]

        all_detections.append((det_boxes, det_scores, det_labels))
        all_groundtruths.append((gt_boxes, gt_labels))

        # Save visualization samples
        if save_samples > 0 and saved_count < save_samples and output_dir is not None:
            # Convert grayscale to RGB for visualization
            img_vis = Image.open(sr_path).convert("RGB")
            draw = ImageDraw.Draw(img_vis)

            # Draw GT boxes (green)
            for gt_box in gt_boxes:
                x1, y1, x2, y2 = gt_box.cpu().tolist()
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

            # Draw prediction boxes (red) with confidence scores
            for pred_box, pred_score, pred_label in zip(det_boxes, det_scores, det_labels):
                x1, y1, x2, y2 = pred_box.cpu().tolist()
                score = pred_score.cpu().item()
                label = pred_label.cpu().item()

                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # Draw label and score
                text = f"C{label}: {score:.2f}"
                # Simple text without font (PIL default)
                draw.text((x1, y1 - 10), text, fill="red")

            # Save image
            save_name = sr_path.name
            save_path = output_dir / save_name
            img_vis.save(save_path)
            saved_count += 1

    if save_samples > 0 and output_dir is not None:
        print(f"\n✓ Saved {saved_count} visualization samples to {output_dir}")

    if skipped > 0:
        print(f"\n⚠ Skipped {skipped} truncated/corrupted images")

    # Compute mAP at different IoU thresholds
    print("\nComputing mAP metrics...")

    # mAP50
    map50 = compute_map_at_iou(all_detections, all_groundtruths, iou_threshold=0.5)

    # mAP25
    map25 = compute_map_at_iou(all_detections, all_groundtruths, iou_threshold=0.25)

    # mAP(50-95) = average of mAP50, mAP55, ..., mAP95
    map_50_95 = []
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        ap = compute_map_at_iou(all_detections, all_groundtruths, iou_threshold=iou_thresh)
        map_50_95.append(ap)
    map_50_95 = np.mean(map_50_95)

    return map_50_95, map50, map25


def main():
    parser = argparse.ArgumentParser(description="Evaluate SonarSR patches with Faster R-CNN")
    parser.add_argument("--sr_dir", type=str, required=True, help="SR patches directory")
    parser.add_argument("--label_dir", type=str, required=True, help="YOLO labels directory")
    parser.add_argument("--detector_path", type=str, required=True, help="Detector checkpoint")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--conf_threshold", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--suffix", type=str, default="_sr", help="File suffix (e.g., '_sr' for SR patches, '' for HR)")
    parser.add_argument("--save_samples", type=int, default=0, help="Number of visualization samples to save (0=none)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for visualization samples")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load detector
    print(f"\nLoading detector from {args.detector_path}")
    detector = SonarFasterRCNNDetector(
        backbone_variant="large",
        num_classes=args.num_classes,
    )

    checkpoint = torch.load(args.detector_path, map_location=device)
    if "model_state_dict" in checkpoint:
        detector.load_state_dict(checkpoint["model_state_dict"])
    else:
        detector.load_state_dict(checkpoint)

    detector = detector.to(device)
    print("✓ Detector loaded")

    # Evaluate
    sr_dir = Path(args.sr_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    map_50_95, map50, map25 = evaluate_detector(
        detector=detector,
        sr_dir=sr_dir,
        label_dir=label_dir,
        device=device,
        conf_threshold=args.conf_threshold,
        suffix=args.suffix,
        save_samples=args.save_samples,
        output_dir=output_dir,
    )

    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results on SonarSR Patches")
    print("=" * 70)
    print(f"mAP(50-95): {map_50_95 * 100:.2f}%")
    print(f"mAP50:      {map50 * 100:.2f}%")
    print(f"mAP25:      {map25 * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
