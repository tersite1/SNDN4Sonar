"""
Evaluate detector on HR and SR images
Usage:
    python eval_detector.py --hr_dir /path/to/hr --sr_dir /path/to/sr --label_dir /path/to/labels
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, 'src')
from archs.det.sonarfasterrcnndetector_arch import build_network as build_detector


def load_yolo_labels(label_path, img_size=640):
    """Load YOLO format labels and convert to pixel coordinates"""
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0]) + 1  # label_offset = 1
            cx, cy, w, h = map(float, parts[1:5])

            # Convert normalized to pixel
            x1 = (cx - w/2) * img_size
            y1 = (cy - h/2) * img_size
            x2 = (cx + w/2) * img_size
            y2 = (cy + h/2) * img_size

            boxes.append([x1, y1, x2, y2])
            labels.append(cls)

    if len(boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """Compute Average Precision at given IoU threshold"""
    # Collect all predictions with scores
    all_preds = []
    all_gts = {i: gt for i, gt in enumerate(ground_truths)}
    gt_matched = {i: [False] * len(gt['boxes']) for i, gt in enumerate(ground_truths)}

    for img_idx, pred in enumerate(predictions):
        for box_idx in range(len(pred['boxes'])):
            all_preds.append({
                'img_idx': img_idx,
                'box': pred['boxes'][box_idx],
                'score': pred['scores'][box_idx],
                'label': pred['labels'][box_idx]
            })

    # Sort by score descending
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    # Count total GT boxes
    total_gt = sum(len(gt['boxes']) for gt in ground_truths)
    if total_gt == 0:
        return 0.0

    # Compute precision-recall
    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for pred in all_preds:
        img_idx = pred['img_idx']
        pred_box = pred['box']
        pred_label = pred['label']

        gt = all_gts[img_idx]
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt['boxes'], gt['labels'])):
            if gt_matched[img_idx][gt_idx]:
                continue
            if gt_label != pred_label:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[img_idx][best_gt_idx] = True
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / total_gt
        precisions.append(precision)
        recalls.append(recall)

    # Compute AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        p = max([p for p, r in zip(precisions, recalls) if r >= t], default=0)
        ap += p / 11

    return ap


def evaluate_detector(detector, image_dir, label_dir, device, score_thresh=0.3):
    """Evaluate detector on images"""
    detector.eval()

    predictions = []
    ground_truths = []

    # Get image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    print(f"Evaluating on {len(image_files)} images from {image_dir}")

    skipped = 0
    for img_file in tqdm(image_files):
        # Load image
        img_path = os.path.join(image_dir, img_file)
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        except (OSError, IOError) as e:
            skipped += 1
            continue

        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

        # Get image size
        img_size = img_tensor.shape[-1]

        # Load labels
        label_file = img_file.replace('.png', '.txt').replace('_sr', '')
        label_path = os.path.join(label_dir, label_file)
        gt_boxes, gt_labels = load_yolo_labels(label_path, img_size)

        ground_truths.append({
            'boxes': gt_boxes.numpy(),
            'labels': gt_labels.numpy()
        })

        # Run detection
        with torch.no_grad():
            img_list = [img_tensor[0]]  # Remove batch dim
            outputs, _ = detector(img_list)

        # Filter by score
        output = outputs[0]
        keep = output['scores'] >= score_thresh

        predictions.append({
            'boxes': output['boxes'][keep].cpu().numpy(),
            'scores': output['scores'][keep].cpu().numpy(),
            'labels': output['labels'][keep].cpu().numpy()
        })

    if skipped > 0:
        print(f"Skipped {skipped} corrupted/truncated images")

    # Compute mAP
    ap50 = compute_ap(predictions, ground_truths, iou_threshold=0.5)
    ap25 = compute_ap(predictions, ground_truths, iou_threshold=0.25)
    ap75 = compute_ap(predictions, ground_truths, iou_threshold=0.75)

    return {
        'mAP@25': ap25,
        'mAP@50': ap50,
        'mAP@75': ap75,
        'num_images': len(image_files)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--sr_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--detector_weights', type=str,
                        default='/mnt/server16_hard1/LIGNEX1/SR4IR/runs_det_pure1ch_pretrained/best_map.pth')
    parser.add_argument('--score_thresh', type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build detector
    print("Loading detector...")
    detector = build_detector(
        use_1ch=True,
        backbone_variant='large',
        num_classes=3,
        image_mean=[0.449],
        image_std=[0.226]
    )

    # Load weights
    state_dict = torch.load(args.detector_weights, map_location='cpu')
    detector.load_state_dict(state_dict, strict=True)
    detector = detector.to(device)
    print(f"Loaded weights from {args.detector_weights}")

    # Evaluate on HR
    print("\n" + "="*60)
    print("Evaluating on HR images")
    print("="*60)
    hr_results = evaluate_detector(detector, args.hr_dir, args.label_dir, device, args.score_thresh)
    print(f"HR Results:")
    print(f"  mAP@25: {hr_results['mAP@25']:.4f}")
    print(f"  mAP@50: {hr_results['mAP@50']:.4f}")
    print(f"  mAP@75: {hr_results['mAP@75']:.4f}")

    # Evaluate on SR
    print("\n" + "="*60)
    print("Evaluating on SR images")
    print("="*60)
    sr_results = evaluate_detector(detector, args.sr_dir, args.label_dir, device, args.score_thresh)
    print(f"SR Results:")
    print(f"  mAP@25: {sr_results['mAP@25']:.4f}")
    print(f"  mAP@50: {sr_results['mAP@50']:.4f}")
    print(f"  mAP@75: {sr_results['mAP@75']:.4f}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Metric':<10} {'HR':<10} {'SR':<10} {'Gap':<10}")
    print("-"*40)
    print(f"{'mAP@25':<10} {hr_results['mAP@25']:<10.4f} {sr_results['mAP@25']:<10.4f} {hr_results['mAP@25'] - sr_results['mAP@25']:<10.4f}")
    print(f"{'mAP@50':<10} {hr_results['mAP@50']:<10.4f} {sr_results['mAP@50']:<10.4f} {hr_results['mAP@50'] - sr_results['mAP@50']:<10.4f}")
    print(f"{'mAP@75':<10} {hr_results['mAP@75']:<10.4f} {sr_results['mAP@75']:<10.4f} {hr_results['mAP@75'] - sr_results['mAP@75']:<10.4f}")


if __name__ == '__main__':
    main()
