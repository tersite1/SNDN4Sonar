#!/usr/bin/env python3
"""
Detector Evaluation Script (Config-based)

Usage:
    python eval_det.py --opt options/eval/eval_det.yml
    python eval_det.py --opt options/eval/eval_det.yml --focus_only
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from archs import build_network
from data.det import YoloDetectionDataset, DetectionPresetEval, collate_fn
from utils.det.coco_eval import CocoEvaluator
from utils.det.coco_utils import get_coco_api_from_dataset


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SimpleLogger:
    """Simple logger that prints to stdout."""
    def write(self, msg):
        print(msg)


def evaluate(model, data_loader, device, coco_evaluator):
    """Run evaluation on dataset."""
    model.eval()

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs, _ = model(images)

            # Format results for COCO evaluator
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to config file')
    parser.add_argument('--focus_only', action='store_true', help='Evaluate only on focus images')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    args = parser.parse_args()

    # Parse config
    opt = load_config(args.opt)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build dataset
    data_opt = opt['data']
    img_dir = Path(data_opt['path'])
    label_dir = img_dir.parent / 'labels'

    focus_only = args.focus_only or data_opt.get('focus_only', False)

    print(f"Dataset path: {img_dir}")
    print(f"Focus only: {focus_only}")

    dataset = YoloDetectionDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transforms=DetectionPresetEval(),
        yolo_normalized=data_opt.get('yolo_normalized', True),
        merge_classes=data_opt.get('merge_classes', False),
        label_offset=data_opt.get('label_offset', 1),
        num_classes=opt['network_det'].get('num_classes', 3),
        focus_only=focus_only,
    )
    print(f"Total images: {len(dataset)}")

    # Create dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Build detector
    logger = SimpleLogger()
    net_det = build_network(opt['network_det'], logger=logger, task='det', tag='net_det')

    # Load checkpoint
    ckpt_path = opt['path'].get('network_det')
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if 'net_det' in ckpt:
                state_dict = ckpt['net_det']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'detector.backbone.body.0.0.weight' in ckpt:
                # Direct state_dict format (from models/net_det_best.pth)
                state_dict = ckpt
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        net_det.load_state_dict(state_dict, strict=True)
        print(f"Checkpoint loaded successfully ({len(state_dict)} keys)")
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}")

    net_det = net_det.to(device)
    net_det.eval()

    # Setup COCO evaluator
    coco = get_coco_api_from_dataset(dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types=['bbox'])

    # Run evaluation
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)

    evaluate(net_det, data_loader, device, coco_evaluator)

    # Print results
    stats = coco_evaluator.coco_eval['bbox'].stats
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"mAP (IoU=0.25:0.95): {stats[0]:.4f}")
    print(f"mAP@25:              {stats[1]:.4f}")
    print(f"mAP@50:              {stats[2]:.4f}")
    print(f"mAP@75:              {stats[3]:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
