"""
Train 1-channel grayscale detector (Faster R-CNN + MobileNetV3)
Modified from train_det_low.py for grayscale input
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Ensure local imports work
FILE_DIR = Path(__file__).resolve().parent
if str(FILE_DIR) not in sys.path:
    sys.path.append(str(FILE_DIR))

from fasterrcnn_mbv3 import SonarFasterRCNNDetector


def load_txt_labels(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, x1, y1, x2, y2 = map(float, line.split()[:5])
        items.append({"class_id": int(cls), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return items


class GrayscaleDetDataset(Dataset):
    """Dataset for 1-channel grayscale detection"""
    def __init__(
        self,
        items: List[Tuple[Path, Path, str]],
        enforce_640: bool = True,
        yolo_normalized: bool = False,
        num_classes: int = 3,
    ) -> None:
        self.items = items
        self.enforce_640 = enforce_640
        self.yolo_normalized = yolo_normalized
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label_dir, label_format = self.items[idx]
        with Image.open(img_path) as im:
            im = im.convert("L")  # ★ Grayscale
            W, H = im.size
            if self.enforce_640 and (W != 640 or H != 640):
                raise RuntimeError(f"Image {img_path} size is {W}x{H}, expected 640x640")
            # [H, W] -> [1, H, W] (keep 1-channel)
            img = torch.from_numpy(np.array(im, dtype=np.float32)).unsqueeze(0) / 255.0

        stem = img_path.stem
        fmt = label_format.lower()
        lbl_path = label_dir / f"{stem}.txt"
        labels = load_txt_labels(lbl_path)

        boxes: List[List[float]] = []
        classes: List[int] = []
        for obj in labels:
            x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
            if self.yolo_normalized:
                cx, cy, w, h = x1, y1, x2, y2
                x1 = (cx - w / 2.0) * W
                y1 = (cy - h / 2.0) * H
                x2 = (cx + w / 2.0) * W
                y2 = (cy + h / 2.0) * H
            x1 = float(np.clip(x1, 0, W - 1))
            y1 = float(np.clip(y1, 0, H - 1))
            x2 = float(np.clip(x2, 0, W - 1))
            y2 = float(np.clip(y2, 0, H - 1))
            if not np.isfinite([x1, y1, x2, y2]).all():
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            cls = obj["class_id"] + 1  # Class 1: cylinder, Class 2: manta (0=background)
            if cls <= 0 or cls >= self.num_classes:
                continue
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(classes, dtype=torch.int64) if boxes else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_metrics(model, loader, device, num_classes=3, iou_thresh=0.5, conf_thresh=0.05):
    """Compute mAP and Recall on validation set"""
    model.eval()

    # Store predictions and ground truths per class
    all_preds = defaultdict(list)  # {class_id: [(conf, is_correct), ...]}
    all_gts = defaultdict(int)  # {class_id: count}
    all_matched_gts = 0  # Total matched GT boxes
    total_gts = 0  # Total GT boxes

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            detections, _ = model(imgs)

            for det, target in zip(detections, targets):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                pred_boxes = det["boxes"].cpu().numpy()
                pred_scores = det["scores"].cpu().numpy()
                pred_labels = det["labels"].cpu().numpy()

                # Filter by confidence
                mask = pred_scores >= conf_thresh
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]

                # Count GTs per class
                for cls in gt_labels:
                    all_gts[int(cls)] += 1
                total_gts += len(gt_labels)

                # Match predictions to GTs
                matched_gts = set()
                for pred_box, pred_score, pred_cls in zip(pred_boxes, pred_scores, pred_labels):
                    pred_cls = int(pred_cls)
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_idx in matched_gts:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    is_correct = (best_iou >= iou_thresh and
                                  best_gt_idx >= 0 and
                                  pred_cls == int(gt_labels[best_gt_idx]))

                    if is_correct:
                        matched_gts.add(best_gt_idx)

                    all_preds[pred_cls].append((pred_score, is_correct))

                # Count total matched GTs for recall calculation
                all_matched_gts += len(matched_gts)

    # Compute AP per class
    aps = {}
    class_names = {1: "cylinder", 2: "manta"}

    for cls in range(1, num_classes):  # Skip background (class 0)
        if cls not in all_preds or all_gts[cls] == 0:
            continue

        preds = all_preds[cls]
        preds.sort(key=lambda x: x[0], reverse=True)  # Sort by confidence

        tp = 0
        fp = 0
        precisions = []
        recalls = []

        for conf, is_correct in preds:
            if is_correct:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / all_gts[cls] if all_gts[cls] > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            precs = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(precs) / 11.0 if precs else 0

        aps[cls] = ap

    mAP = np.mean(list(aps.values())) if aps else 0.0
    recall = all_matched_gts / total_gts if total_gts > 0 else 0.0

    model.train()
    return mAP, recall, aps


def parse_args():
    import sys
    # Filter out --local-rank and --local_rank arguments
    filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith('--local-rank') and not arg.startswith('--local_rank')]

    parser = argparse.ArgumentParser(description="Train 1-channel grayscale detector")
    parser.add_argument("--img_dir", type=Path, required=True, help="images directory")
    parser.add_argument("--label_dir", type=Path, required=True, help="labels directory")
    parser.add_argument("--yolo_normalized", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=Path, default=Path("runs_det_gray1ch"))
    return parser.parse_args(filtered_args)


def main():
    args = parse_args()

    # DDP setup - read from environment variable
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
        device = torch.device(args.device if use_cuda else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0

    # Gather images
    imgs = sorted([p for p in args.img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    train_items = [(img_path, args.label_dir, "txt") for img_path in imgs]

    ds = GrayscaleDetDataset(train_items, enforce_640=True, yolo_normalized=args.yolo_normalized, num_classes=3)

    if is_distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, collate_fn=collate_fn)
    else:
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    num_classes = 3  # Class 0: background, Class 1: cylinder, Class 2: manta
    model = SonarFasterRCNNDetector(
        num_classes=num_classes,
        min_size=640,
        max_size=640,
        box_score_thresh=0.001,
        box_detections_per_img=1000,
        rpn_pre_nms_top_n_test=5000,
        rpn_post_nms_top_n_test=5000,
    ).to(device)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = args.save_dir / "metrics.txt"
        with open(metrics_file, "w") as f:
            f.write("epoch,train_loss,mAP,AP_cylinder,AP_manta,recall\n")
    best_loss = float("inf")
    best_map = 0.0
    best_recall = 0.0
    prev_recall = 0.0

    for epoch in range(1, args.epochs + 1):
        if is_distributed:
            sampler.set_epoch(epoch)

        model.train()
        running = 0.0
        iterator = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=160) if (tqdm and rank == 0) else loader

        for imgs, targets in iterator:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            _, losses = model(imgs, targets)
            loss = sum(losses.values()) if losses else torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running += loss.item()
            if tqdm and rank == 0:
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recall=f"{prev_recall:.4f}",
                    best_recall=f"{best_recall:.4f}"
                )

        epoch_loss = running / max(1, len(loader))

        # Synchronize loss across GPUs
        if is_distributed:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor.item() / world_size

        # Compute validation metrics (only on rank 0 to avoid duplication)
        if rank == 0:
            print(f"\n[Epoch {epoch}] Computing metrics (IoU≥0.5, conf≥0.05)...")
            mAP, recall, aps = compute_metrics(
                model.module if is_distributed else model,
                loader,
                device,
                num_classes=num_classes,
                iou_thresh=0.5,
                conf_thresh=0.05,
            )

            # Print class-wise AP
            class_names = {1: "cylinder", 2: "manta"}
            ap_str = " | ".join([f"AP_{class_names[cls]}={ap:.4f}" for cls, ap in sorted(aps.items())])
            print(f"[Epoch {epoch}] loss={epoch_loss:.4f} | mAP={mAP:.4f} | {ap_str} | recall={recall:.4f}")

            # Update prev_recall for next epoch's tqdm display
            prev_recall = recall

            # Save metrics
            ap_cyl = aps.get(1, 0.0)
            ap_manta = aps.get(2, 0.0)
            with open(metrics_file, "a") as f:
                f.write(f"{epoch},{epoch_loss:.6f},{mAP:.6f},{ap_cyl:.6f},{ap_manta:.6f},{recall:.6f}\n")

            # Save model
            state_dict = model.module.state_dict() if is_distributed else model.state_dict()
            torch.save(state_dict, args.save_dir / f"model_epoch{epoch:03d}.pth")

            # Save best by loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(state_dict, args.save_dir / "best_loss.pth")
                print(f"[Epoch {epoch}] best_loss.pth updated (loss={epoch_loss:.4f})")

            # Save best by mAP
            if mAP > best_map:
                best_map = mAP
                torch.save(state_dict, args.save_dir / "best_map.pth")
                print(f"[Epoch {epoch}] best_map.pth updated (mAP={mAP:.4f})")

            # Save best by recall
            if recall > best_recall:
                best_recall = recall
                torch.save(state_dict, args.save_dir / "best_recall.pth")
                print(f"[Epoch {epoch}] best_recall.pth updated (recall={recall:.4f})")

        # Synchronize prev_recall and best_recall across GPUs for tqdm display
        if is_distributed:
            recall_tensor = torch.tensor([prev_recall, best_recall], device=device)
            dist.broadcast(recall_tensor, src=0)
            prev_recall, best_recall = recall_tensor.tolist()

    if rank == 0:
        print("Done.")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
