"""
Train det_low (Faster R-CNN + MobileNetV3) on small patch data with pixel-coordinate boxes.
Supports txt/json labels.
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

# Ensure local imports work when running as script
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


def load_json_labels(root: Path, stem: str) -> List[Dict]:
    json_path = root if root.suffix.lower() == ".json" else root / "labels.json"
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text())
    images = data.get("images", {})
    return images.get(stem, [])


def load_xml_labels(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        items = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="0")
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            x1 = float(bbox.findtext("xmin", default="0"))
            y1 = float(bbox.findtext("ymin", default="0"))
            x2 = float(bbox.findtext("xmax", default="0"))
            y2 = float(bbox.findtext("ymax", default="0"))
            items.append({"class_id": int(name), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return items
    except Exception:
        return []


class PatchDetDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[Path, Path, str]],
        enforce_640: bool = True,
        yolo_normalized: bool = False,
        num_classes: int = 2,
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
            im = im.convert("RGB")
            W, H = im.size
            if self.enforce_640 and (W != 640 or H != 640):
                raise RuntimeError(f"Image {img_path} size is {W}x{H}, expected 640x640")
            img = torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 255.0

        stem = img_path.stem
        fmt = label_format.lower()
        if fmt == "txt":
            lbl_path = label_dir / f"{stem}.txt"
            labels = load_txt_labels(lbl_path)
        elif fmt == "xml":
            lbl_path = label_dir / f"{stem}.xml"
            labels = load_xml_labels(lbl_path)
        else:
            labels = load_json_labels(label_dir, stem)

        boxes: List[List[float]] = []
        classes: List[int] = []
        for obj in labels:
            x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
            if self.yolo_normalized:
                # labels are cx,cy,w,h normalized in 0~1
                cx, cy, w, h = x1, y1, x2, y2
                x1 = (cx - w / 2.0) * W
                y1 = (cy - h / 2.0) * H
                x2 = (cx + w / 2.0) * W
                y2 = (cy + h / 2.0) * H
            # clamp to image bounds
            x1 = float(np.clip(x1, 0, W - 1))
            y1 = float(np.clip(y1, 0, H - 1))
            x2 = float(np.clip(x2, 0, W - 1))
            y2 = float(np.clip(y2, 0, H - 1))
            if not np.isfinite([x1, y1, x2, y2]).all():
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            cls = 1  # merge all mine types into a single foreground class
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train det_low (FRCNN+MBV3) on patch data")
    parser.add_argument("--img_dir", type=Path, default=None, help="legacy single dir of images")
    parser.add_argument("--label_dir", type=Path, default=None, help="legacy single dir of labels")
    parser.add_argument("--label_format", type=str, default="txt", choices=["txt", "json", "xml"])
    parser.add_argument("--val_img_dir", type=Path, default=None)
    parser.add_argument("--val_label_dir", type=Path, default=None)
    parser.add_argument("--root_dir", type=Path, default=None, help="root with sss/sas/{split}/{raw,meta}")
    parser.add_argument("--datasets", nargs="+", default=["sss", "sas"])
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--val_splits", nargs="+", default=["validate"])
    parser.add_argument("--yolo_normalized", action="store_true", help="Set if labels are YOLO normalized cx cy w h")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_parallel", action="store_true", help="Use DataParallel over all visible GPUs")
    parser.add_argument("--save_dir", type=Path, default=Path("runs_det_low"))
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    def gather_items(splits, label_fmt):
        items = []
        if args.root_dir:
            for ds in args.datasets:
                for sp in splits:
                    raw_dir = args.root_dir / ds / sp / "raw"
                    meta_dir = args.root_dir / ds / sp / "meta"
                    if not raw_dir.exists():
                        continue
                    imgs = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])
                    for img_path in imgs:
                        items.append((img_path, meta_dir, "xml"))
        else:
            if args.img_dir is None or args.label_dir is None:
                raise SystemExit("Either --root_dir or (--img_dir and --label_dir) must be provided.")
            imgs = sorted([p for p in args.img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])
            for img_path in imgs:
                items.append((img_path, args.label_dir, label_fmt))
        return items

    train_items = gather_items(args.splits, args.label_format)
    val_items = gather_items(args.val_splits, args.label_format) if args.val_splits else []

    ds = PatchDetDataset(train_items, enforce_640=True, yolo_normalized=args.yolo_normalized, num_classes=2)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    val_loader: Optional[DataLoader] = None
    if val_items:
        val_ds = PatchDetDataset(val_items, enforce_640=True, yolo_normalized=args.yolo_normalized, num_classes=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    num_classes = 2  # background + mine
    model = SonarFasterRCNNDetector(
        num_classes=num_classes,
        min_size=640,
        max_size=640,
        anchor_sizes=None,
        aspect_ratios=None,
        box_score_thresh=0.001,
        box_detections_per_img=1000,
        rpn_pre_nms_top_n_test=5000,
        rpn_post_nms_top_n_test=5000,
    ).to(device)
    if args.data_parallel and use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        iterator = loader
        if tqdm:
            iterator = tqdm(loader, desc=f"Train {epoch}/{args.epochs}", ncols=120)
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
            if tqdm:
                iterator.set_postfix(loss=loss.item())
        epoch_loss = running / max(1, len(loader))
        val_loss = None
        if val_loader:
            model.eval()
            vtotal = 0.0
            v_iter = val_loader
            if tqdm:
                v_iter = tqdm(val_loader, desc="Val", ncols=120)
            with torch.no_grad():
                for vimgs, vtargets in v_iter:
                    vimgs = [im.to(device) for im in vimgs]
                    vtargets = [{k: v.to(device) for k, v in t.items()} for t in vtargets]
                    _, vlosses = model(vimgs, vtargets)
                    vtotal += (sum(vlosses.values()) if vlosses else torch.tensor(0.0, device=device)).item()
            val_loss = vtotal / max(1, len(val_loader))
        print(f"[epoch {epoch}] train_loss={epoch_loss:.4f}" + (f" val_loss={val_loss:.4f}" if val_loss is not None else ""))

        torch.save(model.state_dict(), args.save_dir / f"model_epoch{epoch:03d}.pth")
        metric = val_loss if val_loss is not None else epoch_loss
        if metric < best_loss:
            best_loss = metric
            torch.save(model.state_dict(), args.save_dir / "best.pth")
            print(f"[epoch {epoch}] best.pth updated (metric={metric:.4f})")
    print("Done.")


if __name__ == "__main__":
    main()
