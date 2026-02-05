from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import VOCDetection

from utils.det import (
    DetectionPresetTrain,
    DetectionPresetEval,
    get_coco,
    create_aspect_ratio_groups,
    GroupedBatchSampler,
    collate_fn,
)

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(img_dir):
    if img_dir is None:
        return []
    return sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS]
    )


def _resolve_custom_paths(data_opt):
    def _to_path(value):
        return Path(value) if value is not None else None

    train_img_dir = _to_path(data_opt.get("train_img_dir") or data_opt.get("train_images"))
    train_label_dir = _to_path(data_opt.get("train_label_dir") or data_opt.get("train_labels"))
    val_img_dir = _to_path(data_opt.get("val_img_dir") or data_opt.get("val_images"))
    val_label_dir = _to_path(data_opt.get("val_label_dir") or data_opt.get("val_labels"))

    if train_img_dir and train_label_dir:
        return train_img_dir, train_label_dir, val_img_dir, val_label_dir

    base = _to_path(data_opt.get("path"))
    if base is None:
        return None, None, None, None

    if base.name == "images":
        train_img_dir = base
        train_label_dir = base.parent / "labels"
        root = base.parents[1] if len(base.parents) > 1 else base.parent
    elif (base / "images").exists():
        train_img_dir = base / "images"
        train_label_dir = base / "labels"
        root = base.parent
    else:
        train_img_dir = base / "train" / "images"
        train_label_dir = base / "train" / "labels"
        root = base

    val_img_dir = root / "val" / "images"
    val_label_dir = root / "val" / "labels"
    return train_img_dir, train_label_dir, val_img_dir, val_label_dir


class YoloDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        transforms=None,
        yolo_normalized=True,
        merge_classes=True,
        label_offset=1,
        num_classes=2,
        label_ext=".txt",
    ):
        self.img_dir = Path(img_dir) if img_dir is not None else None
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.transforms = transforms
        self.yolo_normalized = yolo_normalized
        self.merge_classes = merge_classes
        self.label_offset = label_offset
        self.num_classes = num_classes
        self.label_ext = label_ext

        self.img_paths = _list_images(self.img_dir) if self.img_dir else []
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def get_height_and_width(self, idx):
        img_path = self.img_paths[idx]
        with Image.open(img_path) as img:
            width, height = img.size
        return height, width

    def _load_labels(self, label_path, width, height):
        boxes = []
        labels = []
        if not label_path.exists():
            return boxes, labels

        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                coords = [float(v) for v in parts[1:5]]
            except ValueError:
                continue

            if self.yolo_normalized:
                cx, cy, w, h = coords
                x1 = (cx - w / 2.0) * width
                y1 = (cy - h / 2.0) * height
                x2 = (cx + w / 2.0) * width
                y2 = (cy + h / 2.0) * height
            else:
                x1, y1, x2, y2 = coords

            x1 = max(min(x1, width - 1), 0.0)
            y1 = max(min(y1, height - 1), 0.0)
            x2 = max(min(x2, width - 1), 0.0)
            y2 = max(min(y2, height - 1), 0.0)
            if x2 <= x1 or y2 <= y1:
                continue

            if self.merge_classes:
                label = self.label_offset
            else:
                label = cls + self.label_offset

            if self.num_classes is not None and (label <= 0 or label >= self.num_classes):
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(label)

        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("L")  # ★ Grayscale for SonarSR
            width, height = img.size

            label_dir = self.label_dir if self.label_dir is not None else self.img_dir
            label_path = label_dir / f"{img_path.stem}{self.label_ext}"
            boxes, labels = self._load_labels(label_path, width, height)

            if boxes:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int64)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)

            target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "image_id": idx,
            }

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            if target is not None:
                boxes_out = target["boxes"]
                area = (boxes_out[:, 2] - boxes_out[:, 0]) * (boxes_out[:, 3] - boxes_out[:, 1])
                target["area"] = area
                target["iscrowd"] = torch.zeros((boxes_out.shape[0],), dtype=torch.int64)

        return img, target


def load_det_data(opt):
    use_trainset = opt.get('train', False)
    data_format = opt['data']['format']
    is_voc = opt['data'].get('is_voc', False)
    
    # transform
    transform_train = DetectionPresetTrain(crop_size=opt['data'].get('crop_size', 0))
    transform_test = DetectionPresetEval()
    
    # datasets
    dataset_train = None
    if use_trainset:
        if data_format == 'coco':
            dataset_train = get_coco(root=opt['data']['path'], image_set='train', transforms=transform_train, mode="instances", is_voc=is_voc)
        elif data_format == 'voc':
            dataset_train =  VOCDetection(root=opt['data']['path'], year='2012', image_set='train', transforms=transform_train)
        elif data_format == 'custom':
            train_img_dir, train_label_dir, val_img_dir, val_label_dir = _resolve_custom_paths(opt['data'])
            yolo_normalized = opt['data'].get('yolo_normalized', True)
            merge_classes = opt['data'].get('merge_classes', True)
            label_offset = opt['data'].get('label_offset', 1)
            num_classes = opt['data'].get('num_classes', opt.get('network_det', {}).get('num_classes', None))
            label_ext = opt['data'].get('label_ext', '.txt')
            dataset_train = YoloDetectionDataset(
                train_img_dir,
                train_label_dir,
                transforms=transform_train,
                yolo_normalized=yolo_normalized,
                merge_classes=merge_classes,
                label_offset=label_offset,
                num_classes=num_classes,
                label_ext=label_ext,
            )
            
    if data_format == 'coco':
        dataset_test = get_coco(root=opt['data']['path'], image_set='val', transforms=transform_test, mode="instances", is_voc=is_voc)
    elif data_format == 'voc':
        dataset_test =  VOCDetection(root=opt['data']['path'], year='2012', image_set='val', transforms=transform_test)
    elif data_format == 'custom':
        train_img_dir, train_label_dir, val_img_dir, val_label_dir = _resolve_custom_paths(opt['data'])
        if val_img_dir is None or not val_img_dir.exists():
            val_img_dir = train_img_dir
            val_label_dir = train_label_dir
            print(f"Warning: val split not found. Using train split for evaluation: {val_img_dir}")
        yolo_normalized = opt['data'].get('yolo_normalized', True)
        merge_classes = opt['data'].get('merge_classes', True)
        label_offset = opt['data'].get('label_offset', 1)
        num_classes = opt['data'].get('num_classes', opt.get('network_det', {}).get('num_classes', None))
        label_ext = opt['data'].get('label_ext', '.txt')
        dataset_test = YoloDetectionDataset(
            val_img_dir,
            val_label_dir,
            transforms=transform_test,
            yolo_normalized=yolo_normalized,
            merge_classes=merge_classes,
            label_offset=label_offset,
            num_classes=num_classes,
            label_ext=label_ext,
        )

    # distributed training
    train_sampler = None
    if opt['dist']:
        if use_trainset:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if use_trainset:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    if use_trainset:
        # aspect ratio batch sampler
        aspect_ratio_group_factor = opt['data'].get('aspect_ratio_group_factor', 3)
        if aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset_train, k=aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, opt['train']['batch_size'])
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, opt['train']['batch_size'], drop_last=True)  
    
    # data loader    
    data_loader_train = None
    if use_trainset:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=train_batch_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=collate_fn)

    data_loader_test = None
    if opt.get('test', False):
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=collate_fn)


    return data_loader_train, data_loader_test, train_sampler, test_sampler
