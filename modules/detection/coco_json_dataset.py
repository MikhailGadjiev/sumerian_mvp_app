"""COCO-style detection dataset from JSON (no pycocotools)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoJsonDetection(Dataset):
    """Returns PIL image and target with tensor boxes (xyxy, pixel) and labels (1..num_fg)."""

    def __init__(self, images_root: Path, ann_file: Path, transforms=None):
        self.images_root = Path(images_root)
        self.ann_file = Path(ann_file)
        self.transforms = transforms
        with open(self.ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.id_to_anns: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            self.id_to_anns[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        img_path = self.images_root / info["file_name"]
        image = Image.open(img_path).convert("RGB")
        w, h = int(info["width"]), int(info["height"])
        anns = self.id_to_anns.get(info["id"], [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1 = float(x)
            y1 = float(y)
            x2 = x1 + float(bw)
            y2 = y1 + float(bh)
            x1 = max(0.0, min(x1, float(w)))
            x2 = max(0.0, min(x2, float(w)))
            y1 = max(0.0, min(y1, float(h)))
            y2 = max(0.0, min(y2, float(h)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
