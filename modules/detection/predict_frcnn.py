#!/usr/bin/env python3
"""Run inference with a Faster R-CNN checkpoint (torchvision)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision
import yaml
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def load_names(data_yaml: Path) -> list[str]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return [str(x) for x in names]


def build_model(num_classes: int) -> torch.nn.Module:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
    score_thresh: float,
):
    model.eval()
    im = Image.open(image_path).convert("RGB")
    x = torchvision.transforms.functional.to_tensor(im).to(device)
    out = model([x])[0]
    keep = out["scores"] >= score_thresh
    boxes = out["boxes"][keep].cpu()
    scores = out["scores"][keep].cpu()
    labels = out["labels"][keep].cpu()
    return boxes, labels, scores


def main():
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument(
        "--data-yaml",
        type=Path,
        default=repo_root / "dataset/yolo/dataset_unicode_topN/dataset.yaml",
    )
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    names = load_names(args.data_yaml)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(ckpt.get("num_classes", len(names) + 1))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(num_classes=num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    boxes, labels, scores = predict(model, device, args.image, args.score_thresh)
    # torchvision label 0 is background; expect >=1 for objects
    texts = []
    for lab in labels.tolist():
        if lab <= 0:
            texts.append("?")
        else:
            idx = lab - 1
            texts.append(names[idx] if 0 <= idx < len(names) else "?")

    print(f"image={args.image}  n={len(boxes)}")
    for i in range(len(boxes)):
        b = boxes[i].tolist()
        print(f"  {i}: score={scores[i]:.3f} box={tuple(round(x,1) for x in b)} class={labels[i].item()} glyph={texts[i]!r}")


if __name__ == "__main__":
    main()
