#!/usr/bin/env python3
"""
Evaluate torchvision Faster R-CNN: COCO mAP (mAP50, mAP50-95) and mean CER like YOLO/eval_CER.ipynb.

Run from repo root (recommended):
  uv run models/faster_rcnn/eval_faster_rcnn.py \\
    --checkpoint models/faster_rcnn/runs/last.pth \\
    --data-root dataset/yolo/dataset_unicode_topN \\
    --split val

CER needs full-page GT strings from whole_image_unicode (see make_yaml_dataset.ipynb).
If --gt-unicode-dir is omitted, uses <data-root>/whole_image_unicode/<split>.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from .predict_frcnn import build_model, load_names  # type: ignore[attr-defined]
except ImportError:
    from predict_frcnn import build_model, load_names  # noqa: E402


def find_gt_unicode_dir(split: str, data_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_dir() else None
    candidate = data_root / "whole_image_unicode" / split
    if candidate.is_dir():
        return candidate
    return None


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
    conf: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    im = Image.open(image_path).convert("RGB")
    x = TF.to_tensor(im).to(device)
    out = model([x])[0]
    keep = out["scores"] >= conf
    return (
        out["boxes"][keep].cpu(),
        out["labels"][keep].cpu(),
        out["scores"][keep].cpu(),
    )


def coco_eval_map(gt_json: Path, detections: list[dict]) -> tuple[float, float, list[float]]:
    """Returns (mAP50-95, mAP50, full stats list)."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        coco_gt = COCO(str(gt_json))
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = list(float(x) for x in coco_eval.stats)
    return stats[0], stats[1], stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Faster R-CNN mAP + CER (YOLO-style)")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "dataset/yolo/dataset_unicode_topN",
        help="Dataset root with dataset.yaml, images/, annotations/",
    )
    ap.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Which split (must match annotations/instances_{split}.json and images/{split}/)",
    )
    ap.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Defaults to <data-root>/dataset.yaml",
    )
    ap.add_argument("--conf", type=float, default=0.1, help="Min score (YOLO val often uses 0.1)")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--gt-unicode-dir",
        type=Path,
        default=None,
        help="Directory with {stem}.txt GT strings (one line, full unicode per image)",
    )
    ap.add_argument("--cer-alpha", type=float, default=0.35, help="reading_order alpha (eval_CER.ipynb)")
    ap.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional path to save Ultralytics-style predictions JSON for eval_CER.ipynb",
    )
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    data_yaml = (args.data_yaml or (data_root / "dataset.yaml")).resolve()
    names = load_names(data_yaml)
    split = args.split
    gt_json = data_root / "annotations" / f"instances_{split}.json"
    if not gt_json.is_file():
        raise SystemExit(
            f"Missing {gt_json}. Run faster_rcnn/yolo_to_coco.py first or fix --split."
        )

    with open(gt_json, "r", encoding="utf-8") as f:
        coco_meta = json.load(f)
    images_root = data_root / "images"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(ckpt.get("num_classes", len(names) + 1))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    detections: list[dict] = []
    ultralytics_style: list[dict] = []

    for im in sorted(coco_meta["images"], key=lambda x: x["id"]):
        rel = im["file_name"]
        path = images_root / rel
        if not path.is_file():
            continue
        boxes, labels, scores = predict_image(model, device, path, args.conf)
        fname = Path(rel).name
        stem = Path(rel).stem
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j].tolist()
            wbox = x2 - x1
            hbox = y2 - y1
            lab = int(labels[j].item())
            if lab <= 0:
                continue
            sc = float(scores[j].item())
            detections.append(
                {
                    "image_id": im["id"],
                    "category_id": lab,
                    "bbox": [float(x1), float(y1), float(wbox), float(hbox)],
                    "score": sc,
                }
            )
            ultralytics_style.append(
                {
                    "image_id": stem,
                    "file_name": fname,
                    "category_id": lab,
                    "bbox": [float(x1), float(y1), float(wbox), float(hbox)],
                    "score": sc,
                }
            )

    if not detections:
        print("[mAP] No detections above --conf; mAP is 0 by definition.")
        map5095, map50 = 0.0, 0.0
    else:
        map5095, map50, _stats = coco_eval_map(gt_json, detections)
    print(f"[mAP] COCO bbox  mAP50-95: {map5095:.4f}  mAP50: {map50:.4f}")
    print(
        "      (pycocotools COCOeval; aligned with YOLO val reports as mAP50 / mAP50-95 on the same GT)"
    )

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(ultralytics_style, f, ensure_ascii=False)
        print(f"Saved predictions JSON: {args.save_predictions}")

    yolo_dir = REPO_ROOT / "models/yolo"
    if str(yolo_dir) not in sys.path:
        sys.path.insert(0, str(yolo_dir))
    from CER import CER, reading_order  # noqa: E402

    gt_dir = find_gt_unicode_dir(split, data_root, args.gt_unicode_dir)
    if gt_dir is None:
        print(
            "[CER] Skipped: no GT directory found. "
            "Pass --gt-unicode-dir or generate whole_image_unicode via make_yaml_dataset.ipynb"
        )
        return

    file_to_preds = defaultdict(list)
    for p in ultralytics_style:
        file_to_preds[p["file_name"]].append(p)

    cers: list[float] = []
    unicode_map = {i: names[i - 1] for i in range(1, len(names) + 1)}
    for file_name, preds in file_to_preds.items():
        for pr in preds:
            pr["unicode"] = unicode_map[pr["category_id"]]
        xyxy_bboxes = []
        for bbox in preds:
            x, y, w, h = bbox["bbox"]
            xyxy_bboxes.append([x, y, x + w, y + h])
        ordered_text = reading_order(
            xyxy_bboxes,
            [bbox["unicode"].replace("<OTHER>", "?") for bbox in preds],
            alpha=args.cer_alpha,
        )
        gt_path = gt_dir / f"{file_name[:-4]}.txt"
        if not gt_path.is_file():
            continue
        gt_text = gt_path.read_text(encoding="utf-8").splitlines()[0].strip()
        if len(gt_text) < 1:
            continue
        cers.append(CER(ordered_text, gt_text))

    if cers:
        print(f"[CER] Mean CER: {sum(cers) / len(cers):.4f}  (N={len(cers)} images with GT; alpha={args.cer_alpha})")
    else:
        print("[CER] No matching GT .txt files in", gt_dir)


if __name__ == "__main__":
    main()
