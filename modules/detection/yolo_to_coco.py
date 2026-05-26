#!/usr/bin/env python3
"""Convert Ultralytics-style YOLO labels + dataset.yaml to COCO detection JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_dataset_config(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    nc = int(cfg["nc"])
    if len(names) != nc:
        raise ValueError(f"len(names)={len(names)} != nc={nc} in {yaml_path}")
    cfg["_names_list"] = names
    return cfg


def yolo_line_to_xyxy(
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[list[float], int]:
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    x1 = max(0.0, min(float(x1), float(img_w)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    y2 = max(0.0, min(float(y2), float(img_h)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2], int(class_id) + 1


def categories_from_cfg(cfg: dict) -> list[dict]:
    names = cfg["_names_list"]
    return [{"id": i + 1, "name": str(name)} for i, name in enumerate(names)]


def convert_split(data_root: Path, cfg: dict, split: str, ann_id_start: int) -> tuple[dict, int]:
    rel_img = Path(cfg[split].strip("/"))
    img_dir = data_root / rel_img
    lab_dir = data_root / "labels" / split

    images_out: list[dict] = []
    annotations: list[dict] = []
    ann_id = ann_id_start
    image_id = 1

    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    for img_path in paths:
        with Image.open(img_path) as im:
            w, h = im.size
        rel_name = f"{split}/{img_path.name}"
        images_out.append({"id": image_id, "file_name": rel_name, "width": int(w), "height": int(h)})

        label_path = lab_dir / f"{img_path.stem}.txt"
        if label_path.is_file():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    if cls < 0 or cls >= cfg["nc"]:
                        raise ValueError(f"Bad class {cls} in {label_path}")
                    (x1, y1, x2, y2), cat_id = yolo_line_to_xyxy(cls, cx, cy, bw, bh, w, h)
                    bw_pix = x2 - x1
                    bh_pix = y2 - y1
                    if bw_pix <= 1.0 or bh_pix <= 1.0:
                        continue
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [float(x1), float(y1), float(bw_pix), float(bh_pix)],
                            "area": float(bw_pix * bh_pix),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
        image_id += 1

    coco = {
        "info": {"description": "YOLO to COCO (SumerianSignDetection)", "version": "1.0"},
        "licenses": [],
        "images": images_out,
        "annotations": annotations,
        "categories": categories_from_cfg(cfg),
    }
    return coco, ann_id


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "dataset/yolo/dataset_unicode_topN",
        help="Folder containing images/, labels/, dataset.yaml",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <data-root>/annotations")
    args = p.parse_args()
    root = args.data_root.resolve()
    yaml_path = root / "dataset.yaml"
    if not yaml_path.is_file():
        raise SystemExit(f"Missing {yaml_path}")

    cfg = load_dataset_config(yaml_path)
    out_dir = (args.out_dir or (root / "annotations")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_id = 1
    for split in ("train", "val"):
        if split not in cfg:
            continue
        coco, ann_id = convert_split(root, cfg, split, ann_id_start=ann_id)
        out_path = out_dir / f"instances_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"Wrote {out_path}  images={len(coco['images'])}  ann={len(coco['annotations'])}")


if __name__ == "__main__":
    main()
