#!/usr/bin/env python3
"""Train torchvision Faster R-CNN on COCO JSON produced by yolo_to_coco.py."""

from __future__ import annotations

import argparse
import sys

import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from coco_json_dataset import CocoJsonDetection
from transforms import Compose, PILToTensor, RandomHorizontalFlip


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int, pretrained: bool) -> torch.nn.Module:
    kwargs: dict = {}
    if pretrained:
        kwargs["weights"] = "DEFAULT"
        kwargs["weights_backbone"] = "DEFAULT"
        kwargs["trainable_backbone_layers"] = 5
    else:
        kwargs["weights"] = None
        kwargs["weights_backbone"] = None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train: bool, flip_prob: float):
    ts = []
    if train and flip_prob > 0:
        ts.append(RandomHorizontalFlip(flip_prob))
    ts.append(PILToTensor())
    return Compose(ts)


@torch.no_grad()
def eval_forward(model, loader, device, max_batches: int | None = None):
    model.train()
    n = 0
    total = 0.0
    for i, (images, targets) in enumerate(loader):
        images = [im.to(device) for im in images]
        clean = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
        loss_dict = model(images, clean)
        total += float(sum(v.detach() for v in loss_dict.values()))
        n += 1
        if max_batches is not None and i + 1 >= max_batches:
            break
    return total / max(n, 1)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "dataset/yolo/dataset_unicode_topN",
    )
    ap.add_argument("--train-json", type=Path, default=None)
    ap.add_argument("--val-json", type=Path, default=None)
    ap.add_argument("--images-root", type=Path, default=None, help="Default: <data-root>/images")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--flip-prob", type=float, default=0.0)
    ap.add_argument("--pretrained", action="store_true", help="COCO weights + backbone (then new cls head)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--val-batches", type=int, default=50, help="Max val batches per evaluation (loss proxy)")
    ap.add_argument("--max-train-batches", type=int, default=None, help="Debug: stop train epoch after N batches")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    images_root = (args.images_root or (data_root / "images")).resolve()
    train_json = (args.train_json or (data_root / "annotations" / "instances_train.json")).resolve()
    val_json = (args.val_json or (data_root / "annotations" / "instances_val.json")).resolve()

    if not train_json.is_file():
        raise SystemExit(f"Missing {train_json}. Run faster_rcnn/yolo_to_coco.py first.")
    import json as _json

    with open(train_json, "r", encoding="utf-8") as f:
        meta = _json.load(f)
    n_fg = len(meta["categories"])
    num_classes = n_fg + 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds_tr = CocoJsonDetection(
        images_root,
        train_json,
        transforms=get_transform(True, args.flip_prob),
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    dl_va = None
    if val_json.is_file():
        ds_va = CocoJsonDetection(
            images_root,
            val_json,
            transforms=get_transform(False, 0.0),
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )

    model = build_model(num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.epochs > 1:
        milestones = sorted({max(1, args.epochs * 2 // 3), max(1, args.epochs * 8 // 9)})
        if milestones[0] == milestones[-1]:
            milestones = sorted({milestones[0], max(1, args.epochs - 1)})
    else:
        milestones = []
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  num_fg={n_fg} num_classes={num_classes}  train={len(ds_tr)}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for images, targets in dl_tr:
            images = [im.to(device) for im in images]
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            opt.zero_grad(set_to_none=True)
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

            running += float(losses.detach())
            n_batches += 1

            if args.max_train_batches is not None and n_batches >= args.max_train_batches:
                break

        scheduler.step()
        train_loss = running / max(n_batches, 1)
        msg = f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}"

        if dl_va is not None and args.val_every > 0 and epoch % args.val_every == 0:
            model.eval()
            vloss = eval_forward(model, dl_va, device, max_batches=args.val_batches)
            msg += f"  val_loss~={vloss:.4f}"
            model.train()

        print(msg)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "num_classes": num_classes,
            "n_fg": n_fg,
        }
        torch.save(ckpt, out_dir / "last.pth")

    dt = time.time() - t0
    print(f"done in {dt:.1f}s  checkpoint: {out_dir / 'last.pth'}")


if __name__ == "__main__":
    main()
