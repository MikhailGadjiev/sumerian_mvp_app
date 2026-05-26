import argparse
from pathlib import Path
from typing import Dict


def resolve_defaults() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "repo_root": repo_root,
        "weights": repo_root / "3_YOLO_ultra/yolo12m.pt",
        "data": repo_root / "dataset/yolo/dataset_unicode_topN/dataset.yaml",
    }


def parse_args() -> argparse.Namespace:
    defaults = resolve_defaults()
    parser = argparse.ArgumentParser(description="Train and optionally validate YOLO model for Sumerian sign detection.")

    parser.add_argument("--weights", type=Path, default=defaults["weights"])
    parser.add_argument("--data", type=Path, default=defaults["data"])
    parser.add_argument("--val-data", type=Path, default=None, help="Validation dataset yaml. Defaults to --data.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--project", type=str, default="sumerian_yolo")
    parser.add_argument("--name", type=str, default="unicode_topN_finalfinal")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or device index.")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from specific checkpoint.")

    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=7.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=7.0)
    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.0)
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.1)
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.1)
    parser.add_argument("--mosaic", type=float, default=0.8)
    parser.add_argument("--mixup", type=float, default=0.2)

    parser.add_argument("--run-val", action="store_true")
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--no-save-json", dest="save_json", action="store_false")
    parser.add_argument("--val-iou", type=float, default=0.5)
    parser.add_argument("--val-conf", type=float, default=0.1)
    parser.add_argument("--agnostic-nms", action="store_true", default=True)
    parser.add_argument("--no-agnostic-nms", dest="agnostic_nms", action="store_false")

    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    import torch

    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def preflight_report(selected_device: str) -> None:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Selected device: {selected_device}")


def build_hyp(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
    }


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    train_device = choose_device(args.device)
    preflight_report(train_device)

    if args.resume and args.resume_from:
        raise ValueError("Use either --resume or --resume-from, not both.")

    model_source = args.resume_from if args.resume_from is not None else args.weights
    model = YOLO(str(model_source))

    hyp = build_hyp(args)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "device": train_device,
        **hyp,
    }
    if args.resume:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)

    if args.run_val:
        val_data = args.val_data if args.val_data is not None else args.data
        model.val(
            data=str(val_data),
            save_json=args.save_json,
            iou=args.val_iou,
            conf=args.val_conf,
            agnostic_nms=args.agnostic_nms,
        )


if __name__ == "__main__":
    main()
