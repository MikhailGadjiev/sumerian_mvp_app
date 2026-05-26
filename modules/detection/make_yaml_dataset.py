import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple


LABEL_CHOICES = ("is_sign", "charname", "transliteration", "charname_topN", "unicode", "unicode_topN")


def resolve_defaults() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "dataset"
    return {
        "repo_root": repo_root,
        "train_json": dataset_root / "MaiCuBeDa/train_photo_anno.json",
        "val_json": dataset_root / "MaiCuBeDa/test_photo_anno.json",
        "all_photo_json": dataset_root / "MaiCuBeDa/all_photo_anno.json",
        "images_dir": dataset_root / "HeiCuBeDa/Images_MSII_Filter",
        "output_dir": dataset_root / "yolo/dataset",
        "charname_map": dataset_root / "MaiCuBeDa/charname_to_id.json",
        "transliteration_map": dataset_root / "MaiCuBeDa/transliteration_to_id.json",
        "unicode_map": dataset_root / "MaiCuBeDa/unicode_to_id.json",
    }


def parse_args() -> argparse.Namespace:
    defaults = resolve_defaults()
    parser = argparse.ArgumentParser(description="Generate YOLO dataset folders and dataset.yaml from photo annotations.")

    parser.add_argument("--root", type=Path, default=defaults["repo_root"])
    parser.add_argument("--train-json", type=Path, default=defaults["train_json"])
    parser.add_argument("--val-json", type=Path, default=defaults["val_json"])
    parser.add_argument("--all-photo-json", type=Path, default=defaults["all_photo_json"])
    parser.add_argument("--images-dir", type=Path, default=defaults["images_dir"])

    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--dataset-name-suffix", type=str, default=None)
    parser.add_argument("--yaml-path", type=Path, default=None)
    parser.add_argument("--unicode-string-dir", type=Path, default=None)

    parser.add_argument("--label-key", type=str, choices=LABEL_CHOICES, default="unicode_topN")
    parser.add_argument("--top-n", type=int, default=150)

    parser.add_argument("--charname-map", type=Path, default=defaults["charname_map"])
    parser.add_argument("--transliteration-map", type=Path, default=defaults["transliteration_map"])
    parser.add_argument("--unicode-map", type=Path, default=defaults["unicode_map"])
    parser.add_argument("--write-charname-topn-map", action="store_true")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clear-label-cache", action="store_true", default=True)
    parser.add_argument("--no-clear-label-cache", dest="clear_label_cache", action="store_false")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def convert_bbox(bbox: Iterable[int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


def resolve_output_dir(base_output_dir: Path, label_key: str, dataset_name_suffix: str | None) -> Path:
    if dataset_name_suffix:
        return base_output_dir.parent / f"{base_output_dir.name}_{dataset_name_suffix}"
    if label_key == "is_sign":
        return base_output_dir
    return base_output_dir.parent / f"{base_output_dir.name}_{label_key}"


def load_label_map(args: argparse.Namespace) -> Dict[str, int]:
    if args.label_key in ("charname", "charname_topN"):
        with open(args.charname_map, "r", encoding="utf-8") as f:
            dict_id = json.load(f)
        if args.label_key == "charname_topN":
            dict_id = {k: (v if v < args.top_n else args.top_n) for k, v in dict_id.items()}
            if args.write_charname_topn_map:
                topn_path = args.root / f"dataset/MaiCuBeDa/charname_to_id_top{args.top_n}.json"
                topn_path.parent.mkdir(parents=True, exist_ok=True)
                with open(topn_path, "w", encoding="utf-8") as f:
                    json.dump(dict_id, f, ensure_ascii=False, indent=4)
        return dict_id
    if args.label_key == "transliteration":
        with open(args.transliteration_map, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.label_key in ("unicode", "unicode_topN"):
        with open(args.unicode_map, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sign": 0}


def get_class_id(bbox_entry: Dict, label_key: str, top_n: int) -> int:
    if label_key == "is_sign":
        return 0
    if label_key == "charname":
        return int(bbox_entry["charname_id"])
    if label_key == "transliteration":
        return int(bbox_entry["transliteration_id"])
    if label_key == "charname_topN":
        return min(int(bbox_entry["charname_id"]), top_n)
    if label_key == "unicode":
        return int(bbox_entry["unicode_id"])
    if label_key == "unicode_topN":
        return min(int(bbox_entry["unicode_id"]), top_n)
    raise ValueError(f"Unsupported label_key: {label_key}")


def process_json(
    json_file: Path,
    images_dir: Path,
    img_output_dir: Path,
    lbl_output_dir: Path,
    unicode_output_dir: Path,
    label_key: str,
    top_n: int,
    overwrite: bool,
) -> None:
    from PIL import Image
    from tqdm import tqdm

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for _, item in tqdm(data.items(), desc=f"Processing {json_file.name}"):
        src_img = images_dir / os.path.basename(item["image_path"])
        if not src_img.exists():
            print(f"Warning: image does not exist: {src_img}")
            continue

        dst_img = img_output_dir / src_img.name
        if overwrite or not dst_img.exists():
            shutil.copy(src_img, dst_img)

        with Image.open(src_img) as img:
            w, h = img.size

        label_file = lbl_output_dir / f"{src_img.stem}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            for bbox_entry in item["bboxes"]:
                class_id = get_class_id(bbox_entry, label_key, top_n)
                x_center, y_center, width, height = convert_bbox(bbox_entry["bbox"], w, h)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        unicode_file = unicode_output_dir / f"{src_img.stem}.txt"
        with open(unicode_file, "w", encoding="utf-8") as uf:
            uf.write(item.get("unicode_string", ""))


def collect_class_ids(all_photo_json: Path, label_key: str, top_n: int) -> list[int]:
    with open(all_photo_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    class_ids = set()
    for item in data.values():
        for bbox_entry in item["bboxes"]:
            class_ids.add(get_class_id(bbox_entry, label_key, top_n))
    return sorted(class_ids)


def build_yaml_data(
    output_dir: Path,
    all_photo_json: Path,
    label_key: str,
    top_n: int,
    reverse_dict_id: Dict[int, str],
) -> Dict:
    yaml_data = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": None,
        "names": None,
    }
    if label_key == "is_sign":
        yaml_data["nc"] = 1
        yaml_data["names"] = ["sign"]
        return yaml_data

    class_ids = collect_class_ids(all_photo_json, label_key, top_n)
    yaml_data["nc"] = len(class_ids)
    yaml_data["names"] = [reverse_dict_id.get(class_id, "OTHER") for class_id in class_ids]
    return yaml_data


def clear_cache(output_dir: Path) -> None:
    for split in ("train", "val"):
        cache_file = output_dir / "labels" / f"{split}.cache"
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cleared cache file: {cache_file}")


def main() -> None:
    args = parse_args()

    import yaml

    output_dir = resolve_output_dir(args.output_dir, args.label_key, args.dataset_name_suffix)
    yaml_path = args.yaml_path if args.yaml_path is not None else output_dir / "dataset.yaml"
    unicode_base_dir = (
        args.unicode_string_dir if args.unicode_string_dir is not None else output_dir / "whole_image_unicode"
    )

    img_train_dir = output_dir / "images/train"
    img_val_dir = output_dir / "images/val"
    lbl_train_dir = output_dir / "labels/train"
    lbl_val_dir = output_dir / "labels/val"
    unicode_train_dir = unicode_base_dir / "train"
    unicode_val_dir = unicode_base_dir / "val"

    for path in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir, unicode_train_dir, unicode_val_dir):
        ensure_dir(path)

    label_map = load_label_map(args)
    reverse_dict_id = {int(v): k for k, v in label_map.items()}
    if args.label_key in ("charname_topN", "unicode_topN"):
        reverse_dict_id[args.top_n] = reverse_dict_id.get(args.top_n, "OTHER")

    print("Processing training data...")
    process_json(
        json_file=args.train_json,
        images_dir=args.images_dir,
        img_output_dir=img_train_dir,
        lbl_output_dir=lbl_train_dir,
        unicode_output_dir=unicode_train_dir,
        label_key=args.label_key,
        top_n=args.top_n,
        overwrite=args.overwrite,
    )
    print("Processing validation data...")
    process_json(
        json_file=args.val_json,
        images_dir=args.images_dir,
        img_output_dir=img_val_dir,
        lbl_output_dir=lbl_val_dir,
        unicode_output_dir=unicode_val_dir,
        label_key=args.label_key,
        top_n=args.top_n,
        overwrite=args.overwrite,
    )

    yaml_data = build_yaml_data(
        output_dir=output_dir,
        all_photo_json=args.all_photo_json,
        label_key=args.label_key,
        top_n=args.top_n,
        reverse_dict_id=reverse_dict_id,
    )
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True)
    print(f"YOLO dataset.yaml saved to {yaml_path}")

    if args.clear_label_cache:
        clear_cache(output_dir)
    print("Done! YOLO dataset ready.")


if __name__ == "__main__":
    main()
