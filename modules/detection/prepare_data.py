import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


ACCENTED_CHARS = [
    "Á",
    "´",
    "Û",
    "Ù",
    "Ü",
    "È",
    "É",
    "Ê",
    "Ë",
    "Ì",
    "Í",
    "Î",
    "Ï",
    "Ò",
    "Ó",
    "Ô",
    "Ö",
    "à",
    "á",
    "â",
    "ä",
    "è",
    "é",
    "ê",
    "ë",
    "ì",
    "í",
    "î",
    "ï",
    "ò",
    "ó",
    "ô",
    "ö",
    "ù",
    "ú",
    "û",
    "ü",
    "Ā",
    "ā",
    "Ē",
    "ē",
    "Ī",
    "ī",
    "Ō",
    "ō",
    "Ū",
    "ū",
]

TRANSLATION_MARKER = "#tr.en"
OBVERSE_MARKER = "@obverse"
REVERSE_MARKER = "@reverse"

SIDE_TEXT_MAP = {
    "top": "01_top",
    "left": "02_left",
    "front": "03_front",
    "right": "04_right",
    "bottom": "05_bottom",
    "back": "06_back",
}


def resolve_default_paths() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "dataset"
    return {
        "repo_root": repo_root,
        "bbox_csv": dataset_root / "MaiCuBeDa/translitmetadata.csv",
        "images_dir": dataset_root / "HeiCuBeDa/Images_MSII_Filter",
        "unicode_to_signnames": dataset_root / "MaiCuBeDa/utils/utils/unicode_to_signnames.json",
        "nuolenna_mapping": dataset_root / "MaiCuBeDa/utils/utils/nuolenna_signlist_transliteration_to_unicode_cdli_atf.json",
        "metadata_json": dataset_root / "HeiCuBeDa/HeiCuBeDa_B_Hilprecht_Database_240121.json",
        "simplified_csv": dataset_root / "MaiCuBeDa/translitmetadata_simplified.csv",
        "all_photo_json": dataset_root / "MaiCuBeDa/all_photo_anno.json",
        "train_photo_json": dataset_root / "MaiCuBeDa/train_photo_anno.json",
        "test_photo_json": dataset_root / "MaiCuBeDa/test_photo_anno.json",
        "transliteration_to_id": dataset_root / "MaiCuBeDa/transliteration_to_id.json",
        "charname_to_id": dataset_root / "MaiCuBeDa/charname_to_id.json",
        "unicode_to_id": dataset_root / "MaiCuBeDa/unicode_to_id.json",
    }


def parse_args() -> argparse.Namespace:
    defaults = resolve_default_paths()
    parser = argparse.ArgumentParser(description="Prepare Sumerian sign annotations from CSV metadata.")

    parser.add_argument("--bbox-csv", type=Path, default=defaults["bbox_csv"])
    parser.add_argument("--bbox-sep", type=str, default=";")
    parser.add_argument("--keep-default-na", action="store_true")
    parser.add_argument("--images-dir", type=Path, default=defaults["images_dir"])
    parser.add_argument("--unicode-to-signnames-json", type=Path, default=defaults["unicode_to_signnames"])
    parser.add_argument("--nuolenna-mapping-json", type=Path, default=defaults["nuolenna_mapping"])
    parser.add_argument("--metadata-json", type=Path, default=defaults["metadata_json"])

    parser.add_argument("--simplified-csv-output", type=Path, default=defaults["simplified_csv"])
    parser.add_argument("--all-photo-output", type=Path, default=defaults["all_photo_json"])
    parser.add_argument("--train-photo-output", type=Path, default=defaults["train_photo_json"])
    parser.add_argument("--test-photo-output", type=Path, default=defaults["test_photo_json"])
    parser.add_argument("--transliteration-to-id-output", type=Path, default=defaults["transliteration_to_id"])
    parser.add_argument("--charname-to-id-output", type=Path, default=defaults["charname_to_id"])
    parser.add_argument("--unicode-to-id-output", type=Path, default=defaults["unicode_to_id"])

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-split", action="store_true", help="Only write all_photo_anno and class maps.")
    parser.add_argument("--visualize-key", type=str, default=None, help="Optional side_ID key to visualize.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_charname(raw_name: str) -> str:
    charname = raw_name.split(" (")[0]
    for accented_char in ACCENTED_CHARS + [c.lower() for c in ACCENTED_CHARS]:
        charname = charname.replace(accented_char, "")
    charname = re.sub(r"\(.*?\)", "", charname).strip()
    return charname


def charnames_to_unicode_string(
    charname: str,
    charname_to_unicode: Dict[str, str],
    warn_unknown: bool = True,
) -> str:
    processed = charname.replace("__", ", ").replace("_", " ")
    processed = re.sub(r"\(.*?\)", "", processed).strip()
    charnames = [processed] if "+" not in processed else processed.strip("+").split("+")
    unicode_string = ""

    for cname in charnames:
        cleaned = cname.strip()
        if cleaned in charname_to_unicode:
            unicode_code = charname_to_unicode[cleaned]
            unicode_string += chr(int(unicode_code.replace("U+", ""), 16))
        else:
            if warn_unknown:
                print(f"Warning: charname '{cleaned}' not found in charname_to_unicode mapping.")
            unicode_string += "<OTHER>"
    return unicode_string


def search_side_transcription(text: str, side: str) -> Optional[str]:
    text = text or ""

    def extract_after_marker(src: str, marker: str) -> Optional[str]:
        marker_pos = src.find(marker)
        if marker_pos == -1:
            return None
        start = marker_pos + len(marker)
        next_at = src.find("@", start)
        if next_at == -1:
            return src[start:].strip()
        return src[start:next_at].strip()

    if side == "front":
        side_transcription = extract_after_marker(text, OBVERSE_MARKER)
        if side_transcription is None:
            side_transcription = "" if REVERSE_MARKER in text else text.strip()
    elif side == "back":
        side_transcription = extract_after_marker(text, REVERSE_MARKER)
        if side_transcription is None:
            side_transcription = ""
    else:
        raise ValueError("side must be 'front' or 'back'")

    if side_transcription is None:
        side_transcription = ""
    if "column" in side_transcription:
        return None
    return side_transcription


def transcription_to_unicode_string(transcription: str, nuolenna_mapping: Dict[str, List[str]]) -> Optional[str]:
    transcription = re.sub(r"\d+\.\s*", " ", transcription)
    transcription = re.sub(r"\{.*?\}", "", transcription)
    transcription = re.sub(r"\(.*?\)", "", transcription)
    transcription = re.sub(r"\[.*?\]", "", transcription)

    remove_chars = ["\n", "-", "rest", "[...]", "$", "broken", "?", "#", "space", "!", "blank", "_"]
    for char in remove_chars:
        transcription = transcription.replace(char, " ")

    unicode_string = ""
    for token in transcription.lower().split(" "):
        clean_token = token.strip()
        if clean_token == "":
            continue
        if clean_token not in nuolenna_mapping:
            return None
        for unicode_char in nuolenna_mapping[clean_token]:
            unicode_string += unicode_char
    return unicode_string.strip()


def get_side_transcription(
    tablet_number_id: str,
    side: str,
    total_database: Dict[str, Dict],
    nuolenna_mapping: Dict[str, List[str]],
) -> Optional[str]:
    key = f"HS {tablet_number_id}"
    if key not in total_database:
        return None
    this_tablet_transcription = total_database[key]
    if "from_cdli_archival_view" not in this_tablet_transcription:
        return None
    textual_content = this_tablet_transcription["from_cdli_archival_view"]["textual_content"]
    side_transcription = search_side_transcription(textual_content, side)
    if side_transcription is None:
        return None
    return transcription_to_unicode_string(side_transcription, nuolenna_mapping)


def unpack_bbox_str(bbox_str: str) -> List[int]:
    xmin, xmax, ymin, ymax = map(int, bbox_str.strip("[]").split(","))
    return [xmin, ymin, xmax, ymax]


def get_photo_path(images_dir: Path, tablet_id: str, side: str) -> str:
    side_text = SIDE_TEXT_MAP[side]
    return str(images_dir / f"{tablet_id}_HeiCuBeDa_GMOCF_r1.50_n4_v512_{side_text}.png")


def build_photo_level_dict(
    df: Any,
    images_dir: Path,
    charname_to_unicode: Dict[str, str],
    collection_unicodes: Dict[str, int],
    collection_transliterations: Dict[str, int],
    collection_charnames: Dict[str, int],
    total_database: Dict[str, Dict],
    nuolenna_mapping: Dict[str, List[str]],
) -> Dict[str, Dict]:
    photo_level_dict: Dict[str, Dict] = {}
    success_count = 0

    for _, row in df.iterrows():
        side_id = row["side_ID"]
        tablet_id = row["tablet_ID"]
        side = row["side"]
        bbox = row["bbox"]
        charname = row["charname"]
        transliteration = row["transliteration"]

        if "(" in charname and ")" not in charname:
            charname += ")"

        number_id = tablet_id.split("_")[-1]
        unicode_string = ""
        if side in ("front", "back"):
            fetched = get_side_transcription(number_id, side, total_database, nuolenna_mapping)
            unicode_string = fetched if fetched is not None else ""

        image_path = get_photo_path(images_dir, tablet_id, side)
        if side_id not in photo_level_dict:
            photo_level_dict[side_id] = {
                "tablet_ID": tablet_id,
                "side": side,
                "image_path": image_path,
                "bboxes": [],
                "unicode_string": unicode_string,
            }
            if unicode_string:
                success_count += 1

        unicode_str = charnames_to_unicode_string(charname, charname_to_unicode, warn_unknown=False)
        photo_level_dict[side_id]["bboxes"].append(
            {
                "bbox": unpack_bbox_str(bbox),
                "charname": charname,
                "unicode": unicode_str,
                "unicode_id": collection_unicodes[unicode_str],
                "transliteration": transliteration,
                "charname_id": collection_charnames[charname],
                "transliteration_id": collection_transliterations[transliteration],
            }
        )

    print(f"Successfully retrieved unicode strings for {success_count} out of {len(photo_level_dict)} annotations.")
    return photo_level_dict


def main() -> None:
    args = parse_args()

    import pandas as pd
    from sklearn.model_selection import train_test_split

    bbox_df = pd.read_csv(args.bbox_csv, sep=args.bbox_sep, keep_default_na=args.keep_default_na)

    bbox_df["tablet_ID"] = bbox_df["ID"].str.split("_").str[:-1].str.join("_")
    bbox_df = bbox_df.rename(columns={"ID": "side_ID"})

    ensure_parent(args.simplified_csv_output)
    bbox_df.to_csv(args.simplified_csv_output, sep=",", index=False)

    with open(args.unicode_to_signnames_json, "r", encoding="utf-8") as f:
        unicode_to_charname = json.load(f)
    charname_to_unicode = {
        normalize_charname(meta["signName"]): unicode_code for unicode_code, meta in unicode_to_charname.items()
    }

    charname_freq = bbox_df["charname"].value_counts()
    transliteration_freq = bbox_df["transliteration"].value_counts()
    collection_charname = charname_freq.index.tolist()
    collection_transliteration = transliteration_freq.index.tolist()

    collection_charnames = {name: idx for idx, name in enumerate(collection_charname)}
    collection_transliterations = {name: idx for idx, name in enumerate(collection_transliteration)}

    collection_unicodes_list: List[str] = []
    for charname in collection_charname:
        unicode_str = charnames_to_unicode_string(charname, charname_to_unicode, warn_unknown=not args.quiet)
        if unicode_str not in collection_unicodes_list:
            collection_unicodes_list.append(unicode_str)
    collection_unicodes = {unicode_value: idx for idx, unicode_value in enumerate(collection_unicodes_list)}

    with open(args.nuolenna_mapping_json, "r", encoding="utf-8") as f:
        nuolenna_mapping = json.load(f)
    with open(args.metadata_json, "r", encoding="utf-8") as f:
        total_database = json.load(f)

    photo_level_dict = build_photo_level_dict(
        df=bbox_df,
        images_dir=args.images_dir,
        charname_to_unicode=charname_to_unicode,
        collection_unicodes=collection_unicodes,
        collection_transliterations=collection_transliterations,
        collection_charnames=collection_charnames,
        total_database=total_database,
        nuolenna_mapping=nuolenna_mapping,
    )

    ensure_parent(args.all_photo_output)
    with open(args.all_photo_output, "w", encoding="utf-8") as f:
        json.dump(photo_level_dict, f, indent=4, ensure_ascii=False)

    if not args.skip_split:
        train_ids, test_ids = train_test_split(
            list(photo_level_dict.keys()),
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_dict = {k: photo_level_dict[k] for k in train_ids}
        test_dict = {k: photo_level_dict[k] for k in test_ids}

        ensure_parent(args.train_photo_output)
        with open(args.train_photo_output, "w", encoding="utf-8") as f:
            json.dump(train_dict, f, indent=4, ensure_ascii=False)
        ensure_parent(args.test_photo_output)
        with open(args.test_photo_output, "w", encoding="utf-8") as f:
            json.dump(test_dict, f, indent=4, ensure_ascii=False)

        print(f"Train photos: {len(train_ids)}, Test photos: {len(test_ids)}")

    ensure_parent(args.transliteration_to_id_output)
    with open(args.transliteration_to_id_output, "w", encoding="utf-8") as f:
        json.dump(collection_transliterations, f, indent=4, ensure_ascii=False)
    ensure_parent(args.charname_to_id_output)
    with open(args.charname_to_id_output, "w", encoding="utf-8") as f:
        json.dump(collection_charnames, f, indent=4, ensure_ascii=False)
    ensure_parent(args.unicode_to_id_output)
    with open(args.unicode_to_id_output, "w", encoding="utf-8") as f:
        json.dump(collection_unicodes, f, indent=4, ensure_ascii=False)

    if args.visualize_key:
        import sys

        sys.path.append(str(resolve_default_paths()["repo_root"] / "playgrounds"))
        from visualize_photo import visualize_photo_with_bboxes

        if args.visualize_key not in photo_level_dict:
            raise KeyError(f"Key '{args.visualize_key}' not found in annotations.")
        visualize_photo_with_bboxes(photo_level_dict[args.visualize_key])


if __name__ == "__main__":
    main()
