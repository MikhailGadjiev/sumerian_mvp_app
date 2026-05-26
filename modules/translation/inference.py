import argparse
from pathlib import Path
from typing import Any

import torch

from compute_metrics import load_config


def translate_transliteration(
    model: Any,
    tokenizer: Any,
    transliteration: str,
    source_prefix: str,
    max_source_length: int,
    generation_config: dict[str, Any],
) -> str:
    device = next(model.parameters()).device
    model.eval()

    encoded = tokenizer(
        [source_prefix + transliteration.strip()],
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        generated = model.generate(**encoded, **generation_config)

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def load_model_and_tokenizer(model_path: Path) -> tuple[Any, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model output directory does not exist: {model_path}. "
            "Run train.py first or set model.output_dir to a saved local model."
        )

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate Sumerian transliteration to Russian.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML training/inference config.",
    )
    parser.add_argument(
        "--transliteration",
        required=True,
        help="Input Sumerian transliteration to translate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model, tokenizer = load_model_and_tokenizer(Path(config["model"]["output_dir"]))
    translation = translate_transliteration(
        model=model,
        tokenizer=tokenizer,
        transliteration=args.transliteration,
        source_prefix=str(config["data"]["source_prefix"]),
        max_source_length=int(config["model"]["max_source_length"]),
        generation_config=dict(config["generation"]),
    )

    print(translation)


if __name__ == "__main__":
    main()
