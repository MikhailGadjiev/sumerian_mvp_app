import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MetricLoader = Callable[[str], Any]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    for section in ("data", "model", "generation", "metrics"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config


def default_metric_loader(name: str) -> Any:
    import evaluate

    return evaluate.load(name)


def get_metric_score(result: dict[str, Any], metric_name: str) -> float:
    for key in ("score", metric_name):
        if key in result:
            return float(result[key])

    available_keys = ", ".join(sorted(result))
    raise KeyError(f"Metric result for {metric_name!r} has no score key. Available keys: {available_keys}")


def compute_text_metrics(
    predictions: list[str],
    references: list[str],
    metric_loader: MetricLoader = default_metric_loader,
) -> dict[str, float]:
    bleu = metric_loader("bleu")
    chrf = metric_loader("chrf")

    bleu_result = bleu.compute(
        predictions=predictions,
        references=[[reference] for reference in references],
    )
    chrf_result = chrf.compute(predictions=predictions, references=references)

    return {
        "bleu": get_metric_score(bleu_result, "bleu"),
        "chrf": get_metric_score(chrf_result, "chrf"),
    }


def build_compute_metrics(
    tokenizer: Any,
    metric_loader: MetricLoader = default_metric_loader,
) -> Callable[[tuple[Any, Any]], dict[str, float]]:
    def sanitize_token_ids(token_ids: np.ndarray) -> np.ndarray:
        sanitized = np.asarray(token_ids)
        sanitized = np.where(sanitized != -100, sanitized, tokenizer.pad_token_id)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is not None:
            sanitized = np.where(sanitized < vocab_size, sanitized, tokenizer.pad_token_id)
        sanitized = np.where(sanitized >= 0, sanitized, tokenizer.pad_token_id)
        return sanitized.astype(np.int64)

    def compute_metrics(eval_preds: tuple[Any, Any]) -> dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.asarray(preds)
        labels = np.asarray(labels)

        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        preds = sanitize_token_ids(preds)
        labels = sanitize_token_ids(labels)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [prediction.strip() for prediction in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        return compute_text_metrics(
            predictions=decoded_preds,
            references=decoded_labels,
            metric_loader=metric_loader,
        )

    return compute_metrics


def build_inputs(
    values: list[str],
    source_prefix: str,
) -> list[str]:
    return [source_prefix + str(value).strip() for value in values]


def generate_predictions(
    model: Any,
    tokenizer: Any,
    sources: list[str],
    source_prefix: str,
    max_source_length: int,
    generation_config: dict[str, Any],
    batch_size: int,
) -> list[str]:
    device = next(model.parameters()).device
    predictions: list[str] = []

    for start in range(0, len(sources), batch_size):
        batch_sources = sources[start : start + batch_size]
        inputs = build_inputs(batch_sources, source_prefix)
        encoded = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            generated = model.generate(**encoded, **generation_config)

        predictions.extend(
            item.strip()
            for item in tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

    return predictions


def evaluate_test_dataset(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    model_config = config["model"]
    generation_config = dict(config["generation"])
    metrics_config = config["metrics"]

    test_path = Path(data_config["test_path"])
    source_column = data_config["source_column"]
    target_column = data_config["target_column"]

    frame = pd.read_csv(test_path)
    missing_columns = {source_column, target_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {test_path}: {sorted(missing_columns)}")

    model_path = Path(model_config["output_dir"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model output directory does not exist: {model_path}. "
            "Run train.py first or set model.output_dir to a saved local model."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = int(metrics_config.get("batch_size", generation_config.pop("batch_size", 4)))
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        sources=frame[source_column].astype(str).tolist(),
        source_prefix=str(data_config["source_prefix"]),
        max_source_length=int(model_config["max_source_length"]),
        generation_config=generation_config,
        batch_size=batch_size,
    )
    references = frame[target_column].astype(str).tolist()
    scores = compute_text_metrics(predictions=predictions, references=references)

    return {
        "metrics": scores,
        "num_examples": len(references),
        "model_path": str(model_path),
        "test_path": str(test_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BLEU and chrF for a trained model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML training/evaluation config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = evaluate_test_dataset(config)

    output_path = Path(config["metrics"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
