import argparse
import gc
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from compute_metrics import build_compute_metrics, load_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_target_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"[.!?]+", str(text)) if sentence.strip()]


def split_source_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in str(text).splitlines() if sentence.strip()]


def simple_sentence_aligner(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> pd.DataFrame:
    aligned_rows: list[dict[str, Any]] = []

    for row in frame.to_dict(orient="records"):
        source_parts = split_source_sentences(str(row[source_column]))
        target_parts = split_target_sentences(str(row[target_column]))

        if source_parts and len(source_parts) == len(target_parts):
            for source, target in zip(source_parts, target_parts, strict=True):
                aligned_row = dict(row)
                aligned_row[source_column] = source
                aligned_row[target_column] = target
                aligned_rows.append(aligned_row)
        else:
            aligned_rows.append(row)

    return pd.DataFrame(aligned_rows)


def validate_columns(
    frame: pd.DataFrame,
    path: Path,
    source_column: str,
    target_column: str,
) -> None:
    missing_columns = {source_column, target_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {sorted(missing_columns)}")


def load_training_frame(config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    train_path = Path(data_config["train_path"])
    source_column = data_config["source_column"]
    target_column = data_config["target_column"]

    frame = pd.read_csv(train_path)
    validate_columns(frame, train_path, source_column, target_column)

    if bool(data_config.get("align_sentences", False)):
        frame = simple_sentence_aligner(
            frame,
            source_column=source_column,
            target_column=target_column,
        )

    return frame


def build_datasets(config: dict[str, Any]) -> dict[str, Any]:
    from datasets import Dataset

    data_config = config["data"]
    frame = load_training_frame(config)
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    split = dataset.train_test_split(
        test_size=float(data_config["validation_size"]),
        seed=int(data_config["seed"]),
    )
    return {"train": split["train"], "eval": split["test"]}


def tokenize_datasets(
    datasets: dict[str, Any],
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    data_config = config["data"]
    model_config = config["model"]
    source_column = data_config["source_column"]
    target_column = data_config["target_column"]
    source_prefix = str(data_config["source_prefix"])

    def preprocess_function(examples: dict[str, list[Any]]) -> dict[str, Any]:
        inputs = [source_prefix + str(source) for source in examples[source_column]]
        targets = [str(target) for target in examples[target_column]]

        model_inputs = tokenizer(
            inputs,
            max_length=int(model_config["max_source_length"]),
            truncation=True,
        )
        labels = tokenizer(
            targets,
            max_length=int(model_config["max_target_length"]),
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    columns_to_remove = list(datasets["train"].column_names)
    return {
        name: dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=columns_to_remove,
        )
        for name, dataset in datasets.items()
    }


def normalize_report_to(value: Any) -> Any:
    if value is None:
        return "none"
    return value


def build_training_args(config: dict[str, Any]) -> Any:
    from transformers import Seq2SeqTrainingArguments

    training_config = config["training"]
    model_config = config["model"]
    generation_config = config["generation"]

    return Seq2SeqTrainingArguments(
        output_dir=str(model_config["output_dir"]),
        eval_strategy=str(training_config["eval_strategy"]),
        save_strategy=str(training_config["save_strategy"]),
        learning_rate=float(training_config["learning_rate"]),
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        num_train_epochs=float(training_config["num_train_epochs"]),
        predict_with_generate=bool(training_config["predict_with_generate"]),
        generation_max_length=int(generation_config["max_length"]),
        generation_num_beams=int(generation_config["num_beams"]),
        save_total_limit=int(training_config["save_total_limit"]),
        fp16=bool(training_config["fp16"]),
        report_to=normalize_report_to(training_config.get("report_to")),
        load_best_model_at_end=True,
    )


def train_model(config: dict[str, Any]) -> dict[str, float]:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
    )

    if "training" not in config:
        raise ValueError("Missing required config section: training")

    seed_everything(int(config["data"]["seed"]))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["name_or_path"])
    datasets = build_datasets(config)
    tokenized_datasets = tokenize_datasets(datasets, tokenizer, config)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = build_training_args(config)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(config["model"]["output_dir"])
    tokenizer.save_pretrained(config["model"]["output_dir"])

    return {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ByT5 translation model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML training config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = train_model(config)
    print(metrics)


if __name__ == "__main__":
    main()
