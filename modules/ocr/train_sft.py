"""
SummerianOCR — Supervised Fine-Tuning (SFT) training script.

Usage:
    python train_sft.py --model_id PaddlePaddle/PaddleOCR-VL --output_dir ./outputs/sft
    python train_sft.py --enable_clearml --clearml_project NabuOCR
    python train_sft.py --help
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from trl import SFTConfig, SFTTrainer

from convert_atf import ATFConverter
from get_cdli_dataset import IMG_CACHE, get_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NabuOCR SFT Training")

    p.add_argument("--model_id", type=str, default="PaddlePaddle/PaddleOCR-VL")
    p.add_argument("--cache_dir", type=str, default="./hf_cache/models")
    p.add_argument("--dataset_path", type=str, default="./data/cdli_dataset.parquet")
    p.add_argument("--output_dir", type=str, default="./outputs/sft")

    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--lr_scheduler", type=str, default="linear")
    p.add_argument("--max_seq_length", type=int, default=16000)

    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=0)

    p.add_argument("--fp16", action="store_true", help="Use fp16 (for V100)")
    p.add_argument("--bf16", action="store_true", help="Use bf16 (for A100/H100)")

    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--enable_clearml", action="store_true",
                    help="Enable ClearML experiment tracking")
    p.add_argument("--clearml_project", type=str, default="NabuOCR")
    p.add_argument("--clearml_task", type=str, default="SFT Training")

    return p.parse_args()


def init_clearml(args: argparse.Namespace):
    """Initialize ClearML task if enabled. Returns (task, logger) or (None, None)."""
    if not args.enable_clearml:
        return None, None

    from clearml import Task

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task,
        auto_connect_frameworks={"pytorch": True, "tensorboard": True},
    )
    task.connect(vars(args), name="training_args")
    return task, task.get_logger()


@dataclass
class VisionDataCollator:
    """Processes vision-language examples into model-ready batches."""

    processor: Any
    max_length: int = 16000

    def __call__(self, examples: list[dict]) -> dict:
        texts = []
        images_list = []

        for ex in examples:
            img = Image.open(
                IMG_CACHE / f"P{str(ex['id']).rjust(6, '0')}.jpg"
            ).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "OCR:"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["unicode"]}],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images_list.append(img)

        batch = self.processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch


def expand_tokenizer(processor, train_dataset, test_dataset):
    """Add cuneiform signs and face markers to the tokenizer."""
    atf_converter = ATFConverter()
    used_signs = set()

    for example in train_dataset:
        parsed = atf_converter.parse(example["atf"])
        used_signs.update(parsed.get_used_signs())
    for example in test_dataset:
        parsed = atf_converter.parse(example["atf"])
        used_signs.update(parsed.get_used_signs())

    tokenizer = processor.tokenizer
    base_size = len(tokenizer)

    num_added = tokenizer.add_tokens(list(used_signs))
    num_special = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                f"@{face}" for face in atf_converter.ALL_FACES
            ]
            + atf_converter.SPECIAL_TOKENS
        },
        replace_additional_special_tokens=False,
    )

    logger.info(
        f"Tokenizer expanded: {base_size} -> {len(tokenizer)} "
        f"(+{num_added} tokens, +{num_special} special tokens)"
    )
    return len(tokenizer)


def main():
    args = parse_args()

    task, clearml_logger = init_clearml(args)

    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = get_dataset(args.dataset_path)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    logger.info(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")

    logger.info(f"Loading model: {args.model_id}")
    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    new_vocab_size = expand_tokenizer(processor, train_dataset, test_dataset)
    model.resize_token_embeddings(new_vocab_size)

    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory_gb = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}, Max memory = {max_memory_gb} GB")
    if clearml_logger:
        clearml_logger.report_single_value("gpu_name", gpu_stats.name)
        clearml_logger.report_single_value("gpu_memory_gb", max_memory_gb)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim="adamw_8bit",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        fp16=args.fp16,
        bf16=args.bf16,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=VisionDataCollator(
            processor=processor,
            max_length=args.max_seq_length,
        ),
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    logger.info("Starting SFT training")
    resume = args.resume_from_checkpoint if args.resume_from_checkpoint else False
    stats = trainer.train(resume_from_checkpoint=resume)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    runtime_min = stats.metrics["train_runtime"] / 60
    logger.info(f"Training finished in {runtime_min:.2f} minutes")
    logger.info(f"Peak GPU memory: {used_memory} GB")

    if clearml_logger:
        clearml_logger.report_single_value("train_runtime_min", round(runtime_min, 2))
        clearml_logger.report_single_value("peak_gpu_memory_gb", used_memory)
        clearml_logger.report_single_value("train_loss", stats.metrics.get("train_loss", -1))

    processor.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    logger.info(f"Model and processor saved to {args.output_dir}")

    if task:
        task.close()


if __name__ == "__main__":
    main()
