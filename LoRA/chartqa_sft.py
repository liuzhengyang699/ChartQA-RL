from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.chartqa.sft import collate_vlm_sft, load_chartqa_splits
from utils.config import (
    DEFAULT_CONFIG_PATH,
    apply_path_defaults,
    get_nested,
    load_config,
    load_path_config,
    resolve_model_source,
)
from utils.utils import (
    build_run_metadata,
    build_split_summary,
    save_json,
)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap.parse_known_args()
    config, _ = load_config(bootstrap_args.config)
    path_config, _ = load_path_config()

    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen3-VL-4B on ChartQA.")
    parser.add_argument(
        "--config",
        type=Path,
        default=bootstrap_args.config,
        help="Path to the JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Base Qwen3-VL model used for SFT. Defaults to config/paths.json and falls back to model_id if absent.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Directory containing ChartQA parquet shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Run directory. Checkpoints go to output_dir/checkpoints.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=get_nested(config, "dataset.subset", -1),
        help="Optional number of samples to keep.",
    )
    parser.add_argument("--seed", type=int, default=get_nested(config, "dataset.seed", 42))
    parser.add_argument("--test_size", type=float, default=get_nested(config, "dataset.test_size", 0.1))
    parser.add_argument("--eval_size", type=float, default=get_nested(config, "dataset.eval_size", 0.1))
    parser.add_argument("--num_train_epochs", type=int, default=get_nested(config, "sft.num_train_epochs", 3))
    parser.add_argument("--learning_rate", type=float, default=get_nested(config, "sft.learning_rate", 2e-4))
    parser.add_argument("--weight_decay", type=float, default=get_nested(config, "sft.weight_decay", 0.0))
    parser.add_argument("--warmup_ratio", type=float, default=get_nested(config, "sft.warmup_ratio", 0.03))
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=get_nested(config, "sft.per_device_train_batch_size", 1),
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=get_nested(config, "sft.per_device_eval_batch_size", 1),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=get_nested(config, "sft.gradient_accumulation_steps", 16),
    )
    parser.add_argument("--max_length", type=int, default=get_nested(config, "sft.max_length", 1024))
    parser.add_argument("--logging_steps", type=int, default=get_nested(config, "sft.logging_steps", 10))
    parser.add_argument("--eval_steps", type=int, default=get_nested(config, "sft.eval_steps", 200))
    parser.add_argument("--save_steps", type=int, default=get_nested(config, "sft.save_steps", 200))
    parser.add_argument("--save_total_limit", type=int, default=get_nested(config, "sft.save_total_limit", 3))
    parser.add_argument(
        "--torch_dtype",
        choices=["bfloat16", "float16", "float32"],
        default=get_nested(config, "model.torch_dtype", "bfloat16"),
    )
    parser.add_argument("--lora_r", type=int, default=get_nested(config, "sft.lora_r", 8))
    parser.add_argument("--lora_alpha", type=int, default=get_nested(config, "sft.lora_alpha", 16))
    parser.add_argument("--lora_dropout", type=float, default=get_nested(config, "sft.lora_dropout", 0.05))
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=get_nested(config, "sft.lora_target_modules", ["q_proj", "v_proj"]),
        help="Module names matched by PEFT.",
    )
    parser.add_argument(
        "--report_to",
        nargs="+",
        default=get_nested(config, "logging.report_to", ["swanlab"]),
        help="Training reporters, e.g. --report_to swanlab.",
    )
    parser.add_argument("--run_name", type=str, default=get_nested(config, "logging.run_name", "qwen3vl4b-chartqa-sft"))
    parser.add_argument(
        "--swanlab_project",
        type=str,
        default=get_nested(config, "logging.swanlab_project", "chartqa_rl_lora"),
        help="SwanLab project name. Ignored when report_to does not include swanlab.",
    )
    parser.add_argument(
        "--swanlab_mode",
        choices=["cloud", "local", "disabled"],
        default=get_nested(config, "logging.swanlab_mode", "cloud"),
        help="SwanLab running mode.",
    )
    parser.add_argument(
        "--swanlab_log_dir",
        type=str,
        default=get_nested(config, "logging.swanlab_log_dir"),
        help="Optional SwanLab local log directory.",
    )
    parser.add_argument(
        "--swanlab_api_key",
        type=str,
        default=get_nested(config, "logging.swanlab_api_key"),
        help="Optional SwanLab API key. Prefer environment variables in shared environments.",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        default=get_nested(config, "sft.merge_lora", False),
        help="Export a merged full model to output_dir/merged after training.",
    )
    args = parser.parse_args()
    if args.model_name_or_path is None:
        args.model_name_or_path = resolve_model_source(config, path_config)
    return apply_path_defaults(
        args,
        path_config,
        {
            "data_dir": "chartqa_parquet_dir",
            "output_dir": "sft_root",
            "cache_dir": "hf_cache_dir",
        },
    )


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def merge_adapter(
    model: PeftModel,
    processor: AutoProcessor,
    output_dir: Path,
) -> None:
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(merged_dir)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if "swanlab" in args.report_to:
        os.environ["SWANLAB_PROJECT"] = args.swanlab_project
        os.environ["SWANLAB_MODE"] = args.swanlab_mode
        if args.swanlab_log_dir:
            os.environ["SWANLAB_LOG_DIR"] = args.swanlab_log_dir
        if args.swanlab_api_key:
            os.environ["SWANLAB_API_KEY"] = args.swanlab_api_key

    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    train_dataset, eval_dataset, test_dataset = load_chartqa_splits(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        subset=args.subset,
        seed=args.seed,
        test_size=args.test_size,
        eval_size=args.eval_size,
    )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    processor.tokenizer.padding_side = "right"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collate_fn = partial(collate_vlm_sft, processor=processor)
    training_args = SFTConfig(
        output_dir=str(checkpoints_dir),
        run_name=args.run_name,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        lr_scheduler_type="constant",
        report_to=args.report_to,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )

    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(adapter_dir)

    if trainer.accelerator.is_main_process:
        save_json(
            args.output_dir / "run_config.json",
            build_run_metadata(
                args=vars(args),
                splits=build_split_summary(train_dataset, eval_dataset, test_dataset),
            ),
        )

    trainer.accelerator.wait_for_everyone()

    if args.merge_lora:
        if trainer.accelerator.num_processes != 1:
            raise RuntimeError(
                "--merge_lora currently requires single-process execution. "
                "Run merge_lora.py after distributed training."
            )
        merge_adapter(model, processor, args.output_dir)


if __name__ == "__main__":
    main()
