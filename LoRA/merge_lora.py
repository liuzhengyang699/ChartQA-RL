from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from utils.config import (
    DEFAULT_CONFIG_PATH,
    apply_path_defaults,
    get_nested,
    load_config,
    load_path_config,
    resolve_model_source,
)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap.parse_known_args()
    config, _ = load_config(bootstrap_args.config)
    path_config, _ = load_path_config()

    parser = argparse.ArgumentParser(description="Merge a Qwen3-VL LoRA adapter into the base model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=bootstrap_args.config,
        help="Path to the JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--adapter_path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--processor_path",
        type=Path,
        default=None,
        help="Defaults to adapter_path, then base_model_name_or_path.",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["bfloat16", "float16", "float32"],
        default=get_nested(config, "model.torch_dtype", "bfloat16"),
    )
    args = parser.parse_args()
    if args.base_model_name_or_path is None:
        args.base_model_name_or_path = resolve_model_source(config, path_config)
    args = apply_path_defaults(
        args,
        path_config,
        {
            "adapter_path": "sft_adapter_dir",
            "output_path": "sft_merged_dir",
        },
    )
    if args.processor_path is not None:
        args.processor_path = args.processor_path.expanduser().resolve()
    return args


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_path))
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    processor_source = args.processor_path or args.adapter_path or args.base_model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_source)
    processor.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
