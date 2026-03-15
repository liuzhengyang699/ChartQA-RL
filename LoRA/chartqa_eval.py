from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.chartqa.common import compare_chartqa_answers
from data.chartqa.sft import build_chartqa_messages, load_chartqa_splits, render_generation_inputs
from utils.config import (
    DEFAULT_CONFIG_PATH,
    apply_path_defaults,
    get_path_setting,
    get_nested,
    load_config,
    load_path_config,
    resolve_model_source,
)
from utils.utils import (
    save_json,
)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    bootstrap_args, _ = bootstrap.parse_known_args()
    config, _ = load_config(bootstrap_args.config)
    path_config, _ = load_path_config()

    parser = argparse.ArgumentParser(description="Evaluate a Qwen3-VL ChartQA checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=bootstrap_args.config,
        help="Path to the JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Merged model directory or standalone model directory.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
        help="Base model path, only used with --adapter_path.",
    )
    parser.add_argument(
        "--adapter_path",
        type=Path,
        default=None,
        help="Optional LoRA adapter directory. If set, model_path is ignored for weights.",
    )
    parser.add_argument(
        "--processor_path",
        type=Path,
        default=None,
        help="Optional processor directory. Defaults to adapter/model/base path in that order.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--subset", type=int, default=get_nested(config, "dataset.subset", -1))
    parser.add_argument("--seed", type=int, default=get_nested(config, "dataset.seed", 42))
    parser.add_argument("--test_size", type=float, default=get_nested(config, "dataset.test_size", 0.1))
    parser.add_argument("--eval_size", type=float, default=get_nested(config, "dataset.eval_size", 0.1))
    parser.add_argument("--split", choices=["train", "eval", "test"], default=get_nested(config, "eval.split", "test"))
    parser.add_argument("--max_samples", type=int, default=get_nested(config, "eval.max_samples", -1))
    parser.add_argument("--max_new_tokens", type=int, default=get_nested(config, "eval.max_new_tokens", 32))
    parser.add_argument(
        "--torch_dtype",
        choices=["bfloat16", "float16", "float32"],
        default=get_nested(config, "model.torch_dtype", "bfloat16"),
    )
    parser.add_argument("--device", type=str, default=get_nested(config, "eval.device", "cuda"))
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.base_model_name_or_path is None:
        args.base_model_name_or_path = resolve_model_source(config, path_config)
    args = apply_path_defaults(
        args,
        path_config,
        {
            "model_path": "sft_merged_dir",
            "data_dir": "chartqa_parquet_dir",
            "cache_dir": "hf_cache_dir",
        },
    )
    if args.output_path is None:
        args.output_path = get_path_setting(path_config, "sft_eval_dir") / f"{args.split}_metrics.json"
    else:
        args.output_path = args.output_path.expanduser().resolve()
    if args.adapter_path is not None:
        args.adapter_path = args.adapter_path.expanduser().resolve()
    if args.processor_path is not None:
        args.processor_path = args.processor_path.expanduser().resolve()
    return args


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_processor(args: argparse.Namespace) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    processor_source = args.processor_path or args.adapter_path or args.model_path or args.base_model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_source)
    processor.tokenizer.padding_side = "left"

    device = args.device
    dtype = resolve_torch_dtype(args.torch_dtype)
    if args.adapter_path is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if device != "cpu" else None,
        )
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model_path),
            torch_dtype=dtype,
            device_map="auto" if device != "cpu" else None,
        )
    if device == "cpu":
        model.to(device)
    model.eval()
    return model, processor


@torch.inference_mode()
def predict_one(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    sample: dict,
    device: str,
    max_new_tokens: int,
) -> str:
    messages = build_chartqa_messages(sample, include_answer=False)
    _, model_inputs = render_generation_inputs(processor, messages, device=device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed_ids = generated_ids[:, model_inputs.input_ids.shape[1] :]
    prediction = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return prediction.strip()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model, processor = load_model_and_processor(args)

    train_dataset, eval_dataset, test_dataset = load_chartqa_splits(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        subset=args.subset,
        seed=args.seed,
        test_size=args.test_size,
        eval_size=args.eval_size,
    )
    split_dataset = {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}[args.split]
    if args.max_samples > 0:
        split_dataset = split_dataset.select(range(min(args.max_samples, len(split_dataset))))

    exact_hits = 0
    relaxed_hits = 0
    similarity_sum = 0.0
    rows = []

    for sample in tqdm(split_dataset, desc=f"Evaluating {args.split} split"):
        prediction = predict_one(
            model=model,
            processor=processor,
            sample=sample,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        metrics = compare_chartqa_answers(prediction, sample)
        exact_hits += int(metrics["exact_match"])
        relaxed_hits += int(metrics["relaxed_match"])
        similarity_sum += metrics["similarity_score"]
        rows.append(
            {
                "query": sample["query"],
                "gold": metrics["gold"],
                "prediction": prediction,
                "exact_match": metrics["exact_match"],
                "relaxed_match": metrics["relaxed_match"],
                "similarity_score": metrics["similarity_score"],
            }
        )

    total = max(len(rows), 1)
    payload = {
        "split": args.split,
        "num_examples": len(rows),
        "exact_match": exact_hits / total,
        "relaxed_match": relaxed_hits / total,
        "avg_similarity": similarity_sum / total,
        "predictions": rows,
    }
    save_json(args.output_path, payload)


if __name__ == "__main__":
    main()
