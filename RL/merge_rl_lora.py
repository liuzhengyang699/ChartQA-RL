from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verl.rl_lora import ensure_rl_lora_checkpoint_dir, resolve_rl_lora_adapter_dir


def _require_peft():
    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError("Merging RL LoRA adapters requires `peft`. Install project dependencies first.") from exc
    return PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge an RL-only LoRA adapter into a merged SFT base model.")
    parser.add_argument("--adapter_path", type=Path, required=True, help="Adapter dir or RL checkpoint actor dir.")
    parser.add_argument(
        "--base_model_path",
        type=Path,
        default=None,
        help="Merged SFT base model directory. If omitted, use metadata from an RL checkpoint actor dir.",
    )
    parser.add_argument("--output_path", type=Path, required=True, help="Output directory for the merged full model.")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_base_model_path(adapter_path: Path, base_model_path: Path | None) -> Path:
    if base_model_path is not None:
        return base_model_path.resolve()

    metadata = ensure_rl_lora_checkpoint_dir(adapter_path)
    return Path(metadata["base_model_path"]).resolve()


def load_base_model(base_model_path: Path, trust_remote_code: bool):
    config = AutoConfig.from_pretrained(str(base_model_path), trust_remote_code=trust_remote_code)
    if type(config) in AutoModelForVision2Seq._model_mapping.keys():
        auto_class = AutoModelForVision2Seq
    else:
        auto_class = AutoModelForCausalLM

    model = auto_class.from_pretrained(
        str(base_model_path),
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(str(base_model_path), trust_remote_code=trust_remote_code)
    return model, processor


def main() -> None:
    args = parse_args()
    adapter_dir = resolve_rl_lora_adapter_dir(args.adapter_path)
    base_model_path = resolve_base_model_path(args.adapter_path, args.base_model_path)
    args.output_path.mkdir(parents=True, exist_ok=True)

    PeftModel = _require_peft()
    model, processor = load_base_model(base_model_path, args.trust_remote_code)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(args.output_path)
    if getattr(merged_model, "generation_config", None) is not None:
        merged_model.generation_config.save_pretrained(args.output_path)

    summary = {
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_dir),
        "output_path": str(args.output_path.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
