from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

SCRIPT_DIR = Path(__file__).resolve().parent
RL_ROOT = SCRIPT_DIR
PROJECT_ROOT = RL_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from config.runtime import get_path_setting, load_path_config
from examples.reward_function.structured_chartqa import compute_structured_scores
from verl.tooluse.structured_chartqa import (
    build_baseline_answer_prompt,
    build_generation_messages,
    build_tool_answer_request,
    execute_validated_action,
    parse_action_response,
    validate_action_payload,
)


def parse_args() -> argparse.Namespace:
    path_config, _ = load_path_config()
    parser = argparse.ArgumentParser(description="Evaluate structured ChartQA RL models.")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Merged full model path. RL-only LoRA checkpoints must be exported first via RL/merge_rl_lora.py.",
    )
    parser.add_argument(
        "--data_file",
        type=Path,
        default=get_path_setting(path_config, "rl_parquet_dir") / "val_full.parquet",
    )
    parser.add_argument(
        "--action_prompt",
        type=Path,
        default=RL_ROOT / "examples" / "format_prompt" / "chartQA_action.jinja",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens_action", type=int, default=96)
    parser.add_argument("--max_new_tokens_answer", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rule_weight", type=float, default=0.4)
    parser.add_argument("--judge_weight", type=float, default=0.6)
    parser.add_argument("--tool_gain_weight", type=float, default=0.75)
    parser.add_argument("--invalid_penalty", type=float, default=1.0)
    parser.add_argument("--ineffective_penalty", type=float, default=0.25)
    parser.add_argument(
        "--judge_cache_path",
        type=Path,
        default=RL_ROOT / "judge" / "cache" / "structured_chartqa_judge_cache.jsonl",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.model_path.parent / "structured_eval" / args.model_path.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_model(model_path: Path, device: str):
    if (model_path / "adapter_config.json").exists() or (model_path / "metadata.json").exists():
        raise ValueError(
            "RL/evaluate_structured.py expects a merged full model directory. "
            "Merge the RL adapter first with `python RL/merge_rl_lora.py`."
        )
    processor = AutoProcessor.from_pretrained(str(model_path))
    processor.tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        config=config,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return model, processor


def decode_generation(model, processor, messages, images, device: str, max_new_tokens: int) -> str:
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
    if device.startswith("cuda"):
        model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    dataset = load_dataset("parquet", data_files=str(args.data_file), split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    model, processor = load_model(args.model_path, args.device)
    action_template = Template(args.action_prompt.read_text(encoding="utf-8").strip())

    records = []
    for sample in dataset:
        image_blob = sample["images"][0]
        if isinstance(image_blob, dict) and "bytes" in image_blob:
            image = Image.open(BytesIO(image_blob["bytes"])).convert("RGB")
        elif isinstance(image_blob, bytes):
            image = Image.open(BytesIO(image_blob)).convert("RGB")
        else:
            image = Image.open(image_blob).convert("RGB")

        action_prompt = action_template.render(content=sample["prompt"]).replace("<image>", "").strip()
        action_messages = build_generation_messages(action_prompt, 1)
        action_text = decode_generation(
            model,
            processor,
            action_messages,
            [image],
            device=args.device,
            max_new_tokens=args.max_new_tokens_action,
        )

        metadata = json.loads(sample["metadata"])
        parsed = parse_action_response(action_text)
        validated = validate_action_payload(parsed.get("payload"), metadata)
        executed = execute_validated_action(validated, image.copy(), metadata)

        baseline_prompt = build_baseline_answer_prompt(sample["query"])
        baseline_text = decode_generation(
            model,
            processor,
            build_generation_messages(baseline_prompt, 1),
            [image],
            device=args.device,
            max_new_tokens=args.max_new_tokens_answer,
        )

        tool_text = ""
        final_text = baseline_text
        answer_prompt = baseline_prompt
        if executed["decision"] == "tool" and executed["tool_exec_success"]:
            tool_request = build_tool_answer_request(sample["query"], executed)
            tool_text = decode_generation(
                model,
                processor,
                build_generation_messages(tool_request["prompt_text"], len(tool_request["images"])),
                tool_request["images"],
                device=args.device,
                max_new_tokens=args.max_new_tokens_answer,
            )
            if tool_text:
                final_text = tool_text
                answer_prompt = tool_request["prompt_text"]

        tool_cost = 0.05 + 0.01 * max(0, len(validated["targets"]) - 1) if validated["decision"] == "tool" and validated["valid"] else 0.0
        record = {
            "figure_id": sample["figure_id"],
            "figure_path": sample["figure_path"],
            "query": sample["query"],
            "ground_truth": sample["answer"],
            "action_prompt": action_prompt,
            "action_response_text": action_text,
            "action_target_json": validated["canonical_action_json"],
            "answer_prompt": answer_prompt,
            "baseline_answer_text": baseline_text,
            "tool_answer_text": tool_text,
            "final_answer_text": final_text,
            "tool_requested": validated["decision"] == "tool",
            "tool_executed": executed["tool_exec_success"],
            "invalid_action": validated["decision"] == "tool" and (not validated["valid"] or not executed["tool_exec_success"]),
            "tool_cost": tool_cost,
            "decision": validated["decision"],
            "action_valid": validated["valid"],
        }
        records.append(record)

    reward_records = compute_structured_scores(
        records,
        rule_weight=args.rule_weight,
        judge_weight=args.judge_weight,
        tool_gain_weight=args.tool_gain_weight,
        invalid_penalty=args.invalid_penalty,
        ineffective_penalty=args.ineffective_penalty,
        judge_cache_path=args.judge_cache_path,
    )

    success_mask = []
    per_example_path = args.output_dir / "per_example.jsonl"
    with per_example_path.open("w", encoding="utf-8") as file:
        for record, reward in zip(records, reward_records):
            merged = {**record, **reward}
            success_mask.append(bool(record["tool_executed"]))
            file.write(json.dumps(merged, ensure_ascii=False) + "\n")

    total = max(len(records), 1)
    tool_requested = sum(1 for record in records if record["tool_requested"])
    legal_actions = sum(1 for record in records if record["action_valid"])
    legal_tool_actions = sum(1 for record in records if record["tool_requested"] and record["action_valid"])
    tool_success = sum(1 for record in records if record["tool_executed"])
    invalid_actions = sum(1 for record in records if record["invalid_action"])
    tool_effective = sum(1 for reward, record in zip(reward_records, records) if record["tool_executed"] and reward["effective_tool"] > 0)
    avg_tool_gain = (
        sum(reward["tool_gain"] for reward, record in zip(reward_records, records) if record["tool_executed"]) / max(tool_success, 1)
    )
    summary = {
        "num_examples": len(records),
        "QAAccuracy": float(np.mean([reward["answer_accuracy"] for reward in reward_records])) if reward_records else 0.0,
        "ToolCallRate": tool_requested / total,
        "LegalActionRate": legal_actions / total,
        "ToolExecSuccessRate": tool_success / max(legal_tool_actions, 1),
        "ToolEffectivenessRate": tool_effective / max(tool_success, 1),
        "AvgToolGain": avg_tool_gain,
        "InvalidActionRate": invalid_actions / total,
        "JudgeScore": float(np.mean([reward["judge_score"] for reward in reward_records])) if reward_records else 0.0,
        "RewardScore": float(np.mean([reward["overall"] for reward in reward_records])) if reward_records else 0.0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
