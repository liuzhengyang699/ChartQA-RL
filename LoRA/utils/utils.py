from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info


SYSTEM_MESSAGE = (
    "You are a vision-language model specialized in chart question answering. "
    "Answer the question using only the chart image. "
    "Return a concise final answer, usually a number, short phrase, or entity name."
)
IGNORE_TOKEN_ID = -100


def find_parquet_files(data_dir: str | os.PathLike[str]) -> list[str]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"ChartQA data directory does not exist: {data_dir}")

    files = sorted(str(path) for path in data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")
    return files


def load_chartqa_dataset(
    data_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    subset: int = -1,
) -> Dataset:
    dataset = load_dataset("parquet", data_files=find_parquet_files(data_dir), split="train", cache_dir=cache_dir)
    if subset > 0:
        dataset = dataset.select(range(min(subset, len(dataset))))
    return dataset


def load_chartqa_splits(
    data_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    subset: int = -1,
    seed: int = 42,
    test_size: float = 0.1,
    eval_size: float = 0.1,
) -> tuple[Dataset, Dataset, Dataset]:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if not 0.0 < eval_size < 1.0:
        raise ValueError(f"eval_size must be in (0, 1), got {eval_size}")
    if eval_size + test_size >= 1.0:
        raise ValueError("eval_size + test_size must be smaller than 1.0")

    dataset = load_chartqa_dataset(data_dir=data_dir, cache_dir=cache_dir, subset=subset)
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    train_val_dataset = split["train"]
    test_dataset = split["test"]
    relative_eval_size = eval_size / (1.0 - test_size)
    split = train_val_dataset.train_test_split(test_size=relative_eval_size, seed=seed)
    return split["train"], split["test"], test_dataset


def extract_answer_text(sample: dict[str, Any]) -> str:
    answer = sample["label"] if "label" in sample else sample["answer"]
    if isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)):
        answer = answer[0]
    return str(answer).strip()


def build_chartqa_messages(
    sample: dict[str, Any],
    include_answer: bool = True,
    system_message: str = SYSTEM_MESSAGE,
) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": str(sample["query"]).strip()},
            ],
        },
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": extract_answer_text(sample)}],
            }
        )
    return messages


def _vision_token_ids(processor: Any) -> set[int]:
    token_ids: set[int] = set()
    token_id_attrs = [
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ]
    for attr in token_id_attrs:
        value = getattr(processor, attr, None)
        if isinstance(value, int) and value >= 0:
            token_ids.add(value)

    token_attrs = [
        "image_token",
        "video_token",
        "vision_start_token",
        "vision_end_token",
    ]
    for attr in token_attrs:
        token = getattr(processor, attr, None)
        if token:
            token_id = processor.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id >= 0:
                token_ids.add(token_id)
    return token_ids


def collate_vlm_sft(examples: list[dict[str, Any]], processor: Any) -> dict[str, Any]:
    conversations = [build_chartqa_messages(example, include_answer=True) for example in examples]
    prompt_conversations = [build_chartqa_messages(example, include_answer=False) for example in examples]

    texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in conversations]
    prompt_texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in prompt_conversations
    ]
    image_inputs = [process_vision_info(conv)[0] for conv in conversations]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    prompt_batch = processor(text=prompt_texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = IGNORE_TOKEN_ID

    for token_id in _vision_token_ids(processor):
        labels[labels == token_id] = IGNORE_TOKEN_ID

    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
    for row_index, prompt_length in enumerate(prompt_lengths):
        labels[row_index, : int(prompt_length)] = IGNORE_TOKEN_ID

    batch["labels"] = labels
    return batch


def render_generation_inputs(processor: Any, messages: list[dict[str, Any]], device: str) -> tuple[str, Any]:
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(device)
    return prompt_text, model_inputs


def normalize_text(text: str) -> str:
    normalized = text.strip().lower().strip("\"'")
    normalized = re.sub(r"[.。]+$", "", normalized)
    return re.sub(r"\s+", " ", normalized)


def parse_numeric_variants(text: str) -> list[float]:
    normalized = normalize_text(text)
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("€", "")
    normalized = normalized.replace("£", "")
    normalized = normalized.strip()

    variants: list[float] = []
    candidates = [normalized]
    candidates.extend(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?", normalized))

    for candidate in candidates:
        try:
            variants.append(float(candidate))
        except ValueError:
            if candidate.endswith("%"):
                body = candidate[:-1].strip()
                try:
                    percentage_value = float(body)
                    variants.append(percentage_value)
                    variants.append(percentage_value / 100.0)
                except ValueError:
                    pass
    return list(dict.fromkeys(variants))


def numeric_similarity(a: float, b: float) -> float:
    if a == b:
        return 1.0
    if a == 0.0 or b == 0.0:
        return 0.0
    return max(0.0, 1.0 - (abs(a - b) / max(abs(a), abs(b))))


def compare_chartqa_answers(prediction: str, sample: dict[str, Any]) -> dict[str, Any]:
    gold = extract_answer_text(sample)
    normalized_prediction = normalize_text(prediction)
    normalized_gold = normalize_text(gold)
    exact_match = normalized_prediction == normalized_gold

    prediction_values = parse_numeric_variants(prediction)
    gold_values = parse_numeric_variants(gold)
    best_similarity = 1.0 if exact_match else 0.0
    for pred_value in prediction_values:
        for gold_value in gold_values:
            best_similarity = max(best_similarity, numeric_similarity(pred_value, gold_value))

    relaxed_match = exact_match or best_similarity >= 0.95
    return {
        "gold": gold,
        "exact_match": exact_match,
        "relaxed_match": relaxed_match,
        "similarity_score": best_similarity,
    }


def build_split_summary(train_dataset: Dataset, eval_dataset: Dataset, test_dataset: Dataset) -> dict[str, int]:
    return {
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "test_size": len(test_dataset),
    }


def build_run_metadata(args: dict[str, Any], splits: dict[str, int]) -> dict[str, Any]:
    return {"args": args, "splits": splits}


def save_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
