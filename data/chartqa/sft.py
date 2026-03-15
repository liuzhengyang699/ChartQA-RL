from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info

from .common import extract_answer_text, find_parquet_files


SYSTEM_MESSAGE = (
    "You are a vision-language model specialized in chart question answering. "
    "Answer the question using only the chart image. "
    "Return a concise final answer, usually a number, short phrase, or entity name."
)
IGNORE_TOKEN_ID = -100


def load_chartqa_dataset(
    data_dir: str,
    cache_dir: str | None = None,
    subset: int = -1,
) -> Dataset:
    dataset = load_dataset("parquet", data_files=find_parquet_files(data_dir), split="train", cache_dir=cache_dir)
    if subset > 0:
        dataset = dataset.select(range(min(subset, len(dataset))))
    return dataset


def load_chartqa_splits(
    data_dir: str,
    cache_dir: str | None = None,
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
