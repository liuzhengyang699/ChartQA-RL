from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import torch
from PIL import Image

from ..models.transformers.qwen3_vl import ensure_qwen3_vl_processor, get_rope_index
from ..utils import torch_functional as VF
from .tools import (
    focus_on_x_values_with_draw,
    focus_on_x_values_with_highlight,
    focus_on_x_values_with_mask,
    focus_on_y_values_with_draw,
    focus_on_y_values_with_highlight,
    focus_on_y_values_with_mask,
)


ACTION_DECISIONS = {"direct", "tool"}
ACTION_AXES = {"x", "y"}
ACTION_MODES = {"mask", "draw", "highlight"}
DIRECT_ACTION = {
    "decision": "direct",
    "chart_axis": "x",
    "edit_mode": "highlight",
    "targets": [],
}

TOOL_FN_MAP = {
    ("x", "mask"): focus_on_x_values_with_mask,
    ("x", "draw"): focus_on_x_values_with_draw,
    ("x", "highlight"): focus_on_x_values_with_highlight,
    ("y", "mask"): focus_on_y_values_with_mask,
    ("y", "draw"): focus_on_y_values_with_draw,
    ("y", "highlight"): focus_on_y_values_with_highlight,
}


def canonical_action(action: Dict[str, Any]) -> Dict[str, Any]:
    if (action or {}).get("decision") != "tool":
        return dict(DIRECT_ACTION)
    return {
        "decision": "tool",
        "chart_axis": str(action["chart_axis"]),
        "edit_mode": str(action["edit_mode"]),
        "targets": [str(item) for item in action.get("targets", [])],
    }


def canonical_action_json(action: Dict[str, Any]) -> str:
    return json.dumps(canonical_action(action), ensure_ascii=False, sort_keys=True)


def extract_json_object(text: str) -> Optional[str]:
    content = (text or "").strip()
    if not content:
        return None
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```"):
            content = "\n".join(lines[1:-1]).strip()

    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return None


def parse_action_response(text: str) -> Dict[str, Any]:
    payload_text = extract_json_object(text)
    if payload_text is None:
        return {
            "raw_text": text,
            "payload": None,
            "parse_success": False,
            "error_code": "NO_JSON",
        }
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return {
            "raw_text": text,
            "payload": None,
            "parse_success": False,
            "error_code": "INVALID_JSON",
        }
    if not isinstance(payload, dict):
        return {
            "raw_text": text,
            "payload": None,
            "parse_success": False,
            "error_code": "JSON_NOT_OBJECT",
        }
    return {
        "raw_text": text,
        "payload": payload,
        "parse_success": True,
        "error_code": "",
    }


def _expected_axis(metadata_type: str | None) -> Optional[str]:
    if metadata_type == "v_bar":
        return "x"
    if metadata_type == "h_bar":
        return "y"
    return None


def candidate_labels(metadata: Dict[str, Any], axis: str) -> Dict[str, Any]:
    key = "x_values_bbox" if axis == "x" else "y_values_bbox"
    values = metadata.get(key) or {}
    if not isinstance(values, dict):
        return {}
    return {str(label): bbox for label, bbox in values.items()}


def validate_action_payload(payload: Optional[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "valid": False,
        "decision": "direct",
        "chart_axis": DIRECT_ACTION["chart_axis"],
        "edit_mode": DIRECT_ACTION["edit_mode"],
        "targets": [],
        "canonical_action": dict(DIRECT_ACTION),
        "canonical_action_json": canonical_action_json(DIRECT_ACTION),
        "illegal_action": True,
        "error_code": "",
        "tool_name": "",
    }
    if payload is None:
        result["error_code"] = "MISSING_PAYLOAD"
        return result

    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in ACTION_DECISIONS:
        result["error_code"] = "INVALID_DECISION"
        return result

    if decision == "direct":
        result.update(
            {
                "valid": True,
                "decision": "direct",
                "illegal_action": False,
                "error_code": "",
                "canonical_action": dict(DIRECT_ACTION),
                "canonical_action_json": canonical_action_json(DIRECT_ACTION),
            }
        )
        return result

    axis = str(payload.get("chart_axis", "")).strip().lower()
    edit_mode = str(payload.get("edit_mode", "")).strip().lower()
    targets = payload.get("targets", [])

    if axis not in ACTION_AXES:
        result["error_code"] = "INVALID_AXIS"
        return result
    if edit_mode not in ACTION_MODES:
        result["error_code"] = "INVALID_EDIT_MODE"
        return result
    if not isinstance(targets, list):
        result["error_code"] = "TARGETS_NOT_LIST"
        return result
    if len(targets) == 0:
        result["error_code"] = "EMPTY_TARGETS"
        return result

    normalized_targets = [str(item) for item in targets]
    if len(set(normalized_targets)) != len(normalized_targets):
        result["error_code"] = "DUPLICATE_TARGETS"
        return result

    expected_axis = _expected_axis(metadata.get("type"))
    if expected_axis is None:
        result["error_code"] = "UNSUPPORTED_CHART_TYPE"
        return result
    if axis != expected_axis:
        result["error_code"] = "AXIS_MISMATCH"
        return result

    candidates = candidate_labels(metadata, axis)
    if not candidates:
        result["error_code"] = "MISSING_CANDIDATES"
        return result

    if any(target not in candidates for target in normalized_targets):
        result["error_code"] = "UNKNOWN_TARGET"
        return result

    action = {
        "decision": "tool",
        "chart_axis": axis,
        "edit_mode": edit_mode,
        "targets": normalized_targets,
    }
    result.update(
        {
            "valid": True,
            "decision": "tool",
            "chart_axis": axis,
            "edit_mode": edit_mode,
            "targets": normalized_targets,
            "canonical_action": action,
            "canonical_action_json": canonical_action_json(action),
            "illegal_action": False,
            "error_code": "",
            "tool_name": f"focus_on_{axis}_values_with_{edit_mode}",
        }
    )
    return result


def execute_validated_action(action_result: Dict[str, Any], image: Image.Image, metadata: Dict[str, Any]) -> Dict[str, Any]:
    outcome = dict(action_result)
    outcome.update(
        {
            "tool_executed": False,
            "tool_exec_success": False,
            "tool_error_code": outcome.get("error_code", ""),
            "edited_image": None,
        }
    )
    if outcome.get("decision") != "tool":
        return outcome
    if not outcome.get("valid"):
        outcome["tool_error_code"] = outcome.get("error_code", "ILLEGAL_ACTION")
        return outcome

    tool_fn = TOOL_FN_MAP[(outcome["chart_axis"], outcome["edit_mode"])]
    bbox_mapping = candidate_labels(metadata, outcome["chart_axis"])

    try:
        edited_image = tool_fn(image.copy(), outcome["targets"], bbox_mapping)
        if not isinstance(edited_image, Image.Image):
            outcome["tool_error_code"] = "INVALID_TOOL_OUTPUT"
            return outcome
        outcome["tool_executed"] = True
        outcome["tool_exec_success"] = True
        outcome["edited_image"] = edited_image
        outcome["tool_error_code"] = ""
        return outcome
    except Exception as exc:  # pragma: no cover - defensive path
        outcome["tool_executed"] = True
        outcome["tool_error_code"] = type(exc).__name__
        return outcome


def _build_user_content(prompt_text: str, image_count: int) -> List[Dict[str, str]]:
    content: List[Dict[str, str]] = []
    for _ in range(image_count):
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt_text})
    return content


def build_baseline_answer_prompt(query: str) -> str:
    return (
        "Answer the ChartQA question using the original chart image.\n"
        f"Question: {query}\n"
        "Reply with a short explanation and end with `FINAL ANSWER: <answer>`."
    )


def build_tool_answer_prompt(query: str, action_result: Dict[str, Any]) -> str:
    targets = ", ".join(action_result.get("targets", []))
    return (
        "You are given a tool-focused chart image.\n"
        f"The tool-focused chart was created with {action_result['edit_mode']} on {action_result['chart_axis']}-values: {targets}.\n"
        f"Question: {query}\n"
        "Use the focused chart to answer the question.\n"
        "Reply with a short explanation and end with `FINAL ANSWER: <answer>`."
    )


def build_tool_answer_request(query: str, action_result: Dict[str, Any]) -> Dict[str, Any]:
    edited_image = action_result.get("edited_image")
    if not isinstance(edited_image, Image.Image):
        raise ValueError("tool answer request requires an edited image")
    return {
        "prompt_text": build_tool_answer_prompt(query, action_result),
        "images": [edited_image.copy()],
    }


def build_generation_messages(prompt_text: str, image_count: int) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": _build_user_content(prompt_text, image_count)}]


def build_supervised_messages(prompt_text: str, image_count: int, assistant_text: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": _build_user_content(prompt_text, image_count)},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def _prepare_images(images: List[Image.Image], process_image: Optional[Callable[[Image.Image], Image.Image]] = None) -> List[Image.Image]:
    if process_image is None:
        return images
    return [process_image(image) for image in images]


def _build_qwen3vl_position_ids(processor, input_ids: torch.Tensor, model_inputs: Dict[str, Any], attention_mask: torch.Tensor) -> torch.Tensor:
    processor = ensure_qwen3_vl_processor(processor)
    return get_rope_index(
        processor,
        input_ids=input_ids,
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        attention_mask=attention_mask,
    )


def build_generation_feature(
    processor,
    tokenizer,
    prompt_text: str,
    images: List[Image.Image],
    max_prompt_length: int,
    truncation: str,
    process_image: Optional[Callable[[Image.Image], Image.Image]] = None,
) -> Dict[str, Any]:
    messages = build_generation_messages(prompt_text, len(images))
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    processed_images = _prepare_images(images, process_image=process_image)
    model_inputs = processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
    input_ids = model_inputs.pop("input_ids")[0]
    attention_mask = model_inputs.pop("attention_mask")[0]
    position_ids = _build_qwen3vl_position_ids(processor, input_ids, model_inputs, attention_mask)

    input_ids, attention_mask, position_ids = VF.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=truncation,
    )
    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > max_prompt_length:
        if truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
        elif truncation == "right":
            raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
        else:
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "raw_prompt_ids": raw_prompt_ids,
        "multi_modal_data": {"image": processed_images},
        "multi_modal_inputs": dict(model_inputs),
        "prompt_text": prompt_text,
    }


def _postprocess_labels(labels: torch.Tensor, max_length: int, truncation: str) -> torch.Tensor:
    if labels.size(-1) < max_length:
        return VF.pad_sequence_to_length(labels, max_seq_len=max_length, pad_token_id=-100, left_pad=True)
    if labels.size(-1) > max_length:
        if truncation == "left":
            return labels[..., -max_length:]
        if truncation == "right":
            return labels[..., :max_length]
        raise RuntimeError(f"Label length {labels.size(-1)} is longer than {max_length}.")
    return labels


def build_supervised_feature(
    processor,
    tokenizer,
    prompt_text: str,
    images: List[Image.Image],
    assistant_text: str,
    max_prompt_length: int,
    truncation: str,
    process_image: Optional[Callable[[Image.Image], Image.Image]] = None,
    loss_weight: float = 1.0,
) -> Dict[str, Any]:
    processed_images = _prepare_images(images, process_image=process_image)
    user_messages = build_generation_messages(prompt_text, len(processed_images))
    full_messages = build_supervised_messages(prompt_text, len(processed_images), assistant_text)

    prompt = processor.apply_chat_template(user_messages, add_generation_prompt=True, tokenize=False)
    full_prompt = processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

    prompt_inputs = processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
    full_inputs = processor(processed_images, [full_prompt], add_special_tokens=False, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    input_ids = full_inputs.pop("input_ids")[0]
    attention_mask = full_inputs.pop("attention_mask")[0]
    position_ids = _build_qwen3vl_position_ids(processor, input_ids, full_inputs, attention_mask)

    labels = input_ids.clone()
    labels[:prompt_len] = -100

    input_ids, attention_mask, position_ids = VF.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation=truncation,
    )
    labels = _postprocess_labels(labels, max_prompt_length, truncation)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
        "loss_weight": torch.tensor(float(loss_weight), dtype=torch.float32),
        "multi_modal_data": {"image": processed_images},
        "multi_modal_inputs": dict(full_inputs),
    }
