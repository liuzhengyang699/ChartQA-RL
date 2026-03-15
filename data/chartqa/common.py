from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


CHARTQA_RL_FIELD_NAMES = (
    "metadata",
    "figure_id",
    "figure_path",
    "query",
    "prompt",
    "answer",
    "images",
)
CHARTQA_RL_METADATA_KEYS = (
    "type",
    "figure_bbox",
    "x_values_bbox",
    "y_values_bbox",
)
SUPPORTED_BAR_TYPES = frozenset({"v_bar", "h_bar"})


def find_parquet_files(data_dir: str | os.PathLike[str]) -> list[str]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"ChartQA data directory does not exist: {data_dir}")

    files = sorted(str(path) for path in data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")
    return files


def extract_answer_text(sample_or_answer: Any) -> str:
    if isinstance(sample_or_answer, Mapping):
        answer = sample_or_answer["label"] if "label" in sample_or_answer else sample_or_answer["answer"]
    else:
        answer = sample_or_answer
    if isinstance(answer, Sequence) and not isinstance(answer, (str, bytes)):
        answer = answer[0]
    return str(answer).strip()


def split_ground_truth_answers(sample_or_answer: Any) -> list[str]:
    answer = extract_answer_text(sample_or_answer)
    return [part.strip() for part in answer.split("|||") if part.strip()]


def split_prediction_answers(prediction: str) -> list[str]:
    return [part.strip() for part in str(prediction).split("||") if part.strip()]


def normalize_text(text: str) -> str:
    normalized = str(text).strip().lower().strip("\"'")
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


def candidate_similarity(candidate: str, ground_truth: str) -> float:
    normalized_candidate = normalize_text(candidate)
    normalized_ground_truth = normalize_text(ground_truth)
    if normalized_candidate == normalized_ground_truth:
        return 1.0

    best_similarity = 0.0
    candidate_values = parse_numeric_variants(candidate)
    ground_truth_values = parse_numeric_variants(ground_truth)
    for candidate_value in candidate_values:
        for ground_truth_value in ground_truth_values:
            best_similarity = max(best_similarity, numeric_similarity(candidate_value, ground_truth_value))
    return best_similarity


def compute_chartqa_match_score(prediction: str, sample_or_answer: Any) -> float:
    candidates = split_prediction_answers(prediction)
    ground_truth_parts = split_ground_truth_answers(sample_or_answer)
    if not candidates or not ground_truth_parts:
        return 0.0

    matched = 0.0
    for candidate in candidates:
        matched += max(candidate_similarity(candidate, ground_truth) for ground_truth in ground_truth_parts)
    return min(1.0, matched / max(len(ground_truth_parts), 1))


def compare_chartqa_answers(prediction: str, sample_or_answer: Any) -> dict[str, Any]:
    gold = extract_answer_text(sample_or_answer)
    normalized_prediction = normalize_text(prediction)
    normalized_gold = normalize_text(gold)
    exact_match = normalized_prediction == normalized_gold
    best_similarity = compute_chartqa_match_score(prediction, gold)
    relaxed_match = exact_match or best_similarity >= 0.95
    return {
        "gold": gold,
        "exact_match": exact_match,
        "relaxed_match": relaxed_match,
        "similarity_score": best_similarity,
    }
