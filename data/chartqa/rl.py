from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


from .common import CHARTQA_RL_FIELD_NAMES


CHARTQA_DATASET_PREFIX = "ChartQA/ChartQA Dataset/"


def build_bbox_map(values: Any, bboxes: Any) -> dict[str, Any]:
    if isinstance(bboxes, dict):
        return bboxes or {"x1": "none"}
    if not isinstance(values, list) or not isinstance(bboxes, list):
        return {"x1": "none"}
    out: dict[str, Any] = {}
    for value, bbox in zip(values, bboxes):
        out[str(value)] = bbox
    return out or {"x1": "none"}


def normalize_chart_type(source: str | None) -> str | None:
    if not source:
        return None
    if source.startswith("chartqa_"):
        return source[len("chartqa_") :]
    return source


def to_figure_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    normalized = image_path.replace("\\", "/").strip()
    if not normalized:
        return None
    if normalized.startswith(CHARTQA_DATASET_PREFIX):
        return normalized
    for split in ("train", "val", "test"):
        split_prefix = f"{split}/"
        if normalized.startswith(split_prefix):
            return f"{CHARTQA_DATASET_PREFIX}{normalized}"
    raise ValueError(
        f"Expected a canonical ChartQA figure path under '{CHARTQA_DATASET_PREFIX}', got: {image_path}"
    )


def resolve_image_path(image_path: str | None, raw_dir: Path) -> Path:
    figure_path = to_figure_path(image_path)
    if figure_path is None:
        raise FileNotFoundError("Missing figure path in ChartQA RL record.")
    candidate = raw_dir / figure_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"image not found: {candidate}")


def load_vcot_records(split: str, raw_dir: Path) -> list[dict[str, Any]]:
    source_path = raw_dir / "chartqa_vcot" / f"{split}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing chartqa_vcot annotations for split '{split}': {source_path}")
    with source_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def build_structured_prompt(record: dict[str, Any]) -> str:
    return (
        f"<image> # USER REQUEST #: {record.get('question')}\n"
        "# USER Bounding Box Info: x_values_bbox, storing x values and coordinates. "
        "y_values_bbox, storing x values and coordinates. "
        f"The x values in the image are: {record.get('x_values', [])}. "
        f"The y values in the image are: {record.get('y_values', [])}.\n"
        "# USER IMAGE stored in image_1, as PIL image."
    )


def serialize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return "|||".join(str(item) for item in answer)
    return str(answer)


def build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": normalize_chart_type(record.get("source")),
        "figure_bbox": record.get("figure_bbox"),
        "x_values_bbox": build_bbox_map(record.get("x_values", []), record.get("x_values_bbox", [])),
        "y_values_bbox": build_bbox_map(record.get("y_values", []), record.get("y_values_bbox", [])),
    }


def prepare_rl_split(
    split: str,
    raw_dir: Path,
    output_dir: Path,
) -> Path:
    records = load_vcot_records(split=split, raw_dir=raw_dir)

    columns = {field_name: [] for field_name in CHARTQA_RL_FIELD_NAMES}
    for record in tqdm(records, desc=f"Preparing {split} split"):
        columns["figure_id"].append(str(record["id"]))
        columns["query"].append(record.get("question"))
        columns["prompt"].append(build_structured_prompt(record))
        columns["answer"].append(serialize_answer(record.get("answer")))

        image_path = record.get("image")
        columns["figure_path"].append(to_figure_path(image_path))
        columns["metadata"].append(json.dumps(build_metadata(record), ensure_ascii=False))

        with Image.open(resolve_image_path(image_path=image_path, raw_dir=raw_dir)) as image:
            image_format = image.format or "PNG"
            buffer = BytesIO()
            image.save(buffer, format=image_format)
            columns["images"].append([buffer.getvalue()])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_full.parquet"
    table = pa.table(
        {
            "metadata": pa.array(columns["metadata"], type=pa.string()),
            "figure_id": pa.array(columns["figure_id"], type=pa.string()),
            "figure_path": pa.array(columns["figure_path"], type=pa.string()),
            "query": pa.array(columns["query"], type=pa.string()),
            "prompt": pa.array(columns["prompt"], type=pa.string()),
            "answer": pa.array(columns["answer"], type=pa.string()),
            "images": pa.array(columns["images"], type=pa.list_(pa.binary())),
        }
    )
    pq.write_table(table, output_path)
    return output_path


def prepare_rl_parquet_splits(
    raw_dir: Path,
    output_dir: Path,
    splits: Iterable[str] = ("train", "val"),
) -> list[Path]:
    return [prepare_rl_split(split=split, raw_dir=raw_dir, output_dir=output_dir) for split in splits]
