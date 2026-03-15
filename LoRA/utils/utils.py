from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset


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
