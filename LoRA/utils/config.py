from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from config.runtime import get_path_setting

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "chartqa_qwen3vl4b.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return payload


def _resolve_path_from_file(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def load_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path = path.expanduser().resolve()
    return _load_json(path), path


def get_nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def resolve_config_path(config_path: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return _resolve_path_from_file(config_path, value)


def apply_path_defaults(args: Any, path_config: Mapping[str, Path], mapping: Mapping[str, str]) -> Any:
    for attr_name, path_key in mapping.items():
        current_value = getattr(args, attr_name)
        if current_value is None:
            setattr(args, attr_name, get_path_setting(path_config, path_key))
        elif isinstance(current_value, Path):
            setattr(args, attr_name, current_value.expanduser().resolve())
    return args


def get_model_id(config: dict[str, Any]) -> str:
    return get_nested(
        config,
        "model.model_id",
        get_nested(config, "model.name_or_path", "Qwen/Qwen3-VL-4B-Instruct"),
    )


def get_model_local_dir(path_config: Mapping[str, Path]) -> Path:
    return get_path_setting(path_config, "base_model_dir")


def resolve_model_source(config: dict[str, Any], path_config: Mapping[str, Path]) -> str:
    local_dir = get_model_local_dir(path_config)
    if local_dir.exists():
        return str(local_dir)
    return get_model_id(config)
