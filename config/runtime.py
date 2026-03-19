from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH_CONFIG_PATH = REPO_ROOT / "config" / "paths.json"
PATH_CONFIG_ENV = "CHARTQA_PATH_CONFIG"
PATH_CONFIG_KEYS = (
    "model_root",
    "data_root",
    "base_model_dir",
    "sft_root",
    "sft_adapter_dir",
    "sft_merged_dir",
    "sft_eval_dir",
    "rl_checkpoint_dir",
    "chartqa_parquet_dir",
    "hf_cache_dir",
    "rl_raw_dir",
    "rl_parquet_dir",
)
PATH_KEY_ENV_VARS = {key: f"CHARTQA_{key.upper()}" for key in PATH_CONFIG_KEYS}


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


def load_path_config(path_config: str | Path | None = None) -> tuple[dict[str, Path], Path]:
    source = path_config or os.environ.get(PATH_CONFIG_ENV) or DEFAULT_PATH_CONFIG_PATH
    source_label = "local" if source == DEFAULT_PATH_CONFIG_PATH else "explicit"
    path = Path(source).expanduser().resolve()
    if not path.exists():
        if source_label == "local":
            raise FileNotFoundError(
                "ChartQA path config not found. Create config/paths.json by copying "
                "config/paths.example.json, or set CHARTQA_PATH_CONFIG=/abs/path/to/paths.json."
            )
        raise FileNotFoundError(f"ChartQA path config not found: {path}")

    payload = _load_json(path)
    missing_keys = [key for key in PATH_CONFIG_KEYS if key not in payload]
    if missing_keys:
        raise KeyError(f"ChartQA path config is missing required keys: {', '.join(missing_keys)}")

    resolved: dict[str, Path] = {}
    for key in PATH_CONFIG_KEYS:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Path config value for '{key}' must be a non-empty string.")
        resolved[key] = _resolve_path_from_file(path, value)
    return resolved, path


def get_path_env_var(key: str) -> str:
    if key not in PATH_KEY_ENV_VARS:
        raise KeyError(f"Unknown ChartQA path config key: {key}")
    return PATH_KEY_ENV_VARS[key]


def get_path_setting(path_config: Mapping[str, Path], key: str) -> Path:
    env_var = get_path_env_var(key)
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    if key not in path_config:
        raise KeyError(f"ChartQA path config is missing key: {key}")
    value = path_config[key]
    return value if isinstance(value, Path) else Path(value).expanduser().resolve()
