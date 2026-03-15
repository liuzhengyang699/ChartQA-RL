from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "chartqa_qwen3vl4b.json"
DEFAULT_PATH_CONFIG_PATH = REPO_ROOT / "config" / "paths.json"
EXAMPLE_PATH_CONFIG_PATH = REPO_ROOT / "config" / "paths.example.json"
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


def load_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path = path.expanduser().resolve()
    return _load_json(path), path


def load_path_config(path_config: str | Path | None = None) -> tuple[dict[str, Path], Path]:
    source = path_config
    source_label = "explicit"
    if source is None:
        env_override = os.environ.get(PATH_CONFIG_ENV)
        if env_override:
            source = env_override
            source_label = "env"
        else:
            source = DEFAULT_PATH_CONFIG_PATH
            source_label = "local"
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
        raise KeyError(
            f"ChartQA path config is missing required keys: {', '.join(missing_keys)}"
        )

    resolved: dict[str, Path] = {}
    for key in PATH_CONFIG_KEYS:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Path config value for '{key}' must be a non-empty string.")
        resolved[key] = _resolve_path_from_file(path, value)
    return resolved, path


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
