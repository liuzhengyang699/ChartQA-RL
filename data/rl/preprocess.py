from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
LORA_ROOT = PROJECT_ROOT / "LoRA"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LORA_ROOT) not in sys.path:
    sys.path.insert(0, str(LORA_ROOT))

from data.chartqa.rl import prepare_rl_parquet_splits
from utils.config import get_path_setting, load_path_config


PATH_CONFIG, _ = load_path_config()


def _resolve_runtime_dir(path_key: str, legacy_env_var: str | None = None) -> Path:
    if legacy_env_var:
        override = os.environ.get(legacy_env_var)
        if override:
            return Path(override).expanduser().resolve()
    return get_path_setting(PATH_CONFIG, path_key)


def main() -> None:
    raw_dir = _resolve_runtime_dir("rl_raw_dir", legacy_env_var="CHARTQA_RAW_DIR")
    output_dir = _resolve_runtime_dir("rl_parquet_dir")
    extra_roots = (SCRIPT_DIR, PROJECT_ROOT)
    prepare_rl_parquet_splits(raw_dir=raw_dir, output_dir=output_dir, extra_roots=extra_roots)


if __name__ == "__main__":
    main()
