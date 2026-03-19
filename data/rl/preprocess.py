from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.chartqa.rl import prepare_rl_parquet_splits
from config.runtime import get_path_setting, load_path_config


PATH_CONFIG, _ = load_path_config()

def main() -> None:
    raw_dir = get_path_setting(PATH_CONFIG, "rl_raw_dir")
    output_dir = get_path_setting(PATH_CONFIG, "rl_parquet_dir")
    prepare_rl_parquet_splits(raw_dir=raw_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
