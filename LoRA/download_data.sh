#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/configs/chartqa_qwen3vl4b.json}"

readarray -t CONFIG_VALUES < <(PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python3 - "$CONFIG_PATH" <<'PY'
import sys

from utils.config import get_nested, get_path_setting, load_config, load_path_config

config, _ = load_config(sys.argv[1])
path_config, _ = load_path_config()

print(get_nested(config, "dataset.source"))
print(get_path_setting(path_config, "chartqa_parquet_dir"))
PY
)

DATASET_ID="${CONFIG_VALUES[0]}"
DATA_DIR="${CONFIG_VALUES[1]}"

modelscope download --dataset "${DATASET_ID}" --local_dir "${DATA_DIR}"
