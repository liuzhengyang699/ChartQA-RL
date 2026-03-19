#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/configs/chartqa_qwen3vl4b.json}"

readarray -t CONFIG_VALUES < <(PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python3 - "$CONFIG_PATH" <<'PY'
import sys

from config.runtime import get_path_setting, load_path_config
from LoRA.utils.config import get_model_id, load_config

config, _ = load_config(sys.argv[1])
path_config, _ = load_path_config()

print(get_model_id(config))
print(get_path_setting(path_config, "base_model_dir"))
PY
)

MODEL_ID="${CONFIG_VALUES[0]}"
MODEL_DIR="${CONFIG_VALUES[1]}"

modelscope download --model "${MODEL_ID}" --local_dir "${MODEL_DIR}"
