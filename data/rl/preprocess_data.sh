#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -n "${CHARTQA_RAW_DIR:-}" ]; then
    RAW_DIR="${CHARTQA_RAW_DIR}"
elif [ -n "${CHARTQA_RL_RAW_DIR:-}" ]; then
    RAW_DIR="${CHARTQA_RL_RAW_DIR}"
else
    RAW_DIR="$(PYTHONPATH="${PROJECT_ROOT}/LoRA:${PYTHONPATH:-}" python3 - <<'PY'
from utils.config import get_path_setting, load_path_config

path_config, _ = load_path_config()
print(get_path_setting(path_config, "rl_raw_dir"))
PY
)"
fi

mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"
export CHARTQA_RAW_DIR="${RAW_DIR}"

wget -c -O ChartQA.zip "https://huggingface.co/datasets/ReFocus/ReFocus_Data/resolve/main/images/ChartQA.zip?download=true"
unzip -qo ChartQA.zip
unzip -qo "${SCRIPT_DIR}/train_chartqa_vcot.zip"

python3 "${SCRIPT_DIR}/preprocess.py"
