#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

readarray -t PATH_VALUES < <(PYTHONPATH="${PROJECT_ROOT}/LoRA:${PYTHONPATH:-}" python3 - <<'PY'
from pathlib import Path

from utils.config import get_path_setting, load_path_config

path_config, _ = load_path_config()
rl_parquet_dir = get_path_setting(path_config, "rl_parquet_dir")

print(get_path_setting(path_config, "sft_merged_dir"))
print(Path(rl_parquet_dir) / "train_full.parquet")
print(Path(rl_parquet_dir) / "val_full.parquet")
print(get_path_setting(path_config, "rl_checkpoint_dir"))
print(Path(rl_parquet_dir) / "replay")
print(get_path_setting(path_config, "rl_raw_dir"))
PY
)

CONFIG_PATH="${CONFIG_PATH:-examples/config.yaml}"
MODEL_PATH="${MODEL_PATH:-${PATH_VALUES[0]}}"
TRAIN_FILE="${TRAIN_FILE:-${PATH_VALUES[1]}}"
VAL_FILE="${VAL_FILE:-${PATH_VALUES[2]}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PATH_VALUES[3]}}"
REPLAY_BUFFER_DIR="${REPLAY_BUFFER_DIR:-${PATH_VALUES[4]}}"
RAW_IMAGE_DIR="${CHARTQA_RL_RAW_DIR:-${CHARTQA_RAW_DIR:-${PATH_VALUES[5]}}}"
LOGGERS="${LOGGERS:-console,swanlab}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    echo "Run LoRA/merge_lora.py first or override MODEL_PATH." >&2
    exit 1
fi

export CHARTQA_RL_RAW_DIR="${RAW_IMAGE_DIR}"
export CHARTQA_RAW_DIR="${RAW_IMAGE_DIR}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

if [ -n "${SWANLAB_API_KEY:-}" ]; then
    export SWANLAB_API_KEY
fi
if [ -n "${SWANLAB_LOG_DIR:-}" ]; then
    export SWANLAB_LOG_DIR
fi

CMD=(
    python3 -m verl.trainer.main
    "config=${CONFIG_PATH}"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "worker.actor.model.model_path=${MODEL_PATH}"
    "trainer.save_checkpoint_path=${CHECKPOINT_DIR}"
    "trainer.replay.buffer_dir=${REPLAY_BUFFER_DIR}"
)

normalize_bool() {
    local value="${1,,}"
    case "${value}" in
        1|true|yes|on)
            echo "true"
            ;;
        0|false|no|off)
            echo "false"
            ;;
        *)
            echo "${1}"
            ;;
    esac
}

append_override() {
    local env_name="$1"
    local config_key="$2"
    local value="${!env_name:-}"
    if [ -n "${value}" ]; then
        CMD+=("${config_key}=${value}")
    fi
}

append_bool_override() {
    local env_name="$1"
    local config_key="$2"
    local value="${!env_name:-}"
    if [ -n "${value}" ]; then
        CMD+=("${config_key}=$(normalize_bool "${value}")")
    fi
}

# Respect config.yaml by default; only override when the user explicitly
# provides environment variables.
LOGGERS="${LOGGERS// /}"
if [ -n "${LOGGERS}" ]; then
    CMD+=("trainer.logger=[${LOGGERS}]")
fi

append_override PROJECT_NAME trainer.project_name
append_override EXPERIMENT_NAME trainer.experiment_name
append_override N_GPUS_PER_NODE trainer.n_gpus_per_node
append_override TP_SIZE worker.rollout.tensor_parallel_size
append_override GLOBAL_BATCH_SIZE worker.actor.global_batch_size
append_override MICRO_BATCH_SIZE worker.actor.micro_batch_size_per_device_for_update
append_override ROLLOUT_BATCH_SIZE data.rollout_batch_size
append_override VAL_BATCH_SIZE data.val_batch_size
append_override REWARD_TYPE worker.reward.reward_type
append_override REWARD_FUNCTION worker.reward.reward_function
append_override KL_COEF algorithm.kl_coef
append_bool_override ENABLE_TOOL_BRANCH algorithm.enable_tool_branch
append_bool_override DISABLE_KL algorithm.disable_kl
append_bool_override USE_KL_LOSS algorithm.use_kl_loss

"${CMD[@]}"
