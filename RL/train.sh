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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl4b_chartqa_rl}"
PROJECT_NAME="${PROJECT_NAME:-chartqa_rl}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
REWARD_TYPE="${REWARD_TYPE:-structured_chartqa}"
REWARD_FUNCTION="${REWARD_FUNCTION:-./examples/reward_function/structured_chartqa.py:compute_structured_scores}"
RAW_IMAGE_DIR="${CHARTQA_RL_RAW_DIR:-${CHARTQA_RAW_DIR:-${PATH_VALUES[5]}}}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    echo "Run LoRA/merge_lora.py first or override MODEL_PATH." >&2
    exit 1
fi

export CHARTQA_RL_RAW_DIR="${RAW_IMAGE_DIR}"
export CHARTQA_RAW_DIR="${RAW_IMAGE_DIR}"

python3 -m verl.trainer.main \
    config="${CONFIG_PATH}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    worker.actor.micro_batch_size_per_device_for_update="${MICRO_BATCH_SIZE}" \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    trainer.save_checkpoint_path="${CHECKPOINT_DIR}" \
    trainer.replay.buffer_dir="${REPLAY_BUFFER_DIR}" \
    worker.reward.reward_type="${REWARD_TYPE}" \
    worker.reward.reward_function="${REWARD_FUNCTION}"
