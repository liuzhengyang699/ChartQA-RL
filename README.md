<div align="center">

# 面向 ChartQA 的工具增强多模态 RL

<p>基于 <code>Qwen3-VL-4B-Instruct</code> 的 ChartQA 项目，覆盖 <code>LoRA</code> 监督微调与基于 <code>VeRL</code> 的工具增强多模态 <code>RL</code>。</p>

</div>

当前仓库只保留两条主链：

- `LoRA`：ChartQA SFT、adapter merge、离线评测与可视化
- `RL`：ChartQA 结构化工具增强训练、数据预处理与统一评测

更多细节分别见 [`LoRA/README.md`](LoRA/README.md) 和 [`RL/README.md`](RL/README.md)。

## 快速开始

1. 创建环境

```bash
conda env create -f environment.yml
conda activate chartqa
python -m pip install -U pip setuptools wheel uv
uv pip install https://github.com/vllm-project/vllm/releases/download/v0.11.2/vllm-0.11.2+cu129-cp38-abi3-manylinux1_x86_64.whl \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
pip install -r requirements.txt
```

旧的 `chartqa-rl` 环境现在视为 legacy；主干默认使用新的 `chartqa` 环境。

2. 配置路径

```bash
cp config/paths.example.json config/paths.json
```

`config/paths.json` 控制本地路径；字段说明见 [`config/README.md`](config/README.md)。如需放在别处，可用 `CHARTQA_PATH_CONFIG=/abs/path/to/paths.json` 覆盖。

3. 下载模型并运行 SFT

```bash
bash LoRA/download_models.sh
bash LoRA/download_data.sh
python LoRA/chartqa_sft.py --config LoRA/configs/chartqa_qwen3vl4b.json
python LoRA/merge_lora.py --config LoRA/configs/chartqa_qwen3vl4b.json
python LoRA/chartqa_eval.py --config LoRA/configs/chartqa_qwen3vl4b.json
```

4. 预处理 RL 数据并启动训练

```bash
bash data/rl/preprocess_data.sh
cp RL/judge/judge_info.example.json RL/judge/judge_info.json
export SWANLAB_API_KEY=your_api_key
bash RL/train.sh
python RL/merge_rl_lora.py \
  --adapter_path /abs/path/to/rl_checkpoints/global_step_200/actor \
  --output_path /abs/path/to/rl_merged_model
python RL/evaluate_structured.py --model_path /abs/path/to/rl_merged_model
```

## 主链入口

- SFT 训练与评测：`python LoRA/chartqa_sft.py ...`、`python LoRA/merge_lora.py ...`、`python LoRA/chartqa_eval.py ...`
- RL 数据预处理：`bash data/rl/preprocess_data.sh`
- RL 训练与评测：`bash RL/train.sh`、`python RL/merge_rl_lora.py ...`、`python RL/evaluate_structured.py ...`

## 常用覆盖

```bash
MODEL_PATH=/abs/path/to/merged \
TRAIN_FILE=/abs/path/to/train_full.parquet \
VAL_FILE=/abs/path/to/val_full.parquet \
CHECKPOINT_DIR=/abs/path/to/rl_checkpoints \
REPLAY_BUFFER_DIR=/abs/path/to/replay_buffer \
bash RL/train.sh
```
