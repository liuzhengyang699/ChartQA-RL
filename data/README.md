# ChartQA Data Layer

这个目录存放 LoRA 和 RL 共用的 ChartQA 数据层代码，不存放运行时下载数据或训练产物。

## 结构

- `data/chartqa/common.py`：共享答案归一化、数值相似度、基础 schema 常量。
- `data/chartqa/sft.py`：SFT 所需的 parquet 加载、split、message 构造、collator、生成输入。
- `data/chartqa/rl.py`：RL 原始 ChartQA + vcot 标注转 parquet 的预处理逻辑。
- `data/rl/preprocess.py`：RL parquet 预处理入口。
- `data/rl/preprocess_data.sh`：下载图像、解压标注并调用预处理脚本。
- `data/rl/train_chartqa_vcot.zip`：RL 预处理所需的标注压缩包。

## 说明

- LoRA 和 RL 共用的是底层 ChartQA 数据层，不是完全相同的 prompt 或样本格式。
- 运行时真实模型、原始数据、缓存和输出目录仍由 `config/paths.json` 控制。
- `rl_parquet_dir` 默认不仅包含 `train_full.parquet` / `val_full.parquet`，还会在训练时生成 `replay/` 目录保存高质量 rollout 样本池。
- RL 预处理的 canonical 入口现在是：

```bash
bash data/rl/preprocess_data.sh
```
