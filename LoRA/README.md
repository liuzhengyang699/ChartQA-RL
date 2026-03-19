# LoRA / SFT

[返回项目首页](../README.md)

## 概览

这一部分负责 ChartQA 上的监督微调流程，包括基础模型下载、ChartQA parquet 数据下载、LoRA SFT、LoRA adapter 合并、离线评测和评测可视化。默认基础模型为 `Qwen3-VL-4B-Instruct`，LoRA 训练产物会继续作为 RL 的初始化模型使用。

所有命令都默认从仓库根目录执行：

```bash
cd /abs/path/to/project
```

## 目录与关键脚本

| 文件 | 作用 |
| --- | --- |
| `LoRA/chartqa_sft.py` | ChartQA LoRA SFT 主训练入口 |
| `LoRA/merge_lora.py` | 将 adapter 合并为可直接推理的 merged model |
| `LoRA/chartqa_eval.py` | 对 base / merged / adapter 进行离线评测 |
| `LoRA/visualize_metrics.py` | 根据评测 JSON 生成 SVG 概览图 |
| `LoRA/download_models.sh` | 下载基础视觉语言模型到 `base_model_dir` |
| `LoRA/download_data.sh` | 下载 `swift/ChartQA` parquet 数据到 `chartqa_parquet_dir` |
| `LoRA/configs/chartqa_qwen3vl4b.json` | SFT 与评测默认超参配置 |
| `LoRA/accelerate_config.yaml` | 可选的多卡 `accelerate` 配置模板 |
| `data/chartqa/sft.py` | LoRA 与评测共用的 ChartQA parquet 加载、split 与 message 构造 |
| `data/chartqa/common.py` | LoRA 与 RL 共用的答案归一化与匹配逻辑 |

## 路径与配置

运行前先准备本地路径配置：

```bash
cp config/paths.example.json config/paths.json
```

路径字段说明统一放在 [`config/README.md`](../config/README.md)。LoRA 主链实际会用到 `base_model_dir`、`chartqa_parquet_dir`、`hf_cache_dir`、`sft_root`、`sft_adapter_dir`、`sft_merged_dir` 和 `sft_eval_dir`。

底层数据加载与答案归一化逻辑已经集中在 [`data/chartqa/`](../data/README.md)，LoRA 目录只保留训练、merge、评测和可视化入口。

默认配置文件是 [`LoRA/configs/chartqa_qwen3vl4b.json`](./configs/chartqa_qwen3vl4b.json)，字段分组如下：

| 分组 | 内容 |
| --- | --- |
| `model` | 模型 ID、`torch_dtype` |
| `dataset` | 数据源、划分比例、随机种子 |
| `sft` | epoch、batch size、梯度累积、学习率、LoRA 超参 |
| `logging` | `report_to`、`run_name`、SwanLab 相关配置 |
| `eval` | 默认评测 split、设备、最大生成长度 |

CLI 参数优先级高于 JSON 配置，也高于路径配置文件中的默认目录。

## 数据与模型准备

下载基础模型：

```bash
bash LoRA/download_models.sh
```

下载 ChartQA parquet 数据：

```bash
bash LoRA/download_data.sh
```

这两个脚本会分别读取：

- `model.model_id`
- `dataset.source`
- `base_model_dir`
- `chartqa_parquet_dir`

默认下载工具是 `modelscope download`，因此需要提前安装并登录可访问对应资源的环境。

## SFT 训练

最小训练命令：

```bash
python LoRA/chartqa_sft.py --config LoRA/configs/chartqa_qwen3vl4b.json
```

如果希望把训练记录到 SwanLab 云端，推荐直接使用统一环境变量：

```bash
export SWANLAB_API_KEY=your_api_key
export SWANLAB_LOG_DIR=/abs/path/to/swanlab_logs  # 可选
python LoRA/chartqa_sft.py --config LoRA/configs/chartqa_qwen3vl4b.json
```

常见覆盖项示例：

```bash
python LoRA/chartqa_sft.py \
  --config LoRA/configs/chartqa_qwen3vl4b.json \
  --model_name_or_path /abs/path/to/base_model \
  --data_dir /abs/path/to/chartqa_parquet \
  --output_dir /abs/path/to/sft_run \
  --cache_dir /abs/path/to/hf_cache \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16
```

如果想在训练结束后直接导出 merged model，可以开启：

```bash
python LoRA/chartqa_sft.py \
  --config LoRA/configs/chartqa_qwen3vl4b.json \
  --merge_lora
```

注意：`--merge_lora` 当前只适用于单进程训练。多进程或多卡训练后，建议单独运行 `merge_lora.py`。

如果需要用 `accelerate` 启动多卡训练，可以使用：

```bash
accelerate launch \
  --config_file LoRA/accelerate_config.yaml \
  LoRA/chartqa_sft.py \
  --config LoRA/configs/chartqa_qwen3vl4b.json
```

训练脚本会：

- 从 `chartqa_parquet_dir` 读取数据并按 `test_size`、`eval_size` 划分 train / eval / test
- 将 checkpoint 写入 `output_dir/checkpoints`
- 将最终 adapter 写入 `output_dir/adapter`
- 将运行参数和 split 摘要写入 `output_dir/run_config.json`

## LoRA 合并

训练完成后，将 adapter 合并成完整模型：

```bash
python LoRA/merge_lora.py --config LoRA/configs/chartqa_qwen3vl4b.json
```

常见覆盖项示例：

```bash
python LoRA/merge_lora.py \
  --config LoRA/configs/chartqa_qwen3vl4b.json \
  --base_model_name_or_path /abs/path/to/base_model \
  --adapter_path /abs/path/to/adapter \
  --output_path /abs/path/to/merged_model
```

默认输入输出关系如下：

| 角色 | 默认路径键 |
| --- | --- |
| 基础模型 | `base_model_dir` |
| adapter | `sft_adapter_dir` |
| merged model | `sft_merged_dir` |

## 离线评测

默认评测 merged model：

```bash
python LoRA/chartqa_eval.py --config LoRA/configs/chartqa_qwen3vl4b.json
```

评测基础模型示例：

```bash
python LoRA/chartqa_eval.py \
  --config LoRA/configs/chartqa_qwen3vl4b.json \
  --model_path /abs/path/to/base_model \
  --output_path /abs/path/to/base_test_metrics.json
```

评测 merged model 示例：

```bash
python LoRA/chartqa_eval.py \
  --config LoRA/configs/chartqa_qwen3vl4b.json \
  --model_path /abs/path/to/merged_model \
  --output_path /abs/path/to/adapter_test_metrics.json
```

如果想评测 adapter，而不是 merged model，可以显式传入：

```bash
python LoRA/chartqa_eval.py \
  --config LoRA/configs/chartqa_qwen3vl4b.json \
  --base_model_name_or_path /abs/path/to/base_model \
  --adapter_path /abs/path/to/adapter
```

评测输出是一个 JSON 文件，默认保存到 `sft_eval_dir/<split>_metrics.json`。其中包含整体指标和逐样本预测。

指标含义：

| 指标 | 含义 |
| --- | --- |
| `Exact Match` | 预测答案与标注完全一致 |
| `Relaxed Match` | 数值或文本在较宽松规则下判为正确 |
| `Avg Similarity` | 数值相似度或字符串匹配得分的平均值 |

## 结果与可视化

当前离线评测结果基于 `ChartQA test split (3272 samples)`：

| Model | Exact Match | Relaxed Match | Avg Similarity |
| --- | ---: | ---: | ---: |
| Base | 72.0% | 84.7% | 90.6% |
| SFT (LoRA) | 74.8% | 86.8% | 92.0% |

如果你已经有 base 和 SFT 的评测 JSON，可以重新生成概览图：

```bash
python LoRA/visualize_metrics.py \
  --base-metrics /abs/path/to/base_test_metrics.json \
  --adapter-metrics /abs/path/to/adapter_test_metrics.json
```

训练曲线：

<table>
  <tr>
    <td align="center"><img src="../assets/sft_loss.png" alt="SFT Loss" width="100%"></td>
    <td align="center"><img src="../assets/sft_entropy.png" alt="SFT Entropy" width="100%"></td>
  </tr>
</table>

离线评测概览：

<table>
  <tr>
    <td align="center"><img src="../assets/lora_metrics_overview.svg" alt="LoRA Overview" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><img src="../assets/lora_metrics_delta.svg" alt="LoRA Delta" width="100%"></td>
  </tr>
</table>

## 输出产物

| 产物 | 默认位置 | 说明 |
| --- | --- | --- |
| 训练 checkpoint | `sft_root/checkpoints/` | `SFTTrainer` 保存的中间 checkpoint |
| 最终 adapter | `sft_root/adapter/` | 训练结束后的 PEFT adapter |
| merged model | `sft_root/merged/` 或 `sft_merged_dir` | 合并后的完整推理模型 |
| 运行摘要 | `sft_root/run_config.json` | 记录 CLI、配置和数据划分摘要 |
| 评测 JSON | `sft_eval_dir/*.json` | 整体指标和逐样本预测 |
| 可视化资产 | `assets/` | SVG 概览图和训练曲线图片 |

## 常见问题

**1. 为什么脚本提示找不到 `config/paths.json`？**  
先执行 `cp config/paths.example.json config/paths.json`，再按本机路径修改。

**2. `sft_root`、`sft_adapter_dir`、`sft_merged_dir` 是什么关系？**  
`sft_root` 是一次 SFT 运行的主目录，训练会在其中创建 `checkpoints/`、`adapter/`、`merged/`。`sft_adapter_dir` 和 `sft_merged_dir` 是路径配置里的默认落点，分别指向这个目录下的 `adapter` 和 `merged`。

**3. 什么时候评测 adapter，什么时候评测 merged model？**  
想做推理或作为 RL 初始化模型时，优先使用 merged model。想单独验证 LoRA 权重，也可以用 `--adapter_path` 加基础模型一起评测。

**4. 显存不足怎么办？**  
优先减小 `per_device_train_batch_size`、增加 `gradient_accumulation_steps`，必要时缩短 `max_length` 或降低评测时的 `max_new_tokens`。
