# RoBERTa-base：量化残差 QR-SVD-SAM

在冻结的 4-bit NF4 RoBERTa-base 上训练 LoRA，比较不同 SAM 扰动空间。原项目目录为 `QR-SVD/QLoRA+QR-SVD`。

| 方法 | 定义 |
|---|---|
| E2 | QLoRA + AdamW，无 SAM |
| E3 | LoRA 参数 A/B 空间的 SAM |
| E4 | 随机正交低秩配对模式空间的 SAM |
| E5 | 量化残差 SVD 配对模式空间的 SAM |

分类头参与 AdamW 更新，但不施加 SAM 扰动。`QR` 在这里指 quantization residual；E5 主要使用残差 SVD 模式，不是仅对残差矩阵做 QR 分解。

## 已完成工作

归档 **316 次已完成运行**，包括初始对照、两批诊断、确定性重复、半径/rank/谱加权搜索，以及 **150 次 Stage 6–7 正式确认**。这些运行不能全部视为独立的正式训练种子；E4 的多 basis 先在相同训练 seed 内平均。

正式确认包含 RTE 5%/20%/100%、MRPC 20%、CoLA 20%，每个 setting 为 5 个训练 seed。E5 没有跨任务稳定优于 E4；MRPC 的 E5−E4 为 +0.56 个百分点，但探索性 95% 区间跨 0。

- [初始 E2–E5 实验](reports/initial_comparison.md)
- [阶段总览与实验索引](reports/param_tuning/experiment_index.md)
- [最终五任务确认](reports/param_tuning/stage6-7-正式确认与任务扩展.md)
- [完整配置说明](reports/experiment_settings.md)
- [机器可读运行清单](records/runs.json)、[各阶段运行数](records/stages.json)

## 项目结构

- `src/qlora_qr_svd/`：数据采样、NF4/LoRA 加载、残差 SVD、扰动和训练。
- `configs/`：各阶段 YAML 配置；`scripts/`：训练、聚合、数据准备与确定性核验入口。
- `analysis/`：绘图及归档结果重分析；`tests/`：原有 CPU 单元测试。
- `records/`：配置、环境、数据索引、逐步/逐 epoch 指标、诊断和阶段汇总。
- `reports/`：Markdown 报告与唯一一份 PNG 图表。

新运行继续写入被忽略的 `outputs/`；历史记录放在可提交的 `records/`。历史 JSON 中的旧服务器路径保留原值，属于运行来源信息。

## 环境与外部资产

完整训练使用 Linux/CUDA、Python 3.10 或 3.11。先按 GPU/CUDA 安装匹配的 PyTorch，再安装工程：

```bash
python -m pip install -e '.[dev]'
python -m qlora_qr_svd.cli.check_environment
```

原环境准备入口 `scripts/bootstrap_linux.sh` 固定 PyTorch 2.5.1/CUDA 11.8，其他核心依赖版本记录在 `pyproject.toml` 和每次运行的 `environment.json`。CPU 单元测试不等于完整 NF4/CUDA 训练验证。

自行准备固定 revision 的 RoBERTa-base 与 GLUE/RTE、MRPC、CoLA：模型 revision=`e2da8e2f811d1448a5b465c236feacd80ffbac7b`，数据 revision=`bcdcba7`。训练配置默认使用：

```text
artifacts/
├── hf_offline_bundle/             # 外部 Hugging Face 模型缓存
└── datasets/
    ├── glue_rte_bcdcba7/
    ├── glue_mrpc_bcdcba7/
    └── glue_cola_bcdcba7/
```

准备缓存后，从本项目根目录运行：

```bash
export HF_HOME="$PWD/artifacts/hf_offline_bundle"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m qlora_qr_svd.cli.train --config configs/stage67_formal_qr_svd.yaml --seed 13 --data-seed 13 --output-dir outputs/new_e5_seed13
```

`scripts/prepare_stage67_datasets.py --tasks rte mrpc cola` 可在可访问 Hugging Face 的环境中准备离线数据。完整正式矩阵使用 `bash scripts/run_stage67_formal_matrix.sh`；各阶段入口为 `scripts/run_stage*.sh`，应在全新输出目录中运行。

## 重新分析与测试

以下命令读取 records，不需要模型或数据集，也不启动训练：

```bash
python analysis/run_archived_analysis.py
python -m pytest
```

单阶段聚合示例：

```bash
python scripts/aggregate_stage67.py records/stage67_formal_matrix_20261008_085232 --output outputs/reanalysis/stage67_aggregate.json
```

归档不含 checkpoint tensor。删除前保留的 `checkpoint_hashes.json` 仅用于核验历史确定性；它不能还原模型，也不能用来恢复训练。原始记录的 SHA-256 清单见 [checksums.json](records/checksums.json)。

来源及许可见 [NOTICE.md](NOTICE.md)。
