# Quantization-SAM

量化与 Sharpness-Aware Minimization（SAM）的复现和方法探索。仓库保留实验代码、实际配置、训练记录及分析报告。

| 项目 | 研究问题 | 当前结果 |
|---|---|---|
| [SAQ-Reproduction](SAQ-Reproduction/README.md) | 低比特 QAT 中，SAQ 相对普通 SGD 有何收益？ | CIFAR-100/W4A4：SAQ 最佳 Top-1 为 68.06 ± 0.07%，当前单 seed QAT-SGD 为 66.50% |
| [QR-SVD](QR-SVD/README.md) | 量化残差提供的扰动子空间，是否优于随机方向？ | CIFAR-100 和 RoBERTa-base 两条路线尚未确认稳定优势 |

```text
Quantization-SAM/
├── SAQ-Reproduction/        # SAQ 基础复现与 QAT-SGD 对照
└── QR-SVD/
    ├── cifar100/            # PreResNet-20/W4A4，QR 与 SVD 扰动实验
    └── robertabase/         # 冻结 NF4 主干 + LoRA，GLUE 实验
```

## 阅读与复现

先阅读各项目 README 中的设置、结果和证据边界，再按照对应入口运行。`records/` 保存历史实验产物，`reports/` 保存报告和图表；新训练输出由各项目的忽略规则排除。

本仓库不分发数据集、模型初始化权重、训练 checkpoint 或离线模型缓存。重新训练需要准备相应外部资产；基于归档记录重新分析和绘图不需要运行完整训练。

所有 `±` 均为跨训练 seed 的样本标准差；单 seed 结果单独标明。RoBERTa 的正式结果使用 GLUE official validation，不能解释为隐藏 test 集成绩。

## 来源与许可

SAQ 代码来源及 Apache-2.0 许可见 [SAQ-Reproduction/NOTICE.md](SAQ-Reproduction/NOTICE.md)。QR-SVD 两条实现分别保留其[来源与许可说明](QR-SVD/README.md#来源与许可)。本仓库未对自有代码统一新增开源许可证。
