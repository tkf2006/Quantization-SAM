# RoBERTa-base 实验配置

## 1. 实验目标

在冻结的 4-bit NF4 RoBERTa-base 主干上训练 LoRA，比较普通 QLoRA、LoRA 参数空间 SAM、随机低秩子空间 SAM 和量化残差 QR-SVD 子空间 SAM。

## 2. 实验组

| 组别 | 方法 | SAM 扰动空间 | 每步前/反向 |
| --- | --- | --- | ---: |
| E2 | NF4-QLoRA + AdamW | 无 | 1 |
| E3 | NF4-QLoRA + LoRA-SAM | LoRA 的 $A,B$ 参数空间 | 2 |
| E4 | NF4-QLoRA + Random-subspace SAM | 与 E5 同维度的随机成对低秩子空间 | 2 |
| E5 | NF4-QLoRA + QR-SVD-SAM | 真实量化残差的主导 SVD 子空间 | 2 |

E3–E5 的第一次前/反向用于寻找当前 batch 的一阶最坏扰动，第二次在扰动点计算梯度；最终均由 AdamW 更新 LoRA 和分类头。分类头参与训练，但不施加 SAM 扰动。

## 3. 统一配置

### 模型与量化

| 项目 | 配置 |
| --- | --- |
| 模型 | `FacebookAI/roberta-base` |
| 模型 revision | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| 分类类别 | 2 |
| 主干 | 冻结 |
| 权重量化 | 4-bit NF4 weight-only |
| Double quantization | 开启 |
| 计算精度 | FP16（RTX 2080 Ti 不启用 BF16） |
| Gradient checkpointing | 关闭 |
| 量化/LoRA目标层 | 12 层 Encoder 的 attention query、value |
| 分类头 | FP16、可训练、随 adapter 保存 |

### LoRA

| 项目 | 配置 |
| --- | ---: |
| Rank $r$ | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Bias | 不训练 |
| 可训练参数 | LoRA $A,B$ 与分类头 |

### 数据

| 项目 | 配置 |
| --- | --- |
| 数据集 | `nyu-mll/glue@bcdcba7` |
| 任务 | RTE、MRPC、CoLA |
| 训练集比例 | 100%、20%、5% |
| 验证集 | 完整官方 validation |
| 最大序列长度 | 256 |
| 抽样 | 按标签分层；5% 嵌套于 20%；各方法共享同一索引 |
| 随机种子 | 13、21、42、87、100 |

主要指标：RTE 报告 Accuracy；MRPC 报告 F1 和 Accuracy，以 F1 选最佳 checkpoint；CoLA 报告 Matthews correlation coefficient（MCC）。

### 训练

| 项目 | 配置 |
| --- | ---: |
| 优化器 | `torch.optim.AdamW` |
| 学习率 | $2\times10^{-4}$ |
| Weight decay | 0.01 |
| Scheduler | Linear decay |
| Warmup ratio | 0.06 |
| Epoch | 20 |
| Early-stopping patience | 5 |
| Train batch size | 16 |
| Eval batch size | 64 |
| Gradient accumulation | 1 |
| Gradient clipping | 1.0 |
| 设备 | 单卡 `cuda:0` |
| TF32 / BF16 | 关闭 |

## 4. SAM 与子空间配置

统一使用全局 $L_2$/Frobenius 扰动半径：

$$
\rho=0.05.
$$

- **E3：** 对全部 LoRA $A,B$ 梯度做全局归一化，构造标准两步 LoRA-SAM 扰动。
- **E4：** 每个 query/value 层构造 rank 8 的随机正交成对模式；随机 basis seed 固定为 `12345`，归一化和计算流程与 E5 相同。
- **E5：** 对每个目标层计算

  $$R_l=W_{0,l}-D(Q_{\mathrm{NF4}}(W_{0,l})),$$

  在 CPU 以 FP32 做截断 SVD，保留 rank 8 的单位模式

  $$D_{l,j}=u_{l,j}v_{l,j}^{\top}.$$

  奇异值只用于选择模式和统计残差能量，不缩放扰动。训练时将 clean gradient 投影到所有层的 QR-SVD 模式，做全局归一化后形成半径为 $\rho$ 的最坏扰动。

E4、E5 不显式构造稠密扰动矩阵；E5 的固定 basis 缓存在 `artifacts/cache/qr_svd`。所有扰动仅用于训练，推理时模型均为 `NF4 主干 + LoRA + 分类头`。

## 5. 公平比较与实验规模

- 四组共享模型、数据索引、LoRA 初始化、训练 seed、batch 顺序和全部训练超参数。
- E4 与 E5 共享子空间 rank、扰动半径、目标层、归一化方式及两步训练流程。
- E3–E5 两次前向恢复相同随机数状态，保证 dropout mask 一致。
- 每次运行记录任务指标、峰值显存、训练时间、可训练参数量和 clean/perturbed loss。
- E4、E5额外记录投影梯度能量比例；E5额外记录各层残差范数和 SVD 能量覆盖率。

正式实验共：

$$
4\ \text{组}\times3\ \text{任务}\times3\ \text{数据比例}\times5\ \text{种子}=180\ \text{次运行}.
$$

