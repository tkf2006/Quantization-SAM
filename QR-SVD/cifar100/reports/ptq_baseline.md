# FP32-SGD 与 PTQ 基线：归档核验

本目录保留两份可核验的实验产物，注意它们不是同一条“微调后再量化”流水线。

| 流程 | Seed | FP32 Top-1 | PTQ W4A4 Top-1 | 记录位置 |
|---|---:|---:|---:|---|
| 从原始预训练模型进行 FP32-SGD 微调 200 epoch | 1 | 最佳 67.46%，末轮 66.66% | 尚无对应本地结果文件 | [results.json](../records/fp32_sgd_seed1/results.json) |
| 对原始预训练模型直接进行静态 PTQ | 校准 seed 1 | 67.50% | 41.07% | [results.json](../records/ptq_pretrained_seed1/results.json) |

历史报告曾记录“微调后的最佳 FP32 模型进行 PTQ，Top-1=45.01%，Top-5=75.36%”，但本地没有找到对应结果文件。该数字作为历史文字记录保留，不能视为已由归档产物核验，也不能替换直接 PTQ 的 41.07%。原始 source_checkpoint 路径表明，当前 PTQ JSON 输入为原始 r20_w32a32.pth。

## 设置

FP32：CIFAR-100 / PreResNet-20 / Nesterov SGD / seed 1 / 200 epoch / batch 128 / lr 0.01 / momentum 0.9 / weight decay 1e-4 / epoch 80、120 学习率乘 0.1。最佳模型出现在 epoch 1（从 0 编号）。

PTQ：首层 W8A32，主干 W4A4，末层 W8A8；per-tensor 均匀量化；从训练集取 1024 张图像进行激活校准；无随机数据增强；每层 MSE 搜索 81 个裁剪候选，范围 0.20–1.00。校准样本索引保存在 [calibration_indices.json](../records/ptq_pretrained_seed1/calibration_indices.json)。

## 复现入口

以下命令从 cifar100 根目录运行，路径由使用者提供。

```bash
python train_fp.py --data_path /path/to/data --pretrained /path/to/r20_w32a32.pth --output runs/fp32_seed1 --seed 1 --epochs 200
python ptq_eval.py --data_path /path/to/data --pretrained runs/fp32_seed1/best_fp32_model.pth --output runs/ptq_after_finetune --calib_size 1024 --calib_seed 1 --observer_samples_per_batch 32768 --threshold_candidates 81 --threshold_min_ratio 0.20
```

上述第二条产生的是“微调后 PTQ”的新结果。若复现已归档的 41.07% 直接 PTQ，应把 `--pretrained` 改为原始 `r20_w32a32.pth`，并单独设置新输出目录。
