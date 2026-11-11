# cifar100

在官方 SAQ 基础上探索量化残差子空间引导的 SAM 扰动，并保留随机子空间对照、诊断指标及 FP32/PTQ 基线。

| 方法 | 已完成 seed | 最佳测试 Top-1 |
|---|---|---:|
| 原始 SAQ（历史参照） | 1、2、3 | 68.063 ± 0.068% |
| RS-SAQ，固定 rank-8 随机输出子空间 | 1、2、3 | 67.783 ± 0.061% |
| QR-SAQ，初始残差 QR 输出子空间 | 1、2、3 | 67.783 ± 0.060% |
| SVD-RS-SAQ，随机配对 rank-1 模式 | 1、2、3 | 67.837 ± 0.228% |
| SVD-QR-SAQ，残差 SVD 配对模式 | 1 | 67.700% |

每个量化训练 run 均为 200 epoch；± 为跨 seed 样本标准差，单 seed 不报告标准差。当前两轮结果均未显示残差引导方案稳定优于随机对照或原始 SAQ。原始 SAQ 的三组日志只保存在相邻的 `../../SAQ-Reproduction/records/`，避免重复存储；对照绘图要求 SAQ-Reproduction 保持在 QR-SVD 的同级目录，也可通过 `--saq-root` 指定位置。

- [QR 子空间实验](reports/qr_subspace_report.md)
- [SVD 配对模式实验](reports/svd_modes_report.md)
- [FP32/PTQ 基线与记录核验](reports/ptq_baseline.md)

## 运行

在本项目根目录运行，替换外部资产路径：

```bash
python scripts/run_experiment.py --method qr --seed 1 2 3 --data-root /path/to/data --pretrained /path/to/r20_w32a32.pth
python scripts/run_experiment.py --method rs --seed 1 2 3 --data-root /path/to/data --pretrained /path/to/r20_w32a32.pth
python scripts/run_experiment.py --method svd-rs --seed 1 2 3 --data-root /path/to/data --pretrained /path/to/r20_w32a32.pth
python scripts/run_experiment.py --method svd-qr --seed 1 --data-root /path/to/data --pretrained /path/to/r20_w32a32.pth
```

也支持 `--method saq` 和 `--method qat-sgd`。默认 rank=8；随机方法的 projection seed=10000+训练 seed，残差方法设置为 12345（不参与残差基计算）。可用 `--projection-seed` 覆盖。原报告中随机子空间种子写为 12345 的位置已按实际 settings.log 修正。

```bash
python analysis/plot_projected_qsam_curves.py
python analysis/plot_projected_qsam_diagnostics.py
python analysis/plot_svd_qr_saq_results.py
python -m unittest discover -s tests -v
```

## 环境与外部资产

使用 Python 3.10 或更高版本，在独立环境中安装 `requirements.txt`；按本机 CUDA 选择匹配的 PyTorch/torchvision。依赖清单不是原实验的精确环境锁定。

```bash
python -m pip install -r requirements.txt
```

自行准备 CIFAR-100 和原实验的 FP32 `r20_w32a32.pth` 初始化模型。数据布局为 `<DATA_ROOT>/cifar/cifar-100-python/{train,test,meta}`。预训练模型来源见[上游 Model Zoo](https://github.com/zhuang-group/SAQ/wiki/Model-Zoo)。数据集和权重不随本项目分发；仅有源代码与日志不能替代原始初始化权重。

固定配置：CIFAR-100、PreResNet-20、主干 W4A4、首层 W8A32、末层 W8A8、batch 128、200 epoch、初始学习率 0.01、epoch 80/120 衰减、Nesterov SGD。SAQ 使用 rho=0.4。

## 目录约定

- `core/`、`models/`、`quan_models/`、`utils/`：模型、量化、优化器和必要支持代码。模型注册和导入有依赖关系，保留这些模块。
- `scripts/`：可配置的训练入口；`analysis/`：根据归档记录重新绘图。
- `records/`：正式实验的配置、逐 epoch 数值和原始文本日志。
- `reports/`：实验说明；`reports/figures/`：唯一一份图表及汇总表。
- 新训练输出进入 `runs/`，由 `.gitignore` 排除；归档记录可正常提交。

原始日志中的服务器路径与旧目录名属于历史运行信息，不代表当前文件位置。`log.txt` 七列依次是 epoch、train Top-1 error、train loss、test Top-1 error、test loss、train Top-5 error、test Top-5 error；本次仅规范了行分隔，数值保持不变。

代码来源及修改说明见 [NOTICE.md](NOTICE.md)，保留上游 [Apache-2.0 许可证](LICENSE)。
