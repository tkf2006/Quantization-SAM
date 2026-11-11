# 正式扩展实验记录

- `projected_qsam/`：QR 和随机输出子空间，各 3 个 200-epoch run。
- `svd_random/`：随机配对模式，3 个 200-epoch run。
- `svd_residual/`：残差 SVD 配对模式，1 个 200-epoch run。
- `fp32_sgd_seed1/`：200-epoch FP32 微调的完整 JSON history。
- `ptq_pretrained_seed1/`：原始预训练模型的静态 PTQ 结果和校准索引。

量化训练目录保留原始 experiment.log、settings.log，以及规范为 200 行的 log.txt。指标数值未改动。三组原始 SAQ 和普通 QAT-SGD 的历史记录仅保存在工作区同级项目 SAQ-Reproduction/records 中。
本次已将 TensorBoard 中分析需要的逐 epoch 数值保留在文本日志及绘图输入中；删除重复事件文件、模型权重、数据、冒烟/失败运行及后台进程文件。
