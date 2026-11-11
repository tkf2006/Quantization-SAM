# 归档记录

本目录保留 316 次已完成运行，覆盖 8 批实验输出。清单见 [runs.json](runs.json)，分批计数见 [stages.json](stages.json)。它包含调参、诊断和确定性重复，不等于 316 个独立正式种子。

保留每次运行的 summary、实际配置、环境/模型身份、数据采样索引、metrics.jsonl、subspace_diagnostics.jsonl、basis 报告，以及各阶段 aggregate 和确定性核验结果。

所有从原 outputs 复制的 JSON/JSONL 均保持字节不变，[checksums.json](checksums.json) 记录其原始 SHA-256。两批 Stage 1 的配置相同但结果有差异，均保留以反映确定性问题的排查过程。

删除 adapter/optimizer 权重、重复 tokenizer、数据集/模型缓存、烟雾测试、失败/未完成运行。checkpoint_hashes.json 为删除前实际权重的散列，供确定性核验使用，不能替代权重或恢复训练。

配置中的原 outputs、服务器目录和 basis cache 路径均是历史值。新训练使用 configs/ 和 scripts/，新输出与历史 records 分开。
