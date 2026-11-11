# Source and modifications

Training/model code is derived from zhuang-group/SAQ:
https://github.com/zhuang-group/SAQ

Paper: Sharpness-aware Quantization for Deep Neural Networks,
Jing Liu, Jianfei Cai and Bohan Zhuang (https://arxiv.org/abs/2111.12273).
The upstream Apache-2.0 license is preserved verbatim in LICENSE.

Local modifications include the ordinary QAT-SGD path, experiment records
and plotting utilities. SAQ_Task additionally includes projected QSAM,
SVD-mode variants, FP32 fine-tuning, PTQ evaluation and projection tests.
This is a reproduction/extension project, not the official upstream repository.
