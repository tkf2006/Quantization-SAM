#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/smoke_e3_e4}"

python -m qlora_qr_svd.cli.check_environment
python -m qlora_qr_svd.cli.train \
  --config configs/roberta_rte_20pct_lora_sam.yaml \
  --epochs 1 \
  --output-dir "${OUTPUT_ROOT}/e3"
python -m qlora_qr_svd.cli.train \
  --config configs/roberta_rte_20pct_random_subspace.yaml \
  --epochs 1 \
  --output-dir "${OUTPUT_ROOT}/e4"

echo "E3/E4 smoke runs completed: ${OUTPUT_ROOT}"
