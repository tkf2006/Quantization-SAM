#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage1_diagnostics_seed42}"

python -m qlora_qr_svd.cli.check_environment
python -m qlora_qr_svd.cli.train \
  --config configs/roberta_rte_20pct_random_subspace.yaml \
  --seed 42 \
  --data-seed 42 \
  --output-dir "${OUTPUT_ROOT}/e4_seed42"
python -m qlora_qr_svd.cli.train \
  --config configs/roberta_rte_20pct_qr_svd.yaml \
  --seed 42 \
  --data-seed 42 \
  --output-dir "${OUTPUT_ROOT}/e5_seed42"

echo "Stage-one E4/E5 diagnostics completed: ${OUTPUT_ROOT}"
