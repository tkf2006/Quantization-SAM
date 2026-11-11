#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
SEEDS=(13 21 42 87 100)
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/e2_e5_rte20_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_ROOT}"

python -m qlora_qr_svd.cli.check_environment

for seed in "${SEEDS[@]}"; do
  python -m qlora_qr_svd.cli.train \
    --config configs/roberta_rte_20pct_qlora.yaml \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --output-dir "${OUTPUT_ROOT}/e2_seed${seed}"

  python -m qlora_qr_svd.cli.train \
    --config configs/roberta_rte_20pct_lora_sam.yaml \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --output-dir "${OUTPUT_ROOT}/e3_seed${seed}"

  python -m qlora_qr_svd.cli.train \
    --config configs/roberta_rte_20pct_random_subspace.yaml \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --output-dir "${OUTPUT_ROOT}/e4_seed${seed}"

  python -m qlora_qr_svd.cli.train \
    --config configs/roberta_rte_20pct_qr_svd.yaml \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --output-dir "${OUTPUT_ROOT}/e5_seed${seed}"
done

python scripts/aggregate_results.py "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/aggregate.json"
echo "All E2-E5 runs completed: ${OUTPUT_ROOT}"
