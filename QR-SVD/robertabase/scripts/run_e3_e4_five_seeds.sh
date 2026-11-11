#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
SEEDS=(13 21 42 87 100)
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/e3_e4_rte20_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_ROOT}"
python -m qlora_qr_svd.cli.check_environment

for seed in "${SEEDS[@]}"; do
  echo "===== E3 LoRA-SAM seed=${seed} ====="
  python -m qlora_qr_svd.cli.train \
    --config configs/roberta_rte_20pct_lora_sam.yaml \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --output-dir "${OUTPUT_ROOT}/e3_seed${seed}"

  echo "===== E4 random-subspace SAM seed=${seed} ====="
  python -m qlora_qr_svd.cli.train \
    --config configs/roberta_rte_20pct_random_subspace.yaml \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --output-dir "${OUTPUT_ROOT}/e4_seed${seed}"
done

python scripts/aggregate_results.py "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/aggregate.json"
echo "All E3/E4 runs completed: ${OUTPUT_ROOT}"
