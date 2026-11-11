#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage3_rank_rte20_$(date +%Y%m%d_%H%M%S)}"
AUX_SEEDS=(17 29 43)
RANDOM_BASIS_SEEDS=(12345 2026 3407)
RANKS=(4 8 16 32)
RHO=0.05

mkdir -p "${OUTPUT_ROOT}/runs"
python -m qlora_qr_svd.cli.check_environment

run_if_needed() {
  local output_dir="$1"
  shift
  if [[ -f "${output_dir}/summary.json" ]]; then
    echo "Skip completed run: ${output_dir}"
    return
  fi
  python -m qlora_qr_svd.cli.train "$@" --output-dir "${output_dir}"
}

echo "===== Stage3: paired E4/E5 rank search at rho=${RHO} ====="
for rank in "${RANKS[@]}"; do
  for seed in "${AUX_SEEDS[@]}"; do
    for basis_seed in "${RANDOM_BASIS_SEEDS[@]}"; do
      run_if_needed \
        "${OUTPUT_ROOT}/runs/e4_rho0p05_rank${rank}_basis${basis_seed}_seed${seed}" \
        --config configs/stage2_roberta_rte_20pct_random_subspace.yaml \
        --seed "${seed}" \
        --data-seed "${seed}" \
        --rho "${RHO}" \
        --subspace-rank "${rank}" \
        --random-basis-seed "${basis_seed}" \
        --deterministic
    done

    run_if_needed "${OUTPUT_ROOT}/runs/e5_rho0p05_rank${rank}_seed${seed}" \
      --config configs/stage2_roberta_rte_20pct_qr_svd.yaml \
      --seed "${seed}" \
      --data-seed "${seed}" \
      --rho "${RHO}" \
      --subspace-rank "${rank}" \
      --deterministic
  done
done

python scripts/aggregate_stage3.py "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/stage3_aggregate.json"

echo "Stage3 rank search completed: ${OUTPUT_ROOT}"
