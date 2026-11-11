#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage45_weighting_rte20_$(date +%Y%m%d_%H%M%S)}"
AUX_SEEDS=(17 29 43)
RANDOM_BASIS_SEEDS=(12345 2026 3407)
EXPONENTS=(0 0.5 1)
RHO=0.05
RANK=8

mkdir -p "${OUTPUT_ROOT}/runs"
python -m qlora_qr_svd.cli.check_environment

p_tag() {
  printf '%s' "$1" | sed 's/\./p/g'
}

run_if_needed() {
  local output_dir="$1"
  shift
  if [[ -f "${output_dir}/summary.json" ]]; then
    echo "Skip completed run: ${output_dir}"
    return
  fi
  python -m qlora_qr_svd.cli.train "$@" --output-dir "${output_dir}"
}

echo "===== Stage4/5: spectral weighting and enhanced random controls ====="
echo "rho=${RHO}, rank=${RANK}, output=${OUTPUT_ROOT}"
for exponent in "${EXPONENTS[@]}"; do
  tag="$(p_tag "${exponent}")"
  for seed in "${AUX_SEEDS[@]}"; do
    for basis_seed in "${RANDOM_BASIS_SEEDS[@]}"; do
      run_if_needed \
        "${OUTPUT_ROOT}/runs/e4_spectrum_p${tag}_basis${basis_seed}_seed${seed}" \
        --config configs/stage45_roberta_rte_20pct_random_spectrum.yaml \
        --seed "${seed}" \
        --data-seed "${seed}" \
        --rho "${RHO}" \
        --subspace-rank "${RANK}" \
        --random-basis-seed "${basis_seed}" \
        --singular-value-exponent "${exponent}" \
        --deterministic
    done

    run_if_needed \
      "${OUTPUT_ROOT}/runs/e5_qr_p${tag}_seed${seed}" \
      --config configs/stage45_roberta_rte_20pct_qr_svd.yaml \
      --seed "${seed}" \
      --data-seed "${seed}" \
      --rho "${RHO}" \
      --subspace-rank "${RANK}" \
      --singular-value-exponent "${exponent}" \
      --deterministic
  done
done

python scripts/aggregate_stage45.py "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/stage45_aggregate.json"

echo "Stage4/5 completed: ${OUTPUT_ROOT}"
