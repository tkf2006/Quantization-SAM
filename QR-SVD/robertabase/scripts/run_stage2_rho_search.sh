#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage2_rho_rte20_$(date +%Y%m%d_%H%M%S)}"
AUX_SEEDS=(17 29 43)
RANDOM_BASIS_SEEDS=(12345 2026 3407)
E3_RHOS=(0.0025 0.005 0.01 0.02)
E45_EQUAL_RHOS=(0.025 0.05 0.10)
E5_CALIBRATED_RHOS=(0.0625 0.125)
RANK=8

mkdir -p "${OUTPUT_ROOT}/determinism" "${OUTPUT_ROOT}/runs"
python -m qlora_qr_svd.cli.check_environment

rho_tag() {
  local value="$1"
  echo "${value/./p}"
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

echo "===== Stage2 gate: deterministic repeatability ====="
for method in e4 e5; do
  if [[ "${method}" == "e4" ]]; then
    config="configs/stage2_roberta_rte_20pct_random_subspace.yaml"
  else
    config="configs/stage2_roberta_rte_20pct_qr_svd.yaml"
  fi
  for repeat in a b; do
    run_if_needed "${OUTPUT_ROOT}/determinism/${method}_repeat_${repeat}" \
      --config "${config}" \
      --seed 17 \
      --data-seed 17 \
      --rho 0.05 \
      --subspace-rank "${RANK}" \
      --random-basis-seed 12345 \
      --epochs 3 \
      --deterministic
  done
  python scripts/verify_determinism.py \
    "${OUTPUT_ROOT}/determinism/${method}_repeat_a" \
    "${OUTPUT_ROOT}/determinism/${method}_repeat_b" \
    --output "${OUTPUT_ROOT}/determinism/${method}_verification.json"
done

echo "===== Stage2-A: E3 independent rho search ====="
for seed in "${AUX_SEEDS[@]}"; do
  for rho in "${E3_RHOS[@]}"; do
    tag="$(rho_tag "${rho}")"
    run_if_needed "${OUTPUT_ROOT}/runs/e3_rho${tag}_seed${seed}" \
      --config configs/stage2_roberta_rte_20pct_lora_sam.yaml \
      --seed "${seed}" \
      --data-seed "${seed}" \
      --rho "${rho}" \
      --deterministic
  done
done

echo "===== Stage2-B: E4 equal-radius search with three random bases ====="
for seed in "${AUX_SEEDS[@]}"; do
  for rho in "${E45_EQUAL_RHOS[@]}"; do
    tag="$(rho_tag "${rho}")"
    for basis_seed in "${RANDOM_BASIS_SEEDS[@]}"; do
      run_if_needed \
        "${OUTPUT_ROOT}/runs/e4_rho${tag}_rank${RANK}_basis${basis_seed}_seed${seed}" \
        --config configs/stage2_roberta_rte_20pct_random_subspace.yaml \
        --seed "${seed}" \
        --data-seed "${seed}" \
        --rho "${rho}" \
        --subspace-rank "${RANK}" \
        --random-basis-seed "${basis_seed}" \
        --deterministic
    done
  done
done

echo "===== Stage2-C: E5 equal-radius and calibrated-strength search ====="
for seed in "${AUX_SEEDS[@]}"; do
  for rho in "${E45_EQUAL_RHOS[@]}" "${E5_CALIBRATED_RHOS[@]}"; do
    tag="$(rho_tag "${rho}")"
    run_if_needed "${OUTPUT_ROOT}/runs/e5_rho${tag}_rank${RANK}_seed${seed}" \
      --config configs/stage2_roberta_rte_20pct_qr_svd.yaml \
      --seed "${seed}" \
      --data-seed "${seed}" \
      --rho "${rho}" \
      --subspace-rank "${RANK}" \
      --deterministic
  done
done

python scripts/aggregate_stage2.py "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/stage2_aggregate.json"

echo "Stage2 rho search completed: ${OUTPUT_ROOT}"
