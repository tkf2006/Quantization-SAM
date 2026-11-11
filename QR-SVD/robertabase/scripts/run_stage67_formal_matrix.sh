#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stage67_formal_matrix_$(date +%Y%m%d_%H%M%S)}"
FORMAL_SEEDS=(13 21 42 87 100)
RANDOM_BASIS_SEEDS=(12345 2026 3407)
SETTINGS=(rte5 rte20 rte100 mrpc20 cola20)

declare -A TASK=(
  [rte5]=rte [rte20]=rte [rte100]=rte [mrpc20]=mrpc [cola20]=cola
)
declare -A FRACTION=(
  [rte5]=0.05 [rte20]=0.20 [rte100]=1.00 [mrpc20]=0.20 [cola20]=0.20
)
declare -A DATASET_DIR=(
  [rte5]=artifacts/datasets/glue_rte_bcdcba7
  [rte20]=artifacts/datasets/glue_rte_bcdcba7
  [rte100]=artifacts/datasets/glue_rte_bcdcba7
  [mrpc20]=artifacts/datasets/glue_mrpc_bcdcba7
  [cola20]=artifacts/datasets/glue_cola_bcdcba7
)

for setting in "${SETTINGS[@]}"; do
  dataset_dir="${DATASET_DIR[$setting]}"
  if [[ ! -d "${dataset_dir}" ]]; then
    echo "Missing local dataset: ${dataset_dir}" >&2
    echo "Prepare/upload RTE, MRPC and CoLA datasets before starting Stage 6-7." >&2
    exit 2
  fi
done

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

run_method() {
  local setting="$1"
  local method="$2"
  local seed="$3"
  local config="$4"
  shift 4
  local suffix="${method}_seed${seed}"
  if [[ "${method}" == "e4" ]]; then
    local basis_seed="$1"
    shift
    suffix="${method}_basis${basis_seed}_seed${seed}"
  fi
  run_if_needed "${OUTPUT_ROOT}/runs/${setting}_${suffix}" \
    --config "${config}" \
    --task-name "${TASK[$setting]}" \
    --train-fraction "${FRACTION[$setting]}" \
    --local-dataset-dir "${DATASET_DIR[$setting]}" \
    --validation-source official \
    --seed "${seed}" \
    --data-seed "${seed}" \
    --deterministic \
    "$@"
}

echo "===== Stage 6-7: formal five-seed confirmation and task expansion ====="
echo "settings=${SETTINGS[*]}"
echo "train/data seeds=${FORMAL_SEEDS[*]}"
echo "E4 random basis seeds=${RANDOM_BASIS_SEEDS[*]}"
echo "output=${OUTPUT_ROOT}"

for setting in "${SETTINGS[@]}"; do
  echo "===== ${setting}: ${TASK[$setting]} fraction=${FRACTION[$setting]} ====="
  for seed in "${FORMAL_SEEDS[@]}"; do
    run_method "${setting}" e2 "${seed}" configs/stage67_formal_qlora.yaml
    run_method "${setting}" e3 "${seed}" configs/stage67_formal_lora_sam.yaml
    for basis_seed in "${RANDOM_BASIS_SEEDS[@]}"; do
      run_method "${setting}" e4 "${seed}" \
        configs/stage67_formal_random_subspace.yaml "${basis_seed}" \
        --random-basis-seed "${basis_seed}"
    done
    run_method "${setting}" e5 "${seed}" configs/stage67_formal_qr_svd.yaml
  done
done

python scripts/aggregate_stage67.py "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/stage67_aggregate.json"

echo "Stage 6-7 completed: ${OUTPUT_ROOT}"
