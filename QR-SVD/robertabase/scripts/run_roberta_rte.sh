#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python -m qlora_qr_svd.cli.check_environment
python -m qlora_qr_svd.cli.train --config configs/roberta_rte_20pct_qlora.yaml
python -m qlora_qr_svd.cli.train --config configs/roberta_rte_20pct_lora_sam.yaml
python -m qlora_qr_svd.cli.train --config configs/roberta_rte_20pct_random_subspace.yaml
python -m qlora_qr_svd.cli.train --config configs/roberta_rte_20pct_qr_svd.yaml
