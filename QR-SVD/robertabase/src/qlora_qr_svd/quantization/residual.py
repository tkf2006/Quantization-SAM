from __future__ import annotations

import gc
import importlib.metadata
import re
from pathlib import Path
from typing import Any

import torch

from qlora_qr_svd.config import ProjectConfig
from qlora_qr_svd.qr_svd.basis import (
    LayerBasis,
    compute_svd_basis,
    load_basis_cache,
    save_basis_cache,
)


def _matches_target(name: str, targets: list[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


def _dequantize_linear4bit(module) -> torch.Tensor:
    import bitsandbytes as bnb

    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        raise RuntimeError("Linear4bit weight has no quant_state; ensure the module is on CUDA")
    return bnb.functional.dequantize_4bit(weight.data, quant_state)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def make_cache_metadata(quantized_model, config: ProjectConfig) -> dict[str, Any]:
    commit_hash = getattr(quantized_model.config, "_commit_hash", None)
    return {
        "model_name_or_path": config.model.name_or_path,
        "requested_revision": config.model.revision,
        "resolved_revision": commit_hash,
        "quant_bits": 4,
        "quant_type": config.model.quant_type,
        "double_quant": config.model.double_quant,
        "compute_dtype": config.model.compute_dtype,
        "bitsandbytes_version": importlib.metadata.version("bitsandbytes"),
        "svd_rank": config.sam.subspace_rank,
        "target_modules": sorted(config.lora.target_modules),
        "mode_normalization": "unit_frobenius_paired_uv",
        "residual_sign": "full_precision_minus_dequantized",
        "spectrum_statistics": "full_singular_values_v1",
    }


def cache_path_for(quantized_model, config: ProjectConfig) -> Path:
    metadata = make_cache_metadata(quantized_model, config)
    revision = metadata["resolved_revision"] or metadata["requested_revision"]
    target_tag = "-".join(metadata["target_modules"])
    filename = (
        f"{_safe_slug(config.model.name_or_path)}_rev-{_safe_slug(str(revision))}_"
        f"{config.model.quant_type}_dq-{int(config.model.double_quant)}_"
        f"rank-{config.sam.subspace_rank}_{target_tag}_spectrum-full-v1.pt"
    )
    return Path(config.qr_svd.cache_dir) / filename


@torch.no_grad()
def build_residual_bases(quantized_model, config: ProjectConfig) -> dict[str, LayerBasis]:
    import bitsandbytes as bnb
    from transformers import AutoModelForSequenceClassification

    resolved_revision = getattr(quantized_model.config, "_commit_hash", None)
    reference = AutoModelForSequenceClassification.from_pretrained(
        config.model.name_or_path,
        revision=resolved_revision or config.model.revision,
        num_labels=config.model.num_labels,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    reference_modules = dict(reference.named_modules())
    bases: dict[str, LayerBasis] = {}

    for name, module in quantized_model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit) or not _matches_target(
            name, config.lora.target_modules
        ):
            continue
        reference_module = reference_modules.get(name)
        if reference_module is None or not hasattr(reference_module, "weight"):
            raise RuntimeError(f"Full-precision reference layer not found: {name}")
        full_weight = reference_module.weight.detach().float().cpu()
        dequantized_weight = _dequantize_linear4bit(module).detach().float().cpu()
        if full_weight.shape != dequantized_weight.shape:
            raise RuntimeError(
                f"Weight shape mismatch for {name}: {full_weight.shape} vs {dequantized_weight.shape}"
            )
        residual = full_weight - dequantized_weight
        bases[name] = compute_svd_basis(residual, config.sam.subspace_rank)
        del full_weight, dequantized_weight, residual

    del reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not bases:
        raise RuntimeError(
            "No target Linear4bit layers were found. Check target_modules and quantization setup."
        )
    return bases


def load_or_build_residual_bases(
    quantized_model, config: ProjectConfig
) -> tuple[dict[str, LayerBasis], Path, bool]:
    metadata = make_cache_metadata(quantized_model, config)
    path = cache_path_for(quantized_model, config)
    if path.exists():
        return load_basis_cache(path, metadata), path, False
    bases = build_residual_bases(quantized_model, config)
    save_basis_cache(path, metadata, bases)
    return bases, path, True
