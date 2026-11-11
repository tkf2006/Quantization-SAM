from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any

import torch


PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets")


def collect_environment(strict: bool = True) -> dict[str, Any]:
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    report: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": sys.version.split()[0],
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    errors = []
    if platform.system() != "Linux":
        errors.append("This project configuration targets Linux only")
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        errors.append("Use Python 3.10 or 3.11")
    missing = [name for name, version in packages.items() if version is None]
    if missing:
        errors.append(f"Missing packages: {missing}")
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable")
    else:
        properties = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        report["gpu"] = {
            "name": properties.name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "total_memory_gib": properties.total_memory / (1024**3),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
        if capability < (6, 0):
            errors.append("NF4 requires compute capability 6.0 or newer")
        if properties.total_memory < 20 * 1024**3:
            errors.append("Expected approximately 22 GiB of GPU memory")
        if packages["bitsandbytes"] is not None:
            try:
                import bitsandbytes as bnb

                probe = torch.randn(64, 64, device="cuda", dtype=torch.float16)
                packed, state = bnb.functional.quantize_4bit(probe, quant_type="nf4")
                restored = bnb.functional.dequantize_4bit(packed, state)
                report["bitsandbytes_nf4_kernel"] = bool(torch.isfinite(restored).all().item())
                del probe, packed, restored
            except Exception as error:
                report["bitsandbytes_nf4_kernel"] = False
                errors.append(f"BitsAndBytes NF4 CUDA probe failed: {error}")
    report["errors"] = errors
    if strict and errors:
        raise RuntimeError("Environment check failed: " + "; ".join(errors))
    return report
