from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LayerBasis:
    left: torch.Tensor
    right: torch.Tensor
    singular_values: torch.Tensor
    residual_norm: float
    captured_energy_ratio: float
    source: str = "svd"
    full_singular_values: torch.Tensor | None = None

    @property
    def rank(self) -> int:
        return self.left.shape[1]


def compute_svd_basis(residual: torch.Tensor, rank: int) -> LayerBasis:
    if residual.ndim != 2:
        raise ValueError("Residual must be a matrix")
    effective_rank = min(rank, min(residual.shape))
    if effective_rank <= 0:
        raise ValueError("SVD rank must be positive")
    matrix = residual.detach().to(device="cpu", dtype=torch.float32)
    left, singular_values, right_transpose = torch.linalg.svd(matrix, full_matrices=False)
    left = left[:, :effective_rank].contiguous()
    right = right_transpose[:effective_rank, :].transpose(0, 1).contiguous()
    top_values = singular_values[:effective_rank].contiguous()
    total_energy = singular_values.square().sum()
    captured_energy = top_values.square().sum()
    ratio = float((captured_energy / total_energy).item()) if total_energy.item() > 0 else 0.0
    return LayerBasis(
        left=left,
        right=right,
        singular_values=top_values,
        residual_norm=float(torch.linalg.vector_norm(matrix).item()),
        captured_energy_ratio=ratio,
        full_singular_values=singular_values.contiguous(),
    )


def cumulative_energy_by_rank(
    singular_values: torch.Tensor,
    ranks: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
) -> dict[str, float]:
    """Return cumulative squared-singular-value energy at requested ranks."""
    values = singular_values.detach().to(device="cpu", dtype=torch.float32).flatten()
    if values.numel() == 0:
        return {str(rank): 0.0 for rank in ranks}
    total_energy = values.square().sum()
    if total_energy.item() == 0:
        return {str(rank): 0.0 for rank in ranks}
    cumulative = values.square().cumsum(dim=0) / total_energy
    return {
        str(rank): float(cumulative[min(rank, values.numel()) - 1].item())
        for rank in ranks
    }


def project_onto_paired_modes(
    matrix: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Project a matrix onto span{u_j v_j^T} without forming dense modes."""
    coefficients = (left * matrix.matmul(right)).sum(dim=0)
    return (left * coefficients.unsqueeze(0)).matmul(right.transpose(0, 1))


def save_basis_cache(
    path: str | Path, metadata: dict[str, Any], layers: dict[str, LayerBasis]
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "metadata": metadata,
        "layers": {
            name: {
                "left": basis.left.cpu(),
                "right": basis.right.cpu(),
                "singular_values": basis.singular_values.cpu(),
                "residual_norm": basis.residual_norm,
                "captured_energy_ratio": basis.captured_energy_ratio,
                "source": basis.source,
                "full_singular_values": (
                    basis.full_singular_values.cpu()
                    if basis.full_singular_values is not None
                    else None
                ),
            }
            for name, basis in layers.items()
        },
    }
    torch.save(payload, output_path)


def load_basis_cache(
    path: str | Path, expected_metadata: dict[str, Any]
) -> dict[str, LayerBasis]:
    cache_path = Path(path)
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    if payload.get("format_version") not in {1, 2}:
        raise RuntimeError(f"Unsupported QR-SVD cache format: {cache_path}")
    actual_metadata = payload.get("metadata", {})
    for key, expected in expected_metadata.items():
        if actual_metadata.get(key) != expected:
            raise RuntimeError(
                f"QR-SVD cache metadata mismatch for {key}: "
                f"expected={expected!r}, actual={actual_metadata.get(key)!r}"
            )
    layers = {}
    for name, raw in payload["layers"].items():
        layers[name] = LayerBasis(
            left=raw["left"].float().contiguous(),
            right=raw["right"].float().contiguous(),
            singular_values=raw["singular_values"].float().contiguous(),
            residual_norm=float(raw["residual_norm"]),
            captured_energy_ratio=float(raw["captured_energy_ratio"]),
            source=str(raw.get("source", "svd")),
            full_singular_values=(
                raw["full_singular_values"].float().contiguous()
                if raw.get("full_singular_values") is not None
                else None
            ),
        )
    if not layers:
        raise RuntimeError(f"QR-SVD cache contains no layers: {cache_path}")
    return layers
