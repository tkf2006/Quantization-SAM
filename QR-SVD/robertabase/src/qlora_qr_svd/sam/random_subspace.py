from __future__ import annotations

import torch

from qlora_qr_svd.qr_svd.basis import LayerBasis


def compute_random_paired_basis(
    rows: int,
    columns: int,
    rank: int,
    generator: torch.Generator,
) -> LayerBasis:
    """Create orthonormal paired modes u_j v_j^T with a local CPU RNG."""
    effective_rank = min(rank, rows, columns)
    if rows <= 0 or columns <= 0 or effective_rank <= 0:
        raise ValueError("Random basis dimensions and rank must be positive")
    left_raw = torch.randn(rows, effective_rank, generator=generator, dtype=torch.float32)
    right_raw = torch.randn(
        columns, effective_rank, generator=generator, dtype=torch.float32
    )
    left = torch.linalg.qr(left_raw, mode="reduced").Q.contiguous()
    right = torch.linalg.qr(right_raw, mode="reduced").Q.contiguous()
    return LayerBasis(
        left=left,
        right=right,
        singular_values=torch.ones(effective_rank, dtype=torch.float32),
        residual_norm=0.0,
        captured_energy_ratio=0.0,
        source="random",
    )


def _matches_target(name: str, targets: list[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


@torch.no_grad()
def build_random_subspace_bases(
    quantized_model: torch.nn.Module,
    target_modules: list[str],
    rank: int,
    seed: int,
) -> dict[str, LayerBasis]:
    """Build a deterministic random basis for every target Linear4bit layer."""
    import bitsandbytes as bnb

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bases = {}
    for name, module in sorted(quantized_model.named_modules()):
        if not isinstance(module, bnb.nn.Linear4bit) or not _matches_target(
            name, target_modules
        ):
            continue
        bases[name] = compute_random_paired_basis(
            rows=int(module.out_features),
            columns=int(module.in_features),
            rank=rank,
            generator=generator,
        )
    if not bases:
        raise RuntimeError(
            "No target Linear4bit layers were found for the random SAM subspace"
        )
    return bases


def reuse_reference_singular_values(
    random_bases: dict[str, LayerBasis],
    reference_bases: dict[str, LayerBasis],
) -> dict[str, LayerBasis]:
    """Pair random directions with the QR residual's singular-value spectrum."""
    if set(random_bases) != set(reference_bases):
        missing_random = sorted(set(reference_bases).difference(random_bases))
        missing_reference = sorted(set(random_bases).difference(reference_bases))
        raise RuntimeError(
            "Random/reference layer mismatch: "
            f"missing_random={missing_random[:3]}, "
            f"missing_reference={missing_reference[:3]}"
        )
    weighted = {}
    for name, random_basis in sorted(random_bases.items()):
        reference = reference_bases[name]
        if random_basis.rank != reference.rank:
            raise RuntimeError(
                f"Random/reference rank mismatch for {name}: "
                f"{random_basis.rank} vs {reference.rank}"
            )
        weighted[name] = LayerBasis(
            left=random_basis.left,
            right=random_basis.right,
            singular_values=reference.singular_values.clone(),
            residual_norm=reference.residual_norm,
            captured_energy_ratio=reference.captured_energy_ratio,
            source="random_qr_spectrum",
            full_singular_values=(
                reference.full_singular_values.clone()
                if reference.full_singular_values is not None
                else None
            ),
        )
    return weighted


def random_basis_report(
    bases: dict[str, LayerBasis],
    seed: int,
    singular_value_exponent: float = 0.0,
    spectrum_source: str = "unit",
) -> dict[str, object]:
    return {
        "type": "random_orthonormal_paired_modes",
        "seed": seed,
        "normalization": "unit_frobenius_paired_uv",
        "singular_value_exponent": singular_value_exponent,
        "spectrum_source": spectrum_source,
        "layers": [
            {
                "name": name,
                "rank": basis.rank,
                "out_features": int(basis.left.shape[0]),
                "in_features": int(basis.right.shape[0]),
                "selected_singular_values": basis.singular_values.tolist(),
            }
            for name, basis in sorted(bases.items())
        ],
    }
