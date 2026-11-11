from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from qlora_qr_svd.qr_svd.basis import LayerBasis
from qlora_qr_svd.qr_svd.perturbation import QRSVDPerturbation


class SubspaceComparisonDiagnostics:
    """Measure several subspace projections on the same clean forward pass.

    Every comparison manager injects exactly zero during the clean pass. Its
    coefficient gradients therefore measure alignment without modifying the
    model weights, optimizer gradients, or the primary SAM perturbation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        named_bases: dict[str, dict[str, LayerBasis]],
        singular_value_exponent: float = 0.0,
    ) -> None:
        if not named_bases:
            raise ValueError("At least one comparison subspace is required")
        self.managers = {
            name: QRSVDPerturbation(
                model=model,
                bases=bases,
                rho=0.0,
                singular_value_exponent=singular_value_exponent,
            )
            for name, bases in sorted(named_bases.items())
        }
        self._slices: dict[str, slice] = {}

    def begin_clean_pass(self) -> None:
        offset = 0
        self._slices = {}
        for name, manager in self.managers.items():
            manager.begin_clean_pass()
            count = len(manager.clean_tensors())
            self._slices[name] = slice(offset, offset + count)
            offset += count

    def clean_tensors(self) -> tuple[torch.Tensor, ...]:
        if not self._slices:
            raise RuntimeError("Comparison tensors require an active clean pass")
        return tuple(
            tensor
            for manager in self.managers.values()
            for tensor in manager.clean_tensors()
        )

    def collect(
        self, coefficient_gradients: Sequence[torch.Tensor]
    ) -> dict[str, dict[str, Any]]:
        if not self._slices:
            raise RuntimeError("Comparison collection requires an active clean pass")
        expected = sum(len(manager.layers) for manager in self.managers.values())
        if len(coefficient_gradients) != expected:
            raise ValueError(
                f"Expected {expected} comparison gradients, got {len(coefficient_gradients)}"
            )
        results = {
            name: manager.diagnose_clean_gradients(
                coefficient_gradients[self._slices[name]]
            )
            for name, manager in self.managers.items()
        }
        self._slices = {}
        return results

    def disable(self) -> None:
        self._slices = {}
        for manager in self.managers.values():
            manager.disable()

    def close(self) -> None:
        self._slices = {}
        for manager in self.managers.values():
            manager.close()
        self.managers.clear()

    def report(self) -> dict[str, Any]:
        return {
            "type": "same_clean_batch_subspace_projection",
            "comparison_subspaces": {
                name: {
                    "layers": len(manager.layers),
                    "coordinates": sum(layer.basis.rank for layer in manager.layers),
                    "rank_per_layer": sorted(
                        {layer.basis.rank for layer in manager.layers}
                    ),
                    "source": sorted({layer.basis.source for layer in manager.layers}),
                }
                for name, manager in self.managers.items()
            },
        }
