from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from qlora_qr_svd.qr_svd.basis import LayerBasis, cumulative_energy_by_rank


@dataclass
class _AttachedLayer:
    name: str
    module: torch.nn.Module
    basis: LayerBasis
    left_device: torch.Tensor | None = None
    right_device: torch.Tensor | None = None
    coefficient: torch.Tensor | None = None


class QRSVDPerturbation:
    """Inject a paired low-rank perturbation into frozen quantized linear layers.

    Each basis consists of paired unit-Frobenius modes u_j v_j^T. During the
    clean pass, zero coefficients require gradients. Their gradients are the
    full weight gradient projected onto the quantization-residual SVD modes.
    The global coefficient vector is normalized to radius rho for the second
    pass. No dense weight gradient or dense perturbation is materialized.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bases: dict[str, LayerBasis],
        rho: float,
        singular_value_exponent: float = 0.0,
    ) -> None:
        if rho < 0:
            raise ValueError("rho must be non-negative")
        if not math.isfinite(singular_value_exponent) or singular_value_exponent < 0:
            raise ValueError("singular_value_exponent must be finite and non-negative")
        modules = dict(model.named_modules())
        missing = sorted(set(bases).difference(modules))
        if missing:
            raise RuntimeError(f"Basis layers are missing from the quantized model: {missing[:3]}")
        self.rho = float(rho)
        self.singular_value_exponent = float(singular_value_exponent)
        self.layers = [
            _AttachedLayer(name=name, module=modules[name], basis=basis)
            for name, basis in sorted(bases.items())
        ]
        self._handles = [
            layer.module.register_forward_hook(self._make_hook(layer))
            for layer in self.layers
        ]
        self.mode = "disabled"
        self.last_diagnostics: dict[str, Any] = {}

    def _make_hook(self, layer: _AttachedLayer):
        def hook(_module, inputs, output):
            if self.mode == "disabled" or layer.coefficient is None:
                return output
            if not inputs:
                raise RuntimeError(f"QR-SVD hook for {layer.name} received no input tensor")
            hidden = inputs[0]
            if layer.left_device is None or layer.left_device.device != hidden.device:
                layer.left_device = layer.basis.left.to(hidden.device, dtype=torch.float32)
                layer.right_device = layer.basis.right.to(hidden.device, dtype=torch.float32)
            projected_input = hidden.float().matmul(layer.right_device)
            delta = (projected_input * layer.coefficient).matmul(layer.left_device.transpose(0, 1))
            return output + delta.to(dtype=output.dtype)

        return hook

    def begin_clean_pass(self) -> None:
        if self.mode != "disabled":
            raise RuntimeError("Subspace-SAM clean pass requires a disabled perturbation")
        self.mode = "clean"
        for layer in self.layers:
            device = layer.module.weight.device
            layer.coefficient = torch.zeros(
                layer.basis.rank,
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )

    def clean_coefficients(self) -> tuple[torch.Tensor, ...]:
        if self.mode != "clean":
            raise RuntimeError("Clean coefficients are available only during the clean pass")
        return tuple(layer.coefficient for layer in self.layers)

    def clean_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.clean_coefficients()

    def _summarize_gradients(
        self, coefficient_gradients: Sequence[torch.Tensor]
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if self.mode != "clean":
            raise RuntimeError("Clean-gradient diagnostics require an active clean pass")
        if len(coefficient_gradients) != len(self.layers):
            raise ValueError("One coefficient gradient is required per subspace layer")
        per_layer_gradient_norm: dict[str, float] = {}
        weighted_gradients: list[torch.Tensor] = []
        per_layer_weighted_gradient_norm: dict[str, float] = {}
        all_mode_weights: list[torch.Tensor] = []
        for layer, gradient in zip(self.layers, coefficient_gradients):
            gradient = gradient.detach().float()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    f"Non-finite subspace coefficient gradient in {layer.name}"
                )
            per_layer_gradient_norm[layer.name] = float(
                torch.linalg.vector_norm(gradient).item()
            )
            singular_values = layer.basis.singular_values.to(
                device=gradient.device, dtype=torch.float32
            )
            if singular_values.numel() != gradient.numel():
                raise RuntimeError(
                    f"Singular-value count mismatch for {layer.name}: "
                    f"{singular_values.numel()} vs {gradient.numel()}"
                )
            if not torch.isfinite(singular_values).all() or (singular_values < 0).any():
                raise FloatingPointError(
                    f"Invalid singular values for subspace layer {layer.name}"
                )
            mode_weights = singular_values.pow(self.singular_value_exponent)
            weighted_gradient = gradient * mode_weights
            weighted_gradients.append(weighted_gradient)
            all_mode_weights.append(mode_weights)
            per_layer_weighted_gradient_norm[layer.name] = float(
                torch.linalg.vector_norm(weighted_gradient).item()
            )

        gradient_energy = {
            name: value * value for name, value in per_layer_gradient_norm.items()
        }
        summed_gradient_energy = sum(gradient_energy.values())
        projected_gradient_norm = math.sqrt(summed_gradient_energy)
        denominator = summed_gradient_energy if summed_gradient_energy > 0 else 1.0
        per_layer_gradient_energy_share = {
            name: value / denominator for name, value in gradient_energy.items()
        }
        weighted_gradient_energy = {
            name: value * value
            for name, value in per_layer_weighted_gradient_norm.items()
        }
        summed_weighted_gradient_energy = sum(weighted_gradient_energy.values())
        weighted_gradient_norm = math.sqrt(summed_weighted_gradient_energy)
        weighted_denominator = (
            summed_weighted_gradient_energy
            if summed_weighted_gradient_energy > 0
            else 1.0
        )
        per_layer_weighted_gradient_energy_share = {
            name: value / weighted_denominator
            for name, value in weighted_gradient_energy.items()
        }
        mode_weights = torch.cat(all_mode_weights)
        mode_weight_energy = mode_weights.square()
        mode_weight_energy_sum = float(mode_weight_energy.sum().item())
        normalized_weight_energy = mode_weight_energy / max(mode_weight_energy_sum, 1e-30)
        summary = {
            # gradient_norm is the direction norm used to normalize the actual
            # perturbation. projected_gradient_norm remains the unweighted
            # task-gradient projection for comparisons across p.
            "gradient_norm": weighted_gradient_norm,
            "projected_gradient_norm": projected_gradient_norm,
            "weighted_gradient_norm": weighted_gradient_norm,
            "singular_value_exponent": self.singular_value_exponent,
            "per_layer_projected_gradient_norm": per_layer_gradient_norm,
            "per_layer_gradient_energy_share": per_layer_gradient_energy_share,
            "gradient_energy_share_sum": sum(per_layer_gradient_energy_share.values()),
            "gradient_energy_consistency_error": abs(
                summed_gradient_energy
                - projected_gradient_norm * projected_gradient_norm
            ),
            "per_layer_weighted_gradient_norm": per_layer_weighted_gradient_norm,
            "per_layer_weighted_gradient_energy_share": (
                per_layer_weighted_gradient_energy_share
            ),
            "weighted_gradient_energy_share_sum": sum(
                per_layer_weighted_gradient_energy_share.values()
            ),
            "weighted_gradient_energy_consistency_error": abs(
                summed_weighted_gradient_energy
                - weighted_gradient_norm * weighted_gradient_norm
            ),
            "mode_weight_min": float(mode_weights.min().item()),
            "mode_weight_max": float(mode_weights.max().item()),
            "mode_weight_effective_coordinates": float(
                1.0 / max(float(normalized_weight_energy.square().sum().item()), 1e-30)
            ),
        }
        return weighted_gradients, summary

    @torch.no_grad()
    def diagnose_clean_gradients(
        self, coefficient_gradients: Sequence[torch.Tensor]
    ) -> dict[str, Any]:
        """Measure projection strength without creating a SAM perturbation."""
        _gradients, summary = self._summarize_gradients(coefficient_gradients)
        self.last_diagnostics = summary
        self.disable()
        return summary

    @torch.no_grad()
    def begin_perturbed_pass(
        self, coefficient_gradients: Sequence[torch.Tensor]
    ) -> dict[str, Any]:
        if self.mode != "clean":
            raise RuntimeError("begin_perturbed_pass must follow a clean gradient computation")
        weighted_gradients, summary = self._summarize_gradients(coefficient_gradients)
        global_norm = float(summary["weighted_gradient_norm"])
        scale = self.rho / (global_norm + 1e-12)
        for layer, weighted_gradient in zip(self.layers, weighted_gradients):
            # coefficient is the actual weight-space coefficient multiplying
            # the unit-Frobenius mode. Thus ||delta W||_F remains rho for all p.
            layer.coefficient = (weighted_gradient * scale).detach()
        per_layer_perturbation_norm = {
            layer.name: float(torch.linalg.vector_norm(layer.coefficient).item())
            for layer in self.layers
        }
        perturbation_energy = {
            name: value * value for name, value in per_layer_perturbation_norm.items()
        }
        summed_perturbation_energy = sum(perturbation_energy.values())
        perturbation_norm = math.sqrt(summed_perturbation_energy)
        perturbation_denominator = (
            summed_perturbation_energy if summed_perturbation_energy > 0 else 1.0
        )
        per_layer_perturbation_energy_share = {
            name: value / perturbation_denominator
            for name, value in perturbation_energy.items()
        }
        self.mode = "perturbed"
        self.last_diagnostics = summary | {
            "perturbation_norm": perturbation_norm,
            "rho": self.rho,
            "per_layer_perturbation_norm": per_layer_perturbation_norm,
            "per_layer_perturbation_energy_share": per_layer_perturbation_energy_share,
            "perturbation_energy_share_sum": sum(
                per_layer_perturbation_energy_share.values()
            ),
            "perturbation_energy_consistency_error": abs(
                summed_perturbation_energy - perturbation_norm * perturbation_norm
            ),
        }
        return self.last_diagnostics

    def disable(self) -> None:
        self.mode = "disabled"
        for layer in self.layers:
            layer.coefficient = None

    def close(self) -> None:
        self.disable()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def basis_report(self) -> list[dict[str, Any]]:
        report = []
        for layer in self.layers:
            record: dict[str, float | int | str] = {
                "name": layer.name,
                "rank": layer.basis.rank,
                "source": layer.basis.source,
                "singular_value_exponent": self.singular_value_exponent,
                "selected_singular_values": layer.basis.singular_values.tolist(),
            }
            if layer.basis.source == "svd":
                spectrum = layer.basis.full_singular_values
                record.update(
                    {
                        "residual_norm": layer.basis.residual_norm,
                        "captured_energy_ratio": layer.basis.captured_energy_ratio,
                        "singular_values": (
                            spectrum.tolist() if spectrum is not None else None
                        ),
                        "cumulative_energy_by_rank": (
                            cumulative_energy_by_rank(spectrum)
                            if spectrum is not None
                            else None
                        ),
                    }
                )
            report.append(record)
        return report


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
