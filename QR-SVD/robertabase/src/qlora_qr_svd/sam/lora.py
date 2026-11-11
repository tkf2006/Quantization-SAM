from __future__ import annotations

from typing import Any, Sequence

import torch


class LoraSAMPerturbation:
    """Apply a global L2 SAM perturbation to LoRA A/B parameters only.

    The sequence-classification head remains trainable in the perturbed pass,
    but it is intentionally excluded from the SAM perturbation.
    """

    def __init__(self, model: torch.nn.Module, rho: float) -> None:
        if rho < 0:
            raise ValueError("rho must be non-negative")
        selected = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if not selected:
            raise RuntimeError("No trainable LoRA parameters were found for LoRA-SAM")
        self.rho = float(rho)
        self.names = [name for name, _ in selected]
        self.parameters = [parameter for _, parameter in selected]
        self.mode = "disabled"
        self._originals: list[torch.Tensor] = []
        self.last_diagnostics: dict[str, Any] = {}

    def begin_clean_pass(self) -> None:
        if self.mode != "disabled":
            raise RuntimeError("LoRA-SAM clean pass requires a disabled perturbation")
        self.mode = "clean"

    def clean_tensors(self) -> tuple[torch.Tensor, ...]:
        if self.mode != "clean":
            raise RuntimeError("Clean tensors are available only during the clean pass")
        return tuple(self.parameters)

    @torch.no_grad()
    def begin_perturbed_pass(
        self, parameter_gradients: Sequence[torch.Tensor]
    ) -> dict[str, Any]:
        if self.mode != "clean":
            raise RuntimeError("begin_perturbed_pass must follow a clean gradient computation")
        if len(parameter_gradients) != len(self.parameters):
            raise ValueError("One clean gradient is required per LoRA parameter")

        gradients = []
        per_parameter_gradient_norm = {}
        for name, gradient in zip(self.names, parameter_gradients):
            gradient = gradient.detach().float()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite LoRA-SAM gradient in {name}")
            gradients.append(gradient)
            per_parameter_gradient_norm[name] = float(
                torch.linalg.vector_norm(gradient).item()
            )

        global_norm = torch.sqrt(
            sum(gradient.square().sum() for gradient in gradients)
        )
        scale = self.rho / (float(global_norm.item()) + 1e-12)
        self._originals = [parameter.detach().clone() for parameter in self.parameters]
        applied = []
        try:
            for parameter, gradient in zip(self.parameters, gradients):
                delta = (gradient * scale).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                parameter.add_(delta)
                applied.append(delta)
        except Exception:
            for parameter, original in zip(self.parameters, self._originals):
                parameter.copy_(original)
            self._originals.clear()
            self.mode = "disabled"
            raise

        perturbation_norm = torch.sqrt(
            sum(delta.detach().float().square().sum() for delta in applied)
        )
        self.mode = "perturbed"
        self.last_diagnostics = {
            "gradient_norm": float(global_norm.item()),
            "perturbation_norm": float(perturbation_norm.item()),
            "rho": self.rho,
            "per_parameter_gradient_norm": per_parameter_gradient_norm,
        }
        return self.last_diagnostics

    @torch.no_grad()
    def disable(self) -> None:
        if self._originals:
            for parameter, original in zip(self.parameters, self._originals):
                parameter.copy_(original)
        self._originals.clear()
        self.mode = "disabled"

    def close(self) -> None:
        self.disable()

    def report(self) -> dict[str, Any]:
        return {
            "type": "lora_parameter_sam",
            "rho": self.rho,
            "parameter_tensors": len(self.parameters),
            "parameter_elements": sum(parameter.numel() for parameter in self.parameters),
            "parameter_names": self.names,
        }
