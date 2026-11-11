"""Projected variants of QSAM used by the first QR-SAQ experiment.

Both variants retain the original two-forward QSAM update.  The only changed
component is the perturbation of the effective, fake-quantized weight
(``module.x``): its clean-loss gradient is projected to a fixed, per-layer
output-channel subspace before the SAM perturbation is normalized.

``RandomSubspaceQSAM`` uses a seeded random orthonormal basis.  It is the
dimension-matched control for ``QuantizationResidualQSAM``, which obtains the
basis from a reduced QR decomposition of the initial quantization residual.
The bases are deliberately constructed once, immediately after the first
clean forward pass, and remain fixed for the entire run.
"""

from collections import defaultdict

import torch
import torch.nn as nn

from models.LIQ_wn_qsam import QConv2d, QLinear


class _ProjectedQSAM:
    """Common implementation for fixed-subspace QSAM perturbations."""

    def __init__(
        self,
        optimizer,
        model,
        rho=0.5,
        rank=8,
        include_wclip=False,
        include_aclip=False,
        include_bn=True,
    ):
        if rank <= 0:
            raise ValueError("projection rank must be positive")

        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.rank = rank
        self.include_wclip = include_wclip
        self.include_aclip = include_aclip
        self.include_bn = include_bn
        self.state = defaultdict(dict)

        # Module objects are stable during training and make the bookkeeping
        # robust to the nested names used by PreResNet blocks.
        self.bases = {}
        self._basis_ready = False
        self._initialization_report = []
        self.last_projection_energy_ratio = None

    @staticmethod
    def _is_quant_layer(module):
        return isinstance(module, (QConv2d, QLinear))

    @staticmethod
    def _as_matrix(tensor):
        return tensor.reshape(tensor.shape[0], -1)

    def _make_basis(self, residual, rank):
        """Return an orthonormal [out_features, rank] basis on residual device."""
        raise NotImplementedError

    @torch.no_grad()
    def _initialize_bases(self):
        if self._basis_ready:
            return

        report = []
        for name, module in self.model.named_modules():
            if not self._is_quant_layer(module):
                continue

            source = getattr(module, "quantization_source", None)
            quantized = getattr(module, "x", None)
            if source is None or quantized is None:
                raise RuntimeError(
                    "Projected QSAM needs quantization_source and x after the "
                    "first clean forward pass; missing values for '{}'".format(name)
                )
            if source.shape != quantized.shape:
                raise RuntimeError(
                    "Quantization source and effective weight have incompatible "
                    "shapes for '{}'".format(name)
                )

            residual = self._as_matrix((source.detach() - quantized.detach()).float())
            max_rank = min(residual.shape)
            effective_rank = min(self.rank, max_rank)
            basis = self._make_basis(residual, effective_rank)
            if basis.shape != (residual.shape[0], effective_rank):
                raise RuntimeError("Invalid basis shape for '{}'".format(name))
            self.bases[module] = basis.to(
                device=quantized.device, dtype=quantized.dtype, non_blocking=True
            )

            residual_norm_sq = residual.square().sum()
            captured_norm_sq = (basis.transpose(0, 1).matmul(residual)).square().sum()
            captured_ratio = (
                (captured_norm_sq / residual_norm_sq).item()
                if residual_norm_sq.item() > 0
                else 0.0
            )
            report.append(
                {
                    "name": name,
                    "rank": effective_rank,
                    "residual_norm": residual_norm_sq.sqrt().item(),
                    "residual_energy_ratio": captured_ratio,
                }
            )

        if not self.bases:
            raise RuntimeError("No quantized layers were found for projected QSAM")
        self._basis_ready = True
        self._initialization_report = report

    @torch.no_grad()
    def _project_weight_grad(self, module):
        grad = module.x.grad
        if grad is None:
            return None
        basis = self.bases[module]
        grad_matrix = self._as_matrix(grad)
        projected = basis.matmul(basis.transpose(0, 1).matmul(grad_matrix))
        return projected.reshape_as(grad)

    @staticmethod
    def _append_norm(norms, grad):
        if grad is not None:
            norms.append(torch.norm(grad, p=2))

    @torch.no_grad()
    def _grad_norm_and_projection_ratio(self):
        norms = []
        raw_weight_norm_sq = None
        projected_weight_norm_sq = None

        for _, module in self.model.named_modules():
            if self._is_quant_layer(module):
                projected_grad = self._project_weight_grad(module)
                if projected_grad is not None:
                    self._append_norm(norms, projected_grad)
                    raw_norm_sq = module.x.grad.square().sum()
                    projected_norm_sq = projected_grad.square().sum()
                    raw_weight_norm_sq = (
                        raw_norm_sq
                        if raw_weight_norm_sq is None
                        else raw_weight_norm_sq + raw_norm_sq
                    )
                    projected_weight_norm_sq = (
                        projected_norm_sq
                        if projected_weight_norm_sq is None
                        else projected_weight_norm_sq + projected_norm_sq
                    )

                if self.include_wclip:
                    self._append_norm(norms, module.weight_clip_value.grad)
                if self.include_aclip:
                    self._append_norm(norms, module.activation_clip_value.grad)
                if module.bias is not None:
                    self._append_norm(norms, module.bias.grad)
            elif self.include_bn and isinstance(module, nn.BatchNorm2d):
                self._append_norm(norms, module.weight.grad)
                self._append_norm(norms, module.bias.grad)

        if not norms:
            raise RuntimeError("Projected QSAM received no gradients")

        if raw_weight_norm_sq is not None and raw_weight_norm_sq.item() > 0:
            self.last_projection_energy_ratio = (
                projected_weight_norm_sq / raw_weight_norm_sq
            ).item()
        else:
            self.last_projection_energy_ratio = None
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def _perturb_parameter(self, parameter, scale):
        if parameter.grad is None:
            return
        self.state[parameter]["old_p"] = parameter.data.clone()
        parameter.add_(parameter.grad * scale.to(parameter))

    @torch.no_grad()
    def ascent_step(self):
        self._initialize_bases()
        grad_norm = self._grad_norm_and_projection_ratio()
        scale = self.rho / (grad_norm + 1e-12)

        for _, module in self.model.named_modules():
            if self._is_quant_layer(module):
                projected_grad = self._project_weight_grad(module)
                module.epsilon = (
                    torch.zeros_like(module.x)
                    if projected_grad is None
                    else projected_grad * scale.to(module.x)
                )
                if self.include_wclip:
                    self._perturb_parameter(module.weight_clip_value, scale)
                if self.include_aclip:
                    self._perturb_parameter(module.activation_clip_value, scale)
                if module.bias is not None:
                    self._perturb_parameter(module.bias, scale)
            elif self.include_bn and isinstance(module, nn.BatchNorm2d):
                self._perturb_parameter(module.weight, scale)
                self._perturb_parameter(module.bias, scale)

        self.optimizer.zero_grad()

    @torch.no_grad()
    def _restore_parameter(self, parameter):
        old_p = self.state[parameter].pop("old_p", None)
        if old_p is not None:
            parameter.data.copy_(old_p)

    @torch.no_grad()
    def descent_step(self):
        for _, module in self.model.named_modules():
            if self._is_quant_layer(module):
                if self.include_wclip:
                    self._restore_parameter(module.weight_clip_value)
                if self.include_aclip:
                    self._restore_parameter(module.activation_clip_value)
                if module.bias is not None:
                    self._restore_parameter(module.bias)
            elif self.include_bn and isinstance(module, nn.BatchNorm2d):
                self._restore_parameter(module.weight)
                self._restore_parameter(module.bias)

        self.optimizer.step()
        self.optimizer.zero_grad()
        for _, module in self.model.named_modules():
            if self._is_quant_layer(module):
                module.epsilon = None

    @torch.no_grad()
    def restore_step(self):
        """Restore a partially completed ascent step without an optimizer update."""
        for _, module in self.model.named_modules():
            if self._is_quant_layer(module):
                if self.include_wclip:
                    self._restore_parameter(module.weight_clip_value)
                if self.include_aclip:
                    self._restore_parameter(module.activation_clip_value)
                if module.bias is not None:
                    self._restore_parameter(module.bias)
                module.epsilon = None
            elif self.include_bn and isinstance(module, nn.BatchNorm2d):
                self._restore_parameter(module.weight)
                self._restore_parameter(module.bias)
        self.optimizer.zero_grad()

    def pop_initialization_report(self):
        report = self._initialization_report
        self._initialization_report = []
        return report


class RandomSubspaceQSAM(_ProjectedQSAM):
    """QSAM with a fixed, seeded random subspace in every quantized layer."""

    def __init__(self, *args, projection_seed=12345, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_seed = projection_seed
        # Generate on CPU so the same seed yields the same bases on supported GPUs.
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(projection_seed)

    @torch.no_grad()
    def _make_basis(self, residual, rank):
        random_matrix = torch.randn(
            (residual.shape[0], rank), generator=self._generator, dtype=torch.float32
        )
        basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
        return basis.to(device=residual.device, dtype=residual.dtype)

    def __repr__(self):
        return "RandomSubspaceQSAM(rank={}, projection_seed={})".format(
            self.rank, self.projection_seed
        )


class QuantizationResidualQSAM(_ProjectedQSAM):
    """QSAM projected onto the initial quantization-residual QR subspace."""

    @torch.no_grad()
    def _make_basis(self, residual, rank):
        basis, _ = torch.linalg.qr(residual, mode="reduced")
        return basis[:, :rank]

    def __repr__(self):
        return "QuantizationResidualQSAM(rank={})".format(self.rank)
