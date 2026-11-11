"""SVD-mode projected variants of two-forward QSAM.

In the class names below, ``QR`` means *quantization residual* rather than a
QR matrix decomposition.  ``SVDQuantizationResidualQSAM`` extracts paired
rank-one matrix modes from the truncated SVD of the initial quantization
residual.  ``SVDRandomSubspaceQSAM`` is its structure-matched random control.

For a flattened residual ``R`` with shape ``[m, n]``, the real-residual modes
are ``D_j = u_j v_j^T``.  The singular values select and diagnose the modes,
but do not scale the SAM perturbation.  Consequently every ``D_j`` has unit
Frobenius norm and the global ``rho`` retains the same norm interpretation as
in ordinary QSAM.
"""

from collections import defaultdict

import torch
import torch.nn as nn

from models.LIQ_wn_qsam import QConv2d, QLinear


class _SVDModeProjectedQSAM:
    """Common two-forward QSAM implementation for paired matrix modes."""

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

        # A mode set is stored as (U_r, V_r), not as r dense weight tensors.
        # Module objects are stable keys even when the model is wrapped in DDP.
        self.modes = {}
        self._modes_ready = False
        self._initialization_report = []
        self.last_projection_energy_ratio = None

    @staticmethod
    def _is_quant_layer(module):
        return isinstance(module, (QConv2d, QLinear))

    @staticmethod
    def _as_matrix(tensor):
        return tensor.reshape(tensor.shape[0], -1)

    def _make_modes(self, residual, rank):
        """Return CPU float32 ``(U_r, V_r)`` matrices for paired modes."""
        raise NotImplementedError

    @staticmethod
    def _project_matrix(matrix, left_modes, right_modes):
        """Orthogonally project ``matrix`` onto span{u_j v_j^T}.

        With orthonormal columns in U and V, the paired outer products are an
        orthonormal Frobenius basis.  Coefficient j is ``u_j^T matrix v_j``.
        """
        coefficients = (left_modes * matrix.matmul(right_modes)).sum(dim=0)
        return (left_modes * coefficients.unsqueeze(0)).matmul(
            right_modes.transpose(0, 1)
        )

    @torch.no_grad()
    def _initialize_modes(self):
        if self._modes_ready:
            return

        report = []
        for name, module in self.model.named_modules():
            if not self._is_quant_layer(module):
                continue

            source = getattr(module, "quantization_source", None)
            quantized = getattr(module, "x", None)
            if source is None or quantized is None:
                raise RuntimeError(
                    "SVD projected QSAM needs quantization_source and x after "
                    "the first clean forward pass; missing values for '{}'".format(
                        name
                    )
                )
            if source.shape != quantized.shape:
                raise RuntimeError(
                    "Quantization source and effective weight have incompatible "
                    "shapes for '{}'".format(name)
                )

            # Construct modes on CPU in float32.  This avoids storing a second
            # full residual on the GPU and makes seeded random modes independent
            # of the CUDA RNG implementation.
            residual = self._as_matrix(
                (source.detach() - quantized.detach()).float()
            ).cpu()
            effective_rank = min(self.rank, min(residual.shape))
            left, right = self._make_modes(residual, effective_rank)

            expected_left_shape = (residual.shape[0], effective_rank)
            expected_right_shape = (residual.shape[1], effective_rank)
            if left.shape != expected_left_shape or right.shape != expected_right_shape:
                raise RuntimeError("Invalid SVD mode shapes for '{}'".format(name))

            target_device = quantized.device
            target_dtype = quantized.dtype
            self.modes[module] = (
                left.to(
                    device=target_device,
                    dtype=target_dtype,
                    non_blocking=True,
                ),
                right.to(
                    device=target_device,
                    dtype=target_dtype,
                    non_blocking=True,
                ),
            )

            # The energy captured by an orthonormal paired-mode basis is the
            # squared norm of its projection, regardless of how modes were made.
            projected_residual = self._project_matrix(residual, left, right)
            residual_norm_sq = residual.square().sum()
            captured_norm_sq = projected_residual.square().sum()
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

        if not self.modes:
            raise RuntimeError("No quantized layers were found for SVD projected QSAM")
        self._modes_ready = True
        self._initialization_report = report

    @torch.no_grad()
    def _project_weight_grad(self, module):
        grad = module.x.grad
        if grad is None:
            return None
        left, right = self.modes[module]
        grad_matrix = self._as_matrix(grad)
        projected = self._project_matrix(grad_matrix, left, right)
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
            raise RuntimeError("SVD projected QSAM received no gradients")

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
        self._initialize_modes()
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
        """Restore a partially completed ascent step without updating weights."""
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


class SVDRandomSubspaceQSAM(_SVDModeProjectedQSAM):
    """QSAM using fixed random paired rank-one matrix modes."""

    def __init__(self, *args, projection_seed=12345, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_seed = projection_seed
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(projection_seed)

    @torch.no_grad()
    def _make_modes(self, residual, rank):
        left_random = torch.randn(
            (residual.shape[0], rank),
            generator=self._generator,
            dtype=torch.float32,
        )
        right_random = torch.randn(
            (residual.shape[1], rank),
            generator=self._generator,
            dtype=torch.float32,
        )
        left, _ = torch.linalg.qr(left_random, mode="reduced")
        right, _ = torch.linalg.qr(right_random, mode="reduced")
        return left, right

    def __repr__(self):
        return "SVDRandomSubspaceQSAM(rank={}, projection_seed={})".format(
            self.rank, self.projection_seed
        )


class SVDQuantizationResidualQSAM(_SVDModeProjectedQSAM):
    """QSAM using the leading SVD modes of the quantization residual."""

    @torch.no_grad()
    def _make_modes(self, residual, rank):
        left, _, right_transpose = torch.linalg.svd(
            residual, full_matrices=False
        )
        return left[:, :rank], right_transpose[:rank, :].transpose(0, 1)

    def __repr__(self):
        return "SVDQuantizationResidualQSAM(rank={})".format(self.rank)
