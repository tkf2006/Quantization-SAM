from pathlib import Path

import pytest
import torch
import torch.nn as nn

from qlora_qr_svd.diagnostics import SubspaceComparisonDiagnostics
from qlora_qr_svd.qr_svd.basis import (
    compute_svd_basis,
    cumulative_energy_by_rank,
    load_basis_cache,
    project_onto_paired_modes,
    save_basis_cache,
)
from qlora_qr_svd.qr_svd.perturbation import QRSVDPerturbation
from qlora_qr_svd.sam.random_subspace import (
    compute_random_paired_basis,
    reuse_reference_singular_values,
)


def test_svd_basis_is_orthonormal_and_projection_is_idempotent():
    torch.manual_seed(7)
    residual = torch.randn(7, 5)
    basis = compute_svd_basis(residual, rank=3)
    identity = torch.eye(3)
    assert torch.allclose(basis.left.T @ basis.left, identity, atol=1e-6)
    assert torch.allclose(basis.right.T @ basis.right, identity, atol=1e-6)

    matrix = torch.randn_like(residual)
    projected = project_onto_paired_modes(matrix, basis.left, basis.right)
    projected_twice = project_onto_paired_modes(projected, basis.left, basis.right)
    assert torch.allclose(projected, projected_twice, atol=1e-6)
    assert projected.norm() <= matrix.norm() + 1e-6
    assert basis.full_singular_values is not None
    energy = cumulative_energy_by_rank(basis.full_singular_values, ranks=(1, 3, 9))
    assert 0 < energy["1"] <= energy["3"] <= energy["9"] == 1.0


def test_basis_cache_rejects_identity_mismatch(tmp_path: Path):
    basis = compute_svd_basis(torch.randn(4, 3), rank=2)
    cache = tmp_path / "basis.pt"
    save_basis_cache(cache, {"model": "a", "rank": 2}, {"layer": basis})
    loaded = load_basis_cache(cache, {"model": "a", "rank": 2})
    assert loaded["layer"].left.shape == (4, 2)
    assert torch.equal(
        loaded["layer"].full_singular_values, basis.full_singular_values
    )

    try:
        load_basis_cache(cache, {"model": "b", "rank": 2})
    except RuntimeError as error:
        assert "metadata mismatch" in str(error)
    else:
        raise AssertionError("Mismatched cache metadata was accepted")


def test_hook_builds_global_radius_perturbation_without_changing_weights():
    torch.manual_seed(11)
    model = nn.Sequential()
    model.add_module("layer", nn.Linear(5, 4, bias=False))
    original_weight = model.layer.weight.detach().clone()
    basis = compute_svd_basis(torch.randn(4, 5), rank=3)
    manager = QRSVDPerturbation(model, {"layer": basis}, rho=0.4)

    inputs = torch.randn(8, 5)
    target = torch.randn(8, 4)
    manager.begin_clean_pass()
    clean_output = model(inputs)
    coefficient_gradients = torch.autograd.grad(
        (clean_output - target).square().mean(), manager.clean_coefficients()
    )
    assert model.layer.weight.grad is None
    diagnostics = manager.begin_perturbed_pass(coefficient_gradients)
    perturbed_output = model(inputs)

    assert abs(diagnostics["perturbation_norm"] - 0.4) < 1e-5
    assert abs(diagnostics["gradient_energy_share_sum"] - 1.0) < 1e-6
    assert abs(diagnostics["perturbation_energy_share_sum"] - 1.0) < 1e-6
    assert diagnostics["gradient_energy_consistency_error"] < 1e-8
    assert diagnostics["perturbation_energy_consistency_error"] < 1e-8
    assert not torch.allclose(clean_output, perturbed_output)
    assert torch.equal(model.layer.weight, original_weight)

    manager.disable()
    restored_output = model(inputs)
    assert torch.allclose(clean_output, restored_output, atol=1e-6)
    manager.close()


def test_same_clean_batch_comparison_is_read_only():
    torch.manual_seed(17)
    model = nn.Sequential()
    model.add_module("layer", nn.Linear(5, 4, bias=False))
    primary_basis = compute_svd_basis(torch.randn(4, 5), rank=3)
    comparison_basis = compute_svd_basis(torch.randn(4, 5), rank=3)
    primary = QRSVDPerturbation(model, {"layer": primary_basis}, rho=0.2)
    comparison = SubspaceComparisonDiagnostics(
        model, {"alternate": {"layer": comparison_basis}}
    )

    inputs = torch.randn(6, 5)
    target = torch.randn(6, 4)
    primary.begin_clean_pass()
    comparison.begin_clean_pass()
    clean_output = model(inputs)
    primary_tensors = primary.clean_tensors()
    comparison_tensors = comparison.clean_tensors()
    gradients = torch.autograd.grad(
        (clean_output - target).square().mean(),
        primary_tensors + comparison_tensors,
    )
    primary.begin_perturbed_pass(gradients[: len(primary_tensors)])
    comparison_result = comparison.collect(gradients[len(primary_tensors) :])

    assert comparison_result["alternate"]["projected_gradient_norm"] > 0
    assert comparison.managers["alternate"].mode == "disabled"
    primary.disable()
    assert torch.allclose(model(inputs), clean_output, atol=1e-6)
    comparison.close()
    primary.close()


def test_singular_value_weighting_preserves_radius_and_weights_direction():
    torch.manual_seed(31)
    model = nn.Sequential()
    model.add_module("layer", nn.Linear(5, 4, bias=False))
    basis = compute_svd_basis(torch.randn(4, 5), rank=3)
    manager = QRSVDPerturbation(
        model,
        {"layer": basis},
        rho=0.3,
        singular_value_exponent=0.5,
    )

    manager.begin_clean_pass()
    gradient = torch.tensor([1.0, -2.0, 0.5])
    diagnostics = manager.begin_perturbed_pass((gradient,))
    expected_direction = gradient * basis.singular_values.sqrt()
    expected = 0.3 * expected_direction / expected_direction.norm()

    assert torch.allclose(manager.layers[0].coefficient, expected, atol=1e-6)
    assert diagnostics["weighted_gradient_norm"] == pytest.approx(
        float(expected_direction.norm())
    )
    assert diagnostics["projected_gradient_norm"] == pytest.approx(
        float(gradient.norm())
    )
    assert diagnostics["perturbation_norm"] == pytest.approx(0.3, abs=1e-6)
    manager.close()


def test_random_directions_can_reuse_exact_qr_singular_values():
    torch.manual_seed(37)
    reference = compute_svd_basis(torch.randn(6, 5), rank=3)
    random = compute_random_paired_basis(
        6, 5, rank=3, generator=torch.Generator().manual_seed(12345)
    )
    reused = reuse_reference_singular_values(
        {"layer": random}, {"layer": reference}
    )["layer"]

    assert reused.source == "random_qr_spectrum"
    assert torch.equal(reused.singular_values, reference.singular_values)
    assert torch.equal(reused.left, random.left)
    assert torch.equal(reused.right, random.right)
    assert not torch.equal(reused.left, reference.left)
