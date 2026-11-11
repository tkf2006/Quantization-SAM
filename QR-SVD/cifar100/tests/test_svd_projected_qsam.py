import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.LIQ_wn_qsam import QLinear
from utils.optim import get_minimizer
from utils.svd_projected_qsam import (
    SVDQuantizationResidualQSAM,
    SVDRandomSubspaceQSAM,
    _SVDModeProjectedQSAM,
)


class SVDModeProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.residual = torch.randn(7, 5)

    def test_svd_modes_are_orthonormal_and_capture_top_energy(self):
        builder = object.__new__(SVDQuantizationResidualQSAM)
        left, right = builder._make_modes(self.residual, rank=3)

        identity = torch.eye(3)
        self.assertTrue(torch.allclose(left.T @ left, identity, atol=1e-6))
        self.assertTrue(torch.allclose(right.T @ right, identity, atol=1e-6))

        projected = _SVDModeProjectedQSAM._project_matrix(
            self.residual, left, right
        )
        singular_values = torch.linalg.svdvals(self.residual)
        expected_energy = singular_values[:3].square().sum()
        self.assertTrue(
            torch.allclose(projected.square().sum(), expected_energy, atol=1e-5)
        )

    def test_projection_is_idempotent_and_nonexpansive(self):
        builder = object.__new__(SVDQuantizationResidualQSAM)
        left, right = builder._make_modes(self.residual, rank=4)
        matrix = torch.randn_like(self.residual)

        projected = _SVDModeProjectedQSAM._project_matrix(matrix, left, right)
        projected_twice = _SVDModeProjectedQSAM._project_matrix(
            projected, left, right
        )
        self.assertTrue(torch.allclose(projected, projected_twice, atol=1e-6))
        self.assertLessEqual(projected.norm().item(), matrix.norm().item() + 1e-6)

    def test_random_control_has_matching_orthonormal_rank_one_modes(self):
        builder = object.__new__(SVDRandomSubspaceQSAM)
        builder._generator = torch.Generator(device="cpu")
        builder._generator.manual_seed(12345)
        left, right = builder._make_modes(self.residual, rank=3)

        identity = torch.eye(3)
        self.assertEqual(left.shape, (7, 3))
        self.assertEqual(right.shape, (5, 3))
        self.assertTrue(torch.allclose(left.T @ left, identity, atol=1e-6))
        self.assertTrue(torch.allclose(right.T @ right, identity, atol=1e-6))

        # Paired outer products must be mutually orthonormal in Frobenius space.
        dense_modes = torch.stack(
            [
                left[:, index : index + 1] @ right[:, index : index + 1].T
                for index in range(3)
            ]
        ).reshape(3, -1)
        self.assertTrue(
            torch.allclose(dense_modes @ dense_modes.T, identity, atol=1e-6)
        )

    def test_random_modes_are_reproducible_for_a_fixed_seed(self):
        first = object.__new__(SVDRandomSubspaceQSAM)
        first._generator = torch.Generator(device="cpu")
        first._generator.manual_seed(99)
        second = object.__new__(SVDRandomSubspaceQSAM)
        second._generator = torch.Generator(device="cpu")
        second._generator.manual_seed(99)

        first_left, first_right = first._make_modes(self.residual, rank=3)
        second_left, second_right = second._make_modes(self.residual, rank=3)
        self.assertTrue(torch.equal(first_left, second_left))
        self.assertTrue(torch.equal(first_right, second_right))

    def test_optimizer_names_route_to_svd_implementations(self):
        cases = (
            ("SVD_QR_QSAM_SGD", SVDQuantizationResidualQSAM),
            ("SVD_RS_QSAM_SGD", SVDRandomSubspaceQSAM),
        )
        for optimizer_name, expected_type in cases:
            model = nn.Sequential(
                QLinear(3, 2, bits_weights=4, bits_activations=4)
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            args = SimpleNamespace(
                opt_type=optimizer_name,
                rho=0.4,
                projection_rank=1,
                projection_seed=12345,
                include_wclip=False,
                include_aclip=False,
                include_bn=False,
                eta=0.01,
            )
            minimizer = get_minimizer(model, optimizer, args)
            self.assertIs(type(minimizer), expected_type)

    def test_two_forward_update_has_the_requested_weight_radius(self):
        cases = (SVDQuantizationResidualQSAM, SVDRandomSubspaceQSAM)
        for minimizer_type in cases:
            torch.manual_seed(11)
            layer = QLinear(
                5,
                4,
                bias=False,
                bits_weights=4,
                bits_activations=4,
            )
            model = nn.Sequential(layer)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            kwargs = {
                "rho": 0.4,
                "rank": 3,
                "include_wclip": False,
                "include_aclip": False,
                "include_bn": False,
            }
            if minimizer_type is SVDRandomSubspaceQSAM:
                kwargs["projection_seed"] = 12345
            minimizer = minimizer_type(optimizer, model, **kwargs)

            inputs = torch.randn(8, 5)
            target = torch.randn(8, 4)
            clean_loss = (model(inputs) - target).square().mean()
            clean_loss.backward()
            minimizer.ascent_step()

            self.assertAlmostEqual(layer.epsilon.norm().item(), 0.4, places=5)
            layer.set_second_forward()
            perturbed_loss = (model(inputs) - target).square().mean()
            perturbed_loss.backward()
            minimizer.descent_step()
            layer.set_first_forward()
            self.assertIsNone(layer.epsilon)


if __name__ == "__main__":
    unittest.main()
