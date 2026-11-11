import pytest

from scripts.aggregate_stage45 import (
    basis_sensitivity,
    paired_direction_comparison,
    paired_weighting_effect,
)


def _record(method, exponent, seed, score, basis_seed=None):
    return {
        "method": method,
        "singular_value_exponent": exponent,
        "train_seed": seed,
        "basis_seed": basis_seed,
        "best_score": score,
    }


def test_direction_comparison_averages_random_bases_within_training_seed():
    records = [
        _record("random_subspace", 0.5, 17, 0.60, 12345),
        _record("random_subspace", 0.5, 17, 0.62, 2026),
        _record("qr_svd", 0.5, 17, 0.63),
        _record("random_subspace", 0.5, 29, 0.55, 12345),
        _record("random_subspace", 0.5, 29, 0.57, 2026),
        _record("qr_svd", 0.5, 29, 0.58),
    ]

    result = paired_direction_comparison(records, 0.5)

    assert result["seeds"] == [17, 29]
    assert result["qr_minus_random"]["values"] == pytest.approx([0.02, 0.02])


def test_weighting_effect_is_paired_against_p0_within_method():
    records = [
        _record("qr_svd", 0.0, 17, 0.60),
        _record("qr_svd", 0.5, 17, 0.63),
        _record("qr_svd", 0.0, 29, 0.55),
        _record("qr_svd", 0.5, 29, 0.56),
    ]

    result = paired_weighting_effect(records, "qr_svd", 0.5)

    assert result["weighted_minus_p0"]["values"] == pytest.approx([0.03, 0.01])


def test_basis_sensitivity_uses_ranges_within_training_seed():
    records = [
        _record("random_subspace", 1.0, 17, 0.60, 12345),
        _record("random_subspace", 1.0, 17, 0.64, 2026),
        _record("random_subspace", 1.0, 29, 0.55, 12345),
        _record("random_subspace", 1.0, 29, 0.56, 2026),
    ]

    result = basis_sensitivity(records, 1.0)

    assert result["values"] == pytest.approx([0.04, 0.01])
