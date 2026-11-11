import pytest

from scripts.aggregate_stage3 import basis_sensitivity, paired_comparison


def _record(method: str, rank: int, seed: int, score: float, basis_seed=None):
    return {
        "method": method,
        "rank": rank,
        "train_seed": seed,
        "basis_seed": basis_seed,
        "best_score": score,
    }


def test_stage3_pairing_averages_e4_bases_within_training_seed():
    records = [
        _record("random_subspace", 16, 17, 0.60, 12345),
        _record("random_subspace", 16, 17, 0.62, 2026),
        _record("qr_svd", 16, 17, 0.63),
        _record("random_subspace", 16, 29, 0.55, 12345),
        _record("random_subspace", 16, 29, 0.57, 2026),
        _record("qr_svd", 16, 29, 0.58),
    ]

    result = paired_comparison(records, 16)

    assert result["seeds"] == [17, 29]
    assert result["e5_minus_e4"]["values"] == pytest.approx([0.02, 0.02])


def test_stage3_basis_sensitivity_uses_within_training_seed_ranges():
    records = [
        _record("random_subspace", 8, 17, 0.60, 12345),
        _record("random_subspace", 8, 17, 0.63, 2026),
        _record("random_subspace", 8, 29, 0.55, 12345),
        _record("random_subspace", 8, 29, 0.56, 2026),
    ]

    result = basis_sensitivity(records, 8)

    assert result["values"] == pytest.approx([0.03, 0.01])
    assert result["mean"] == pytest.approx(0.02)
