import pytest

from scripts.aggregate_stage67 import paired_comparison


def _record(method, seed, score, basis_seed=None):
    return {
        "setting": "rte20",
        "method": method,
        "train_seed": seed,
        "basis_seed": basis_seed,
        "best_score": score,
    }


def test_e5_e4_comparison_averages_random_bases_within_training_seed():
    records = [
        _record("e4", 13, 0.60, 12345),
        _record("e4", 13, 0.62, 2026),
        _record("e4", 13, 0.61, 3407),
        _record("e5", 13, 0.63),
        _record("e4", 21, 0.55, 12345),
        _record("e4", 21, 0.57, 2026),
        _record("e4", 21, 0.56, 3407),
        _record("e5", 21, 0.55),
    ]

    result = paired_comparison(records, "rte20", "e4")

    assert result["values"] == pytest.approx([0.02, -0.01])
    assert result["e4_value_per_seed"] == "mean_across_random_basis_seeds"
    assert result["wins_ties_losses"] == {"wins": 1, "ties": 0, "losses": 1}
