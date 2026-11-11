import pytest

from scripts.aggregate_stage2 import grouped_summaries, paired_comparison, stats


def _record(method: str, rho: float, seed: int, score: float, basis_seed=None):
    return {
        "method": method,
        "rho": rho,
        "rank": 8,
        "basis_seed": basis_seed,
        "train_seed": seed,
        "best_score": score,
        "mean_loss_increase": 0.001,
        "mean_projected_gradient_norm": 0.01,
        "seconds_per_step": 0.2,
    }


def test_stage2_pairing_averages_random_bases_within_train_seed():
    records = [
        _record("random_subspace", 0.025, 17, 0.60, 12345),
        _record("random_subspace", 0.025, 17, 0.62, 2026),
        _record("qr_svd", 0.0625, 17, 0.63),
        _record("random_subspace", 0.025, 29, 0.55, 12345),
        _record("random_subspace", 0.025, 29, 0.57, 2026),
        _record("qr_svd", 0.0625, 29, 0.58),
    ]

    result = paired_comparison(records, 0.025, 0.0625)

    assert result["seeds"] == [17, 29]
    assert result["e5_minus_e4"]["values"] == pytest.approx([0.02, 0.02])


def test_stage2_grouping_and_empty_stats_are_json_safe():
    records = [_record("lora_sam", 0.005, 17, 0.61)]

    groups = grouped_summaries(records)

    assert groups[0]["best_score"]["mean"] == 0.61
    assert stats([]) == {"n": 0, "mean": None, "sample_std": None, "values": []}
