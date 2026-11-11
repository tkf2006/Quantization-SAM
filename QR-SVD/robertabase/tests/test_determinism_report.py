from scripts.verify_determinism import normalized


def test_determinism_normalization_ignores_only_wall_time():
    left = {
        "score": 0.75,
        "elapsed_seconds": 1.0,
        "nested": [{"loss": 0.5, "elapsed_seconds": 2.0}],
    }
    right = {
        "score": 0.75,
        "elapsed_seconds": 99.0,
        "nested": [{"loss": 0.5, "elapsed_seconds": 100.0}],
    }

    assert normalized(left) == normalized(right)
    right["nested"][0]["loss"] = 0.6
    assert normalized(left) != normalized(right)
