from qlora_qr_svd.data.glue import (
    stratified_holdout_indices,
    stratified_prefix_indices,
)


def test_stratified_subsets_are_reproducible_balanced_and_nested():
    labels = [0] * 80 + [1] * 20
    five_percent = stratified_prefix_indices(labels, 0.05, seed=17)
    twenty_percent = stratified_prefix_indices(labels, 0.20, seed=17)
    repeated = stratified_prefix_indices(labels, 0.20, seed=17)

    assert twenty_percent == repeated
    assert set(five_percent).issubset(twenty_percent)
    assert sum(labels[index] == 0 for index in twenty_percent) == 16
    assert sum(labels[index] == 1 for index in twenty_percent) == 4


def test_inner_holdout_is_reproducible_stratified_and_disjoint():
    labels = [0] * 80 + [1] * 20
    train, validation = stratified_holdout_indices(labels, 0.2, seed=314159)
    repeated_train, repeated_validation = stratified_holdout_indices(
        labels, 0.2, seed=314159
    )

    assert train == repeated_train
    assert validation == repeated_validation
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(len(labels)))
    assert len(validation) == 20
    assert sum(labels[index] == 0 for index in validation) == 16
    assert sum(labels[index] == 1 for index in validation) == 4

