from pathlib import Path

from qlora_qr_svd.config import load_config


def test_e2_to_e5_configs_share_control_variables():
    root = Path(__file__).resolve().parents[1]
    configs = [
        load_config(root / "configs/roberta_rte_20pct_qlora.yaml"),
        load_config(root / "configs/roberta_rte_20pct_lora_sam.yaml"),
        load_config(root / "configs/roberta_rte_20pct_random_subspace.yaml"),
        load_config(root / "configs/roberta_rte_20pct_qr_svd.yaml"),
    ]

    assert [config.experiment.method for config in configs] == [
        "qlora",
        "lora_sam",
        "random_subspace",
        "qr_svd",
    ]
    baseline = configs[0]
    for config in configs[1:]:
        assert baseline.model == config.model
        assert baseline.data == config.data
        assert baseline.lora == config.lora
        assert baseline.runtime == config.runtime
        baseline_training = baseline.training.__dict__.copy()
        method_training = config.training.__dict__.copy()
        baseline_training.pop("output_dir")
        method_training.pop("output_dir")
        assert baseline_training == method_training
        assert baseline.sam.rho == config.sam.rho
        assert baseline.sam.subspace_rank == config.sam.subspace_rank
        assert baseline.sam.random_basis_seed == config.sam.random_basis_seed

    assert baseline.data.local_dataset_dir == "artifacts/datasets/glue_rte_bcdcba7"
    assert [config.sam.enabled for config in configs] == [False, True, True, True]
    assert [config.qr_svd.enabled for config in configs] == [False, False, False, True]
    assert all(config.data.validation_source == "official" for config in configs)


def test_stage2_configs_use_inner_dev_and_deterministic_runtime():
    root = Path(__file__).resolve().parents[1]
    paths = [
        "stage2_roberta_rte_20pct_lora_sam.yaml",
        "stage2_roberta_rte_20pct_random_subspace.yaml",
        "stage2_roberta_rte_20pct_qr_svd.yaml",
    ]
    configs = [load_config(root / "configs" / path) for path in paths]

    assert [config.experiment.method for config in configs] == [
        "lora_sam",
        "random_subspace",
        "qr_svd",
    ]
    assert all(config.data.validation_source == "inner_train" for config in configs)
    assert all(config.data.inner_validation_fraction == 0.2 for config in configs)
    assert all(config.data.inner_validation_seed == 314159 for config in configs)
    assert all(config.runtime.deterministic for config in configs)


def test_stage45_configs_fix_rank_rho_and_share_qr_spectrum():
    root = Path(__file__).resolve().parents[1]
    random_config = load_config(
        root / "configs/stage45_roberta_rte_20pct_random_spectrum.yaml"
    )
    qr_config = load_config(root / "configs/stage45_roberta_rte_20pct_qr_svd.yaml")

    assert random_config.sam.rho == qr_config.sam.rho == 0.05
    assert random_config.sam.subspace_rank == qr_config.sam.subspace_rank == 8
    assert random_config.sam.singular_value_exponent == 0.0
    assert qr_config.sam.singular_value_exponent == 0.0
    assert random_config.sam.reuse_qr_singular_values is True
    assert qr_config.sam.reuse_qr_singular_values is False
    assert random_config.data.validation_source == qr_config.data.validation_source == "inner_train"
    assert random_config.runtime.deterministic and qr_config.runtime.deterministic


def test_stage67_configs_fix_selected_hyperparameters_and_compact_outputs():
    root = Path(__file__).resolve().parents[1]
    paths = [
        "stage67_formal_qlora.yaml",
        "stage67_formal_lora_sam.yaml",
        "stage67_formal_random_subspace.yaml",
        "stage67_formal_qr_svd.yaml",
    ]
    configs = [load_config(root / "configs" / path) for path in paths]

    assert [config.experiment.method for config in configs] == [
        "qlora",
        "lora_sam",
        "random_subspace",
        "qr_svd",
    ]
    assert configs[1].sam.rho == 0.01
    assert all(config.sam.rho == 0.05 for config in configs[2:])
    assert all(config.sam.subspace_rank == 8 for config in configs[2:])
    assert all(config.sam.singular_value_exponent == 0.0 for config in configs[2:])
    assert all(config.data.validation_source == "official" for config in configs)
    assert all(config.runtime.deterministic for config in configs)
    assert all(not config.qr_svd.collect_subspace_comparisons for config in configs)
    assert all(not config.training.save_last_adapter for config in configs)
    assert all(not config.training.save_optimizer_state for config in configs)
    assert all(not config.training.save_tokenizer_with_adapter for config in configs)
