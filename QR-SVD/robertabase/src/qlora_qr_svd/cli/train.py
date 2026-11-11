from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

import torch

from qlora_qr_svd.config import load_config
from qlora_qr_svd.data.glue import build_glue_data
from qlora_qr_svd.diagnostics import SubspaceComparisonDiagnostics
from qlora_qr_svd.environment import collect_environment
from qlora_qr_svd.models.encoder import attach_lora, load_quantized_encoder
from qlora_qr_svd.qr_svd.perturbation import (
    QRSVDPerturbation,
    capture_rng_state,
    restore_rng_state,
)
from qlora_qr_svd.quantization.residual import load_or_build_residual_bases
from qlora_qr_svd.sam.lora import LoraSAMPerturbation
from qlora_qr_svd.sam.random_subspace import (
    build_random_subspace_bases,
    random_basis_report,
    reuse_reference_singular_values,
)
from qlora_qr_svd.training.engine import train_encoder
from qlora_qr_svd.utils import ensure_dir, seed_everything, write_json


DIAGNOSTIC_RANDOM_BASIS_SEEDS = (12345, 2026, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RoBERTa E2-E5 QLoRA experiment")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--seed", type=int, help="Override the model/training seed")
    parser.add_argument("--data-seed", type=int, help="Override the low-resource sampling seed")
    parser.add_argument(
        "--task-name",
        choices=("rte", "mrpc", "cola"),
        help="Override the GLUE task",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        help="Override the fraction of the GLUE training split",
    )
    parser.add_argument(
        "--local-dataset-dir",
        help="Override the local datasets.save_to_disk directory",
    )
    parser.add_argument(
        "--validation-source",
        choices=("official", "inner_train"),
        help="Use official validation or a held-out split of the selected training subset",
    )
    parser.add_argument(
        "--inner-validation-fraction",
        type=float,
        help="Fraction of the selected training subset reserved for inner-dev",
    )
    parser.add_argument(
        "--inner-validation-seed",
        type=int,
        help="Fixed seed for the stratified inner-dev split",
    )
    parser.add_argument("--output-dir", help="Override the output directory")
    parser.add_argument("--rho", type=float, help="Override the SAM radius")
    parser.add_argument(
        "--random-basis-seed", type=int, help="Override the E4 random basis seed"
    )
    parser.add_argument(
        "--subspace-rank",
        "--svd-rank",
        dest="subspace_rank",
        type=int,
        help="Override the E4/E5 paired-subspace rank",
    )
    parser.add_argument(
        "--singular-value-exponent",
        type=float,
        help="Weight subspace modes by residual singular_value ** exponent",
    )
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs")
    deterministic_group = parser.add_mutually_exclusive_group()
    deterministic_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Require deterministic PyTorch algorithms",
    )
    deterministic_group.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic PyTorch algorithms",
    )
    parser.set_defaults(deterministic=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.training.seed = args.seed
    if args.data_seed is not None:
        config.data.data_seed = args.data_seed
    if args.task_name is not None:
        config.data.task_name = args.task_name
    if args.train_fraction is not None:
        config.data.train_fraction = args.train_fraction
    if args.local_dataset_dir is not None:
        config.data.local_dataset_dir = args.local_dataset_dir
    if args.validation_source is not None:
        config.data.validation_source = args.validation_source
    if args.inner_validation_fraction is not None:
        config.data.inner_validation_fraction = args.inner_validation_fraction
    if args.inner_validation_seed is not None:
        config.data.inner_validation_seed = args.inner_validation_seed
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.rho is not None:
        config.sam.rho = args.rho
    if args.random_basis_seed is not None:
        config.sam.random_basis_seed = args.random_basis_seed
    if args.subspace_rank is not None:
        config.sam.subspace_rank = args.subspace_rank
    if args.singular_value_exponent is not None:
        config.sam.singular_value_exponent = args.singular_value_exponent
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.deterministic is not None:
        config.runtime.deterministic = args.deterministic
    config.validate()
    if platform.system() != "Linux":
        raise RuntimeError("Training is intentionally configured for the remote Linux server")
    seed_everything(config.training.seed, config.runtime.deterministic)
    environment = collect_environment(strict=True)
    environment["determinism"] = {
        "enabled": config.runtime.deterministic,
        "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32

    output_dir = ensure_dir(config.training.output_dir)
    write_json(output_dir / "resolved_config.json", config.to_dict())
    write_json(output_dir / "environment.json", environment)

    quantized_model, tokenizer = load_quantized_encoder(config)
    write_json(
        output_dir / "model_identity.json",
        {
            "name_or_path": config.model.name_or_path,
            "requested_revision": config.model.revision,
            "resolved_revision": getattr(quantized_model.config, "_commit_hash", None),
        },
    )
    perturbation = None
    comparison_diagnostics = None
    if config.experiment.method == "random_subspace":
        bases = build_random_subspace_bases(
            quantized_model=quantized_model,
            target_modules=config.lora.target_modules,
            rank=config.sam.subspace_rank,
            seed=config.sam.random_basis_seed,
        )
        spectrum_source = "unit"
        spectrum_cache_path = None
        spectrum_cache_created = None
        if config.sam.reuse_qr_singular_values:
            initialization_rng_state = capture_rng_state()
            try:
                reference_bases, cache_path, cache_created = (
                    load_or_build_residual_bases(quantized_model, config)
                )
            finally:
                restore_rng_state(initialization_rng_state)
            bases = reuse_reference_singular_values(bases, reference_bases)
            spectrum_source = "qr_svd_quantization_residual"
            spectrum_cache_path = str(Path(cache_path).resolve())
            spectrum_cache_created = cache_created
        perturbation = QRSVDPerturbation(
            model=quantized_model,
            bases=bases,
            rho=config.sam.rho,
            singular_value_exponent=config.sam.singular_value_exponent,
        )
        write_json(
            output_dir / "random_basis.json",
            random_basis_report(
                bases,
                config.sam.random_basis_seed,
                singular_value_exponent=config.sam.singular_value_exponent,
                spectrum_source=spectrum_source,
            )
            | {
                "spectrum_cache_path": spectrum_cache_path,
                "spectrum_cache_created": spectrum_cache_created,
            },
        )
    elif config.experiment.method == "qr_svd":
        # Loading the full-precision reference initializes an unused random
        # classifier head. Restore RNG afterwards so baseline and QR-SVD start
        # from identical LoRA/classifier initialization for the same seed.
        initialization_rng_state = capture_rng_state()
        try:
            bases, cache_path, cache_created = load_or_build_residual_bases(
                quantized_model, config
            )
        finally:
            restore_rng_state(initialization_rng_state)
        perturbation = QRSVDPerturbation(
            model=quantized_model,
            bases=bases,
            rho=config.sam.rho,
            singular_value_exponent=config.sam.singular_value_exponent,
        )
        if config.qr_svd.collect_subspace_comparisons:
            comparison_bases = {
                f"random_seed_{seed}": reuse_reference_singular_values(
                    build_random_subspace_bases(
                        quantized_model=quantized_model,
                        target_modules=config.lora.target_modules,
                        rank=config.sam.subspace_rank,
                        seed=seed,
                    ),
                    bases,
                )
                for seed in DIAGNOSTIC_RANDOM_BASIS_SEEDS
            }
            comparison_diagnostics = SubspaceComparisonDiagnostics(
                model=quantized_model,
                named_bases=comparison_bases,
                singular_value_exponent=config.sam.singular_value_exponent,
            )
        write_json(
            output_dir / "qr_svd_basis.json",
            {
                "cache_path": str(Path(cache_path).resolve()),
                "cache_created": cache_created,
                "singular_value_exponent": config.sam.singular_value_exponent,
                "layers": perturbation.basis_report(),
            },
        )
        if comparison_diagnostics is not None:
            write_json(
                output_dir / "subspace_comparison.json",
                comparison_diagnostics.report()
                | {
                    "primary": "qr_svd",
                    "random_basis_seeds": list(DIAGNOSTIC_RANDOM_BASIS_SEEDS),
                    "random_spectrum_source": "qr_svd_quantization_residual",
                    "singular_value_exponent": config.sam.singular_value_exponent,
                    "collection_interval_steps": config.training.logging_steps,
                    "changes_training_hyperparameters": False,
                },
            )

    bundle = attach_lora(quantized_model, tokenizer, config)
    if config.experiment.method == "lora_sam":
        perturbation = LoraSAMPerturbation(bundle.model, rho=config.sam.rho)
        write_json(output_dir / "lora_sam.json", perturbation.report())
    write_json(output_dir / "parameter_summary.json", bundle.parameter_summary)
    data = build_glue_data(bundle.tokenizer, config.data, config.training)
    write_json(
        output_dir / "data_summary.json",
        {
            "task": config.data.task_name,
            "dataset_name": config.data.dataset_name,
            "dataset_revision": config.data.dataset_revision,
            "local_dataset_dir": config.data.local_dataset_dir,
            "train_fraction": config.data.train_fraction,
            "validation_source": data.validation_source,
            "inner_validation_fraction": config.data.inner_validation_fraction,
            "inner_validation_seed": config.data.inner_validation_seed,
            "official_validation_used": data.validation_source == "official",
            "selected_training_pool_examples": len(data.selected_indices),
            "train_examples": len(data.train_dataset),
            "validation_examples": len(data.validation_dataset),
            "indices_path": str(data.indices_path.resolve()),
            "inner_split_path": (
                str(data.inner_split_path.resolve()) if data.inner_split_path else None
            ),
            "train_indices": data.train_indices,
            "validation_indices": data.validation_indices,
        },
    )

    try:
        summary = train_encoder(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            train_loader=data.train_loader,
            validation_loader=data.validation_loader,
            config=config,
            perturbation=perturbation,
            comparison_diagnostics=comparison_diagnostics,
        )
    finally:
        if comparison_diagnostics is not None:
            comparison_diagnostics.close()
        if perturbation is not None:
            perturbation.close()
    print(summary)


if __name__ == "__main__":
    main()
