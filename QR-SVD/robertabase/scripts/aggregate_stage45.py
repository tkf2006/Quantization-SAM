from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_EXPONENTS = (0.0, 0.5, 1.0)
EXPECTED_TRAIN_SEEDS = (17, 29, 43)
EXPECTED_BASIS_SEEDS = (12345, 2026, 3407)
EXPECTED_RHO = 0.05
EXPECTED_RANK = 8


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sample_std": None, "values": []}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


def _layer_spectrum(report_path: Path) -> dict[str, list[float]]:
    report = read_json(report_path)
    return {
        layer["name"]: [float(value) for value in layer["selected_singular_values"]]
        for layer in report["layers"]
    }


def collect_runs(root: Path) -> list[dict[str, Any]]:
    records = []
    for summary_path in sorted((root / "runs").glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        config = read_json(run_dir / "resolved_config.json")
        data = read_json(run_dir / "data_summary.json")
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        train_steps = [item for item in metrics if item.get("type") == "train_step"]
        loss_increases = [
            float(item["loss"] - item["clean_loss"])
            for item in train_steps
            if "clean_loss" in item
        ]
        projected_norms = [
            float(item["projected_gradient_norm"])
            for item in train_steps
            if "projected_gradient_norm" in item
        ]
        weighted_norms = [
            float(item["weighted_gradient_norm"])
            for item in train_steps
            if "weighted_gradient_norm" in item
        ]
        perturbation_norms = [
            float(item["perturbation_norm"])
            for item in train_steps
            if "perturbation_norm" in item
        ]
        method = str(summary["method"])
        basis_report_path = (
            run_dir / "random_basis.json"
            if method == "random_subspace"
            else run_dir / "qr_svd_basis.json"
        )
        basis_report = read_json(basis_report_path)
        records.append(
            {
                "path": str(run_dir),
                "method": method,
                "rho": float(config["sam"]["rho"]),
                "rank": int(config["sam"]["subspace_rank"]),
                "singular_value_exponent": float(
                    config["sam"]["singular_value_exponent"]
                ),
                "reuse_qr_singular_values": bool(
                    config["sam"]["reuse_qr_singular_values"]
                ),
                "spectrum_source": basis_report.get(
                    "spectrum_source", "qr_svd_quantization_residual"
                ),
                "basis_seed": (
                    int(config["sam"]["random_basis_seed"])
                    if method == "random_subspace"
                    else None
                ),
                "train_seed": int(config["training"]["seed"]),
                "data_seed": int(config["data"]["data_seed"]),
                "validation_source": str(config["data"]["validation_source"]),
                "official_validation_used": bool(data["official_validation_used"]),
                "deterministic": bool(config["runtime"]["deterministic"]),
                "best_score": float(summary["best_score"]),
                "best_epoch": int(summary["best_epoch"]),
                "epochs_completed": int(summary["epochs_completed"]),
                "global_steps": int(summary["global_steps"]),
                "seconds_per_step": float(summary["elapsed_seconds"])
                / int(summary["global_steps"]),
                "mean_loss_increase": float(np.mean(loss_increases)),
                "positive_loss_increase_fraction": float(
                    np.mean(np.asarray(loss_increases) > 0)
                ),
                "mean_projected_gradient_norm": float(np.mean(projected_norms)),
                "mean_weighted_gradient_norm": float(np.mean(weighted_norms)),
                "mean_perturbation_norm": float(np.mean(perturbation_norms)),
                "max_perturbation_radius_error": float(
                    max(
                        (abs(value - EXPECTED_RHO) for value in perturbation_norms),
                        default=0.0,
                    )
                ),
                "spectrum": _layer_spectrum(basis_report_path),
            }
        )
    validate_matrix(records)
    return records


def validate_matrix(records: list[dict[str, Any]]) -> None:
    expected_runs = len(EXPECTED_EXPONENTS) * len(EXPECTED_TRAIN_SEEDS) * (
        len(EXPECTED_BASIS_SEEDS) + 1
    )
    if len(records) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} Stage 4/5 runs, found {len(records)}")
    for record in records:
        if record["method"] not in {"random_subspace", "qr_svd"}:
            raise RuntimeError(f"Unexpected method in {record['path']}")
        if record["rho"] != EXPECTED_RHO or record["rank"] != EXPECTED_RANK:
            raise RuntimeError(f"Unexpected rho/rank in {record['path']}")
        if record["singular_value_exponent"] not in EXPECTED_EXPONENTS:
            raise RuntimeError(f"Unexpected exponent in {record['path']}")
        if record["train_seed"] not in EXPECTED_TRAIN_SEEDS:
            raise RuntimeError(f"Unexpected training seed in {record['path']}")
        if record["data_seed"] != record["train_seed"]:
            raise RuntimeError(f"Train/data seed mismatch in {record['path']}")
        if record["validation_source"] != "inner_train" or record[
            "official_validation_used"
        ]:
            raise RuntimeError(f"Official validation leaked into {record['path']}")
        if not record["deterministic"]:
            raise RuntimeError(f"Non-deterministic run: {record['path']}")
        if not np.isfinite(record["mean_weighted_gradient_norm"]):
            raise RuntimeError(f"Missing weighted-gradient diagnostics in {record['path']}")
        if record["max_perturbation_radius_error"] > 1e-5:
            raise RuntimeError(
                f"Perturbation radius mismatch in {record['path']}: "
                f"{record['max_perturbation_radius_error']}"
            )
        if record["method"] == "random_subspace":
            if not record["reuse_qr_singular_values"]:
                raise RuntimeError(f"Random run did not reuse QR spectrum: {record['path']}")
            if record["spectrum_source"] != "qr_svd_quantization_residual":
                raise RuntimeError(f"Unexpected random spectrum source: {record['path']}")

    for exponent in EXPECTED_EXPONENTS:
        random_matrix = {
            (record["train_seed"], record["basis_seed"])
            for record in records
            if record["method"] == "random_subspace"
            and record["singular_value_exponent"] == exponent
        }
        expected_random = {
            (seed, basis_seed)
            for seed in EXPECTED_TRAIN_SEEDS
            for basis_seed in EXPECTED_BASIS_SEEDS
        }
        qr_seeds = {
            record["train_seed"]
            for record in records
            if record["method"] == "qr_svd"
            and record["singular_value_exponent"] == exponent
        }
        if random_matrix != expected_random:
            raise RuntimeError(
                f"Incomplete random matrix at p={exponent}: {sorted(random_matrix)}"
            )
        if qr_seeds != set(EXPECTED_TRAIN_SEEDS):
            raise RuntimeError(f"Incomplete QR matrix at p={exponent}: {sorted(qr_seeds)}")


def spectrum_verification(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference = next(record["spectrum"] for record in records if record["method"] == "qr_svd")
    max_error = 0.0
    compared_values = 0
    for record in records:
        if set(record["spectrum"]) != set(reference):
            raise RuntimeError(f"Spectrum layer mismatch in {record['path']}")
        for name, expected in reference.items():
            actual = record["spectrum"][name]
            if len(actual) != len(expected):
                raise RuntimeError(f"Spectrum rank mismatch for {name} in {record['path']}")
            compared_values += len(expected)
            max_error = max(
                max_error,
                max((abs(a - b) for a, b in zip(actual, expected)), default=0.0),
            )
    if max_error != 0.0:
        raise RuntimeError(f"Random and QR spectra differ; max error={max_error}")
    return {
        "all_runs_share_identical_selected_spectrum": True,
        "layers": len(reference),
        "compared_values": compared_values,
        "max_absolute_error": max_error,
    }


def per_seed_values(
    records: list[dict[str, Any]], method: str, exponent: float, metric: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if (
            record["method"] == method
            and record["singular_value_exponent"] == exponent
        ):
            grouped[record["train_seed"]].append(float(record[metric]))
    return {seed: float(np.mean(values)) for seed, values in sorted(grouped.items())}


def method_summary(
    records: list[dict[str, Any]], method: str, exponent: float
) -> dict[str, Any]:
    metrics = (
        "best_score",
        "mean_loss_increase",
        "positive_loss_increase_fraction",
        "mean_projected_gradient_norm",
        "mean_weighted_gradient_norm",
        "mean_perturbation_norm",
        "seconds_per_step",
    )
    return {
        metric: stats(list(per_seed_values(records, method, exponent, metric).values()))
        for metric in metrics
    }


def basis_sensitivity(records: list[dict[str, Any]], exponent: float) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if (
            record["method"] == "random_subspace"
            and record["singular_value_exponent"] == exponent
        ):
            grouped[record["train_seed"]].append(record["best_score"])
    return stats([max(values) - min(values) for _seed, values in sorted(grouped.items())])


def paired_direction_comparison(
    records: list[dict[str, Any]], exponent: float
) -> dict[str, Any]:
    random_values = per_seed_values(records, "random_subspace", exponent, "best_score")
    qr_values = per_seed_values(records, "qr_svd", exponent, "best_score")
    seeds = sorted(set(random_values) & set(qr_values))
    return {
        "singular_value_exponent": exponent,
        "random_value_per_seed": "mean_across_random_basis_seeds",
        "seeds": seeds,
        "qr_minus_random": stats(
            [qr_values[seed] - random_values[seed] for seed in seeds]
        ),
    }


def paired_weighting_effect(
    records: list[dict[str, Any]], method: str, exponent: float
) -> dict[str, Any]:
    baseline = per_seed_values(records, method, 0.0, "best_score")
    weighted = per_seed_values(records, method, exponent, "best_score")
    seeds = sorted(set(baseline) & set(weighted))
    return {
        "method": method,
        "singular_value_exponent": exponent,
        "seeds": seeds,
        "weighted_minus_p0": stats(
            [weighted[seed] - baseline[seed] for seed in seeds]
        ),
    }


def spectral_weight_summary(
    records: list[dict[str, Any]], exponent: float
) -> dict[str, float]:
    spectrum = next(record["spectrum"] for record in records if record["method"] == "qr_svd")
    singular_values = np.asarray(
        [value for values in spectrum.values() for value in values], dtype=np.float64
    )
    weights = singular_values**exponent
    energy = weights**2
    shares = energy / energy.sum()
    return {
        "singular_value_exponent": exponent,
        "coordinates": int(weights.size),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "effective_coordinates": float(1.0 / np.square(shares).sum()),
        "largest_coordinate_energy_share": float(shares.max()),
    }


def grouped_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["method"],
                record["singular_value_exponent"],
                record["basis_seed"],
            )
        ].append(record)
    output = []
    for (method, exponent, basis_seed), runs in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        output.append(
            {
                "method": method,
                "singular_value_exponent": exponent,
                "basis_seed": basis_seed,
                "best_score": stats([run["best_score"] for run in runs]),
                "mean_loss_increase": stats(
                    [run["mean_loss_increase"] for run in runs]
                ),
                "mean_weighted_gradient_norm": stats(
                    [run["mean_weighted_gradient_norm"] for run in runs]
                ),
                "seconds_per_step": stats([run["seconds_per_step"] for run in runs]),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Stage 4/5 spectral-weighting experiments"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = collect_runs(args.root)
    verification = spectrum_verification(records)
    payload = {
        "selection_split": "inner_train",
        "official_validation_used": False,
        "rho": EXPECTED_RHO,
        "rank": EXPECTED_RANK,
        "singular_value_exponents": list(EXPECTED_EXPONENTS),
        "train_seeds": list(EXPECTED_TRAIN_SEEDS),
        "random_basis_seeds": list(EXPECTED_BASIS_SEEDS),
        "run_count": len(records),
        "spectrum_verification": verification,
        "spectral_weighting": [
            spectral_weight_summary(records, exponent)
            for exponent in EXPECTED_EXPONENTS
        ],
        "exponent_summaries": {
            f"{exponent:g}": {
                "random_spectrum": method_summary(
                    records, "random_subspace", exponent
                ),
                "qr_svd": method_summary(records, "qr_svd", exponent),
                "random_basis_accuracy_range": basis_sensitivity(records, exponent),
            }
            for exponent in EXPECTED_EXPONENTS
        },
        "paired_direction_comparisons": [
            paired_direction_comparison(records, exponent)
            for exponent in EXPECTED_EXPONENTS
        ],
        "paired_weighting_effects": [
            paired_weighting_effect(records, method, exponent)
            for method in ("random_subspace", "qr_svd")
            for exponent in (0.5, 1.0)
        ],
        "groups": grouped_summaries(records),
        "max_perturbation_radius_error": max(
            record["max_perturbation_radius_error"] for record in records
        ),
        "runs": [
            {key: value for key, value in record.items() if key != "spectrum"}
            for record in records
        ],
    }
    output = args.output or args.root / "stage45_aggregate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Aggregated {len(records)} Stage 4/5 runs: {output}")


if __name__ == "__main__":
    main()
