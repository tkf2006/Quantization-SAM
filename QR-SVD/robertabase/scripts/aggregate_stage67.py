from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SEEDS = (13, 21, 42, 87, 100)
EXPECTED_BASIS_SEEDS = (12345, 2026, 3407)
METHOD_ID = {
    "qlora": "e2",
    "lora_sam": "e3",
    "random_subspace": "e4",
    "qr_svd": "e5",
}
SETTING_SPECS = {
    "rte5": {"task": "rte", "fraction": 0.05, "selection_metric": "accuracy"},
    "rte20": {"task": "rte", "fraction": 0.20, "selection_metric": "accuracy"},
    "rte100": {"task": "rte", "fraction": 1.00, "selection_metric": "accuracy"},
    "mrpc20": {
        "task": "mrpc",
        "fraction": 0.20,
        "selection_metric": "combined_score",
    },
    "cola20": {
        "task": "cola",
        "fraction": 0.20,
        "selection_metric": "matthews_correlation",
    },
}
T95_DF4 = 2.7764451051977987


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sample_std": None, "values": []}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


def optional_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def infer_setting(task: str, fraction: float) -> str:
    matches = [
        setting
        for setting, spec in SETTING_SPECS.items()
        if task == spec["task"] and math.isclose(fraction, spec["fraction"], abs_tol=1e-9)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Unexpected or ambiguous task/fraction: {task}/{fraction}")
    return matches[0]


def best_validation(metrics: list[dict[str, Any]], best_epoch: int) -> dict[str, float]:
    matches = [
        record["validation"]
        for record in metrics
        if record.get("type") == "epoch" and int(record["epoch"]) == best_epoch
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one validation record at best epoch {best_epoch}")
    return {key: float(value) for key, value in matches[0].items()}


def collect_runs(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted((root / "runs").glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        config = read_json(run_dir / "resolved_config.json")
        data = read_json(run_dir / "data_summary.json")
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        train_steps = [record for record in metrics if record.get("type") == "train_step"]
        method = METHOD_ID[str(summary["method"])]
        task = str(config["data"]["task_name"])
        fraction = float(config["data"]["train_fraction"])
        setting = infer_setting(task, fraction)
        best_epoch = int(summary["best_epoch"])
        validation = best_validation(metrics, best_epoch)
        loss_increases = [
            float(record["loss"] - record["clean_loss"])
            for record in train_steps
            if "clean_loss" in record
        ]
        projected_norms = [
            float(record["projected_gradient_norm"])
            for record in train_steps
            if "projected_gradient_norm" in record
        ]
        perturbation_norms = [
            float(record["perturbation_norm"])
            for record in train_steps
            if "perturbation_norm" in record
        ]
        global_steps = int(summary["global_steps"])
        seconds_per_step = summary.get("seconds_per_step")
        if seconds_per_step is None:
            seconds_per_step = float(summary["elapsed_seconds"]) / global_steps
        records.append(
            {
                "path": str(run_dir),
                "setting": setting,
                "task": task,
                "train_fraction": fraction,
                "method": method,
                "train_seed": int(config["training"]["seed"]),
                "data_seed": int(config["data"]["data_seed"]),
                "basis_seed": (
                    int(config["sam"]["random_basis_seed"]) if method == "e4" else None
                ),
                "rho": float(config["sam"]["rho"]) if method != "e2" else None,
                "rank": int(config["sam"]["subspace_rank"]) if method in {"e4", "e5"} else None,
                "singular_value_exponent": (
                    float(config["sam"]["singular_value_exponent"])
                    if method in {"e4", "e5"}
                    else None
                ),
                "validation_source": str(config["data"]["validation_source"]),
                "official_validation_used": bool(data["official_validation_used"]),
                "deterministic": bool(config["runtime"]["deterministic"]),
                "collect_subspace_comparisons": bool(
                    config["qr_svd"].get("collect_subspace_comparisons", True)
                ),
                "save_last_adapter": bool(config["training"].get("save_last_adapter", True)),
                "save_optimizer_state": bool(
                    config["training"].get("save_optimizer_state", True)
                ),
                "save_tokenizer_with_adapter": bool(
                    config["training"].get("save_tokenizer_with_adapter", True)
                ),
                "selected_training_examples": int(data["selected_training_pool_examples"]),
                "validation_examples": int(data["validation_examples"]),
                "selection_metric": str(SETTING_SPECS[setting]["selection_metric"]),
                "best_score": float(summary["best_score"]),
                "best_validation": validation,
                "best_epoch": best_epoch,
                "epochs_completed": int(summary["epochs_completed"]),
                "global_steps": global_steps,
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "seconds_per_step": float(seconds_per_step),
                "initial_memory_allocated_gib": float(
                    summary["initial_memory_allocated_gib"]
                ),
                "peak_memory_allocated_gib": float(summary["peak_memory_allocated_gib"]),
                "peak_memory_reserved_gib": float(summary["peak_memory_reserved_gib"]),
                "mean_loss_increase": optional_mean(loss_increases),
                "positive_loss_increase_fraction": (
                    float(np.mean(np.asarray(loss_increases) > 0))
                    if loss_increases
                    else None
                ),
                "mean_projected_gradient_norm": optional_mean(projected_norms),
                "mean_perturbation_norm": optional_mean(perturbation_norms),
            }
        )
    validate_matrix(records)
    return records


def validate_matrix(records: list[dict[str, Any]]) -> None:
    expected_count = len(SETTING_SPECS) * len(EXPECTED_SEEDS) * (
        3 + len(EXPECTED_BASIS_SEEDS)
    )
    if len(records) != expected_count:
        raise RuntimeError(f"Expected {expected_count} Stage 6-7 runs, found {len(records)}")
    for record in records:
        if record["train_seed"] not in EXPECTED_SEEDS:
            raise RuntimeError(f"Unexpected train seed in {record['path']}")
        if record["data_seed"] != record["train_seed"]:
            raise RuntimeError(f"Train/data seed mismatch in {record['path']}")
        if record["validation_source"] != "official" or not record["official_validation_used"]:
            raise RuntimeError(f"Official validation was not used in {record['path']}")
        if not record["deterministic"]:
            raise RuntimeError(f"Non-deterministic run: {record['path']}")
        if record["collect_subspace_comparisons"]:
            raise RuntimeError(f"Formal cost run contains diagnostic comparison overhead: {record['path']}")
        if record["save_last_adapter"] or record["save_optimizer_state"] or record["save_tokenizer_with_adapter"]:
            raise RuntimeError(f"Stage 6-7 compact-save policy mismatch: {record['path']}")
        if not all(
            math.isfinite(record[key])
            for key in (
                "best_score",
                "seconds_per_step",
                "peak_memory_allocated_gib",
                "peak_memory_reserved_gib",
            )
        ):
            raise RuntimeError(f"Non-finite metric in {record['path']}")
        metric_name = record["selection_metric"]
        if not math.isclose(
            record["best_score"], record["best_validation"][metric_name], abs_tol=1e-12
        ):
            raise RuntimeError(f"Best-score/validation mismatch in {record['path']}")
        if record["method"] == "e3" and not math.isclose(record["rho"], 0.01):
            raise RuntimeError(f"E3 rho mismatch in {record['path']}")
        if record["method"] in {"e4", "e5"}:
            if not math.isclose(record["rho"], 0.05) or record["rank"] != 8:
                raise RuntimeError(f"E4/E5 rho/rank mismatch in {record['path']}")
            if record["singular_value_exponent"] != 0.0:
                raise RuntimeError(f"E4/E5 exponent mismatch in {record['path']}")

    for setting in SETTING_SPECS:
        for method in ("e2", "e3", "e5"):
            observed = {
                record["train_seed"]
                for record in records
                if record["setting"] == setting and record["method"] == method
            }
            if observed != set(EXPECTED_SEEDS):
                raise RuntimeError(f"Incomplete {setting}/{method} matrix: {sorted(observed)}")
        observed_e4 = {
            (record["train_seed"], record["basis_seed"])
            for record in records
            if record["setting"] == setting and record["method"] == "e4"
        }
        expected_e4 = {
            (seed, basis_seed)
            for seed in EXPECTED_SEEDS
            for basis_seed in EXPECTED_BASIS_SEEDS
        }
        if observed_e4 != expected_e4:
            raise RuntimeError(f"Incomplete {setting}/e4 matrix: {sorted(observed_e4)}")


def per_seed_values(
    records: list[dict[str, Any]], setting: str, method: str, metric: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record["setting"] != setting or record["method"] != method:
            continue
        value: Any
        if metric.startswith("validation."):
            value = record["best_validation"].get(metric.split(".", 1)[1])
        else:
            value = record.get(metric)
        if value is not None:
            grouped[record["train_seed"]].append(float(value))
    return {seed: float(np.mean(values)) for seed, values in sorted(grouped.items())}


def stats_by_seed(values: dict[int, float]) -> dict[str, Any]:
    result = stats([values[seed] for seed in sorted(values)])
    result["values_by_seed"] = {str(seed): values[seed] for seed in sorted(values)}
    return result


def method_summary(
    records: list[dict[str, Any]], setting: str, method: str
) -> dict[str, Any]:
    metric_names = (
        "best_score",
        "validation.accuracy",
        "validation.f1",
        "validation.combined_score",
        "validation.matthews_correlation",
        "validation.loss",
        "seconds_per_step",
        "peak_memory_allocated_gib",
        "peak_memory_reserved_gib",
        "mean_loss_increase",
        "positive_loss_increase_fraction",
        "mean_projected_gradient_norm",
        "mean_perturbation_norm",
    )
    output: dict[str, Any] = {}
    for metric in metric_names:
        values = per_seed_values(records, setting, method, metric)
        if values:
            output[metric.replace("validation.", "validation_")] = stats_by_seed(values)
    return output


def paired_comparison(
    records: list[dict[str, Any]], setting: str, reference: str
) -> dict[str, Any]:
    e5 = per_seed_values(records, setting, "e5", "best_score")
    other = per_seed_values(records, setting, reference, "best_score")
    seeds = sorted(set(e5) & set(other))
    differences = [e5[seed] - other[seed] for seed in seeds]
    result = stats(differences)
    mean = float(result["mean"])
    sample_std = float(result["sample_std"])
    half_width = T95_DF4 * sample_std / math.sqrt(len(differences))
    result.update(
        {
            "reference": reference,
            "e4_value_per_seed": (
                "mean_across_random_basis_seeds" if reference == "e4" else None
            ),
            "seeds": seeds,
            "values_by_seed": {
                str(seed): e5[seed] - other[seed] for seed in seeds
            },
            "mean_percentage_points": 100.0 * mean,
            "sample_std_percentage_points": 100.0 * sample_std,
            "exploratory_95ci_percentage_points": [
                100.0 * (mean - half_width),
                100.0 * (mean + half_width),
            ],
            "wins_ties_losses": {
                "wins": sum(value > 1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "losses": sum(value < -1e-12 for value in differences),
            },
        }
    )
    return result


def e4_basis_sensitivity(records: list[dict[str, Any]], setting: str) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record["setting"] == setting and record["method"] == "e4":
            grouped[record["train_seed"]].append(record["best_score"])
    ranges = {seed: max(values) - min(values) for seed, values in sorted(grouped.items())}
    return stats_by_seed(ranges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the formal Stage 6-7 matrix")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = collect_runs(args.root)
    payload = {
        "selection_split": "official_validation",
        "official_validation_used": True,
        "formal_seeds": list(EXPECTED_SEEDS),
        "random_basis_seeds": list(EXPECTED_BASIS_SEEDS),
        "fixed_hyperparameters": {
            "e3_rho": 0.01,
            "e4_e5_rho": 0.05,
            "e4_e5_rank": 8,
            "e4_e5_singular_value_exponent": 0.0,
        },
        "settings": SETTING_SPECS,
        "method_summaries": {
            setting: {
                method: method_summary(records, setting, method)
                for method in ("e2", "e3", "e4", "e5")
            }
            for setting in SETTING_SPECS
        },
        "paired_comparisons": {
            setting: {
                "e5_minus_e4": paired_comparison(records, setting, "e4"),
                "e5_minus_e2": paired_comparison(records, setting, "e2"),
                "e5_minus_e3": paired_comparison(records, setting, "e3"),
            }
            for setting in SETTING_SPECS
        },
        "e4_basis_sensitivity": {
            setting: e4_basis_sensitivity(records, setting) for setting in SETTING_SPECS
        },
        "runs": records,
    }
    output = args.output or args.root / "stage67_aggregate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Aggregated {len(records)} Stage 6-7 runs: {output}")


if __name__ == "__main__":
    main()
