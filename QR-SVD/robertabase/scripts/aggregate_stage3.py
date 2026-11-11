from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_RANKS = (4, 8, 16, 32)
EXPECTED_TRAIN_SEEDS = (17, 29, 43)
EXPECTED_BASIS_SEEDS = (12345, 2026, 3407)
EXPECTED_RHO = 0.05


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
        "values": values,
    }


def qr_energy_summary(run_dir: Path) -> dict[str, float | int]:
    layers = read_json(run_dir / "qr_svd_basis.json")["layers"]
    if len(layers) != 24:
        raise RuntimeError(f"Expected 24 QR-SVD target layers in {run_dir}, got {len(layers)}")
    all_values = [float(layer["captured_energy_ratio"]) for layer in layers]
    query_values = [
        float(layer["captured_energy_ratio"])
        for layer in layers
        if layer["name"].endswith(".query")
    ]
    value_values = [
        float(layer["captured_energy_ratio"])
        for layer in layers
        if layer["name"].endswith(".value")
    ]
    return {
        "target_layers": len(layers),
        "mean": float(np.mean(all_values)),
        "minimum": float(np.min(all_values)),
        "maximum": float(np.max(all_values)),
        "query_mean": float(np.mean(query_values)),
        "value_mean": float(np.mean(value_values)),
    }


def collect_runs(root: Path) -> list[dict[str, Any]]:
    records = []
    for summary_path in sorted((root / "runs").glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        config = read_json(run_dir / "resolved_config.json")
        data = read_json(run_dir / "data_summary.json")
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        train_steps = [record for record in metrics if record.get("type") == "train_step"]
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
        method = summary["method"]
        records.append(
            {
                "path": str(run_dir),
                "method": method,
                "rho": float(config["sam"]["rho"]),
                "rank": int(config["sam"]["subspace_rank"]),
                "basis_seed": (
                    int(config["sam"]["random_basis_seed"])
                    if method == "random_subspace"
                    else None
                ),
                "train_seed": int(config["training"]["seed"]),
                "data_seed": int(config["data"]["data_seed"]),
                "validation_source": config["data"]["validation_source"],
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
                    np.mean([value > 0 for value in loss_increases])
                ),
                "mean_projected_gradient_norm": float(np.mean(projected_norms)),
                "qr_residual_energy": (
                    qr_energy_summary(run_dir) if method == "qr_svd" else None
                ),
            }
        )
    validate_matrix(records)
    return records


def validate_matrix(records: list[dict[str, Any]]) -> None:
    expected_runs = len(EXPECTED_RANKS) * len(EXPECTED_TRAIN_SEEDS) * (
        len(EXPECTED_BASIS_SEEDS) + 1
    )
    if len(records) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} Stage 3 runs, found {len(records)}")
    for record in records:
        if record["rho"] != EXPECTED_RHO:
            raise RuntimeError(f"Unexpected rho in {record['path']}: {record['rho']}")
        if record["rank"] not in EXPECTED_RANKS:
            raise RuntimeError(f"Unexpected rank in {record['path']}: {record['rank']}")
        if record["train_seed"] not in EXPECTED_TRAIN_SEEDS:
            raise RuntimeError(
                f"Unexpected training seed in {record['path']}: {record['train_seed']}"
            )
        if record["data_seed"] != record["train_seed"]:
            raise RuntimeError(f"Train/data seed mismatch in {record['path']}")
        if record["validation_source"] != "inner_train" or record[
            "official_validation_used"
        ]:
            raise RuntimeError(f"Official validation leaked into {record['path']}")
        if not record["deterministic"]:
            raise RuntimeError(f"Non-deterministic Stage 3 run: {record['path']}")

    for rank in EXPECTED_RANKS:
        e4 = {
            (record["train_seed"], record["basis_seed"])
            for record in records
            if record["method"] == "random_subspace" and record["rank"] == rank
        }
        expected_e4 = {
            (seed, basis_seed)
            for seed in EXPECTED_TRAIN_SEEDS
            for basis_seed in EXPECTED_BASIS_SEEDS
        }
        e5 = {
            record["train_seed"]
            for record in records
            if record["method"] == "qr_svd" and record["rank"] == rank
        }
        if e4 != expected_e4:
            raise RuntimeError(f"Incomplete E4 matrix at rank {rank}: {sorted(e4)}")
        if e5 != set(EXPECTED_TRAIN_SEEDS):
            raise RuntimeError(f"Incomplete E5 matrix at rank {rank}: {sorted(e5)}")


def per_seed_values(
    records: list[dict[str, Any]], method: str, rank: int, metric: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record["method"] == method and record["rank"] == rank:
            grouped[record["train_seed"]].append(float(record[metric]))
    return {seed: float(np.mean(values)) for seed, values in sorted(grouped.items())}


def method_rank_summary(
    records: list[dict[str, Any]], method: str, rank: int
) -> dict[str, Any]:
    metrics = (
        "best_score",
        "mean_loss_increase",
        "positive_loss_increase_fraction",
        "mean_projected_gradient_norm",
        "seconds_per_step",
    )
    return {
        metric: stats(list(per_seed_values(records, method, rank, metric).values()))
        for metric in metrics
    }


def basis_sensitivity(records: list[dict[str, Any]], rank: int) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record["method"] == "random_subspace" and record["rank"] == rank:
            grouped[record["train_seed"]].append(record["best_score"])
    ranges = [max(values) - min(values) for _seed, values in sorted(grouped.items())]
    return stats(ranges)


def paired_comparison(records: list[dict[str, Any]], rank: int) -> dict[str, Any]:
    e4 = per_seed_values(records, "random_subspace", rank, "best_score")
    e5 = per_seed_values(records, "qr_svd", rank, "best_score")
    seeds = sorted(set(e4) & set(e5))
    differences = [e5[seed] - e4[seed] for seed in seeds]
    return {
        "rank": rank,
        "e4_value_per_seed": "mean_across_random_basis_seeds",
        "seeds": seeds,
        "e5_minus_e4": stats(differences),
    }


def energy_summary(records: list[dict[str, Any]], rank: int) -> dict[str, Any]:
    selected = [
        record["qr_residual_energy"]
        for record in records
        if record["method"] == "qr_svd" and record["rank"] == rank
    ]
    return {
        "rank": rank,
        "mean_across_layers": stats([item["mean"] for item in selected]),
        "query_mean": stats([item["query_mean"] for item in selected]),
        "value_mean": stats([item["value_mean"] for item in selected]),
        "layer_minimum": float(min(item["minimum"] for item in selected)),
        "layer_maximum": float(max(item["maximum"] for item in selected)),
    }


def grouped_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["rank"], record["basis_seed"])].append(record)
    output = []
    for (method, rank, basis_seed), runs in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        output.append(
            {
                "method": method,
                "rank": rank,
                "basis_seed": basis_seed,
                "best_score": stats([run["best_score"] for run in runs]),
                "mean_loss_increase": stats(
                    [run["mean_loss_increase"] for run in runs]
                ),
                "mean_projected_gradient_norm": stats(
                    [run["mean_projected_gradient_norm"] for run in runs]
                ),
                "seconds_per_step": stats([run["seconds_per_step"] for run in runs]),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage 3 rank sweeps")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = collect_runs(args.root)
    payload = {
        "selection_split": "inner_train",
        "official_validation_used": False,
        "rho": EXPECTED_RHO,
        "ranks": list(EXPECTED_RANKS),
        "train_seeds": list(EXPECTED_TRAIN_SEEDS),
        "random_basis_seeds": list(EXPECTED_BASIS_SEEDS),
        "rank_summaries": {
            str(rank): {
                "e4": method_rank_summary(records, "random_subspace", rank),
                "e5": method_rank_summary(records, "qr_svd", rank),
                "e4_basis_accuracy_range": basis_sensitivity(records, rank),
            }
            for rank in EXPECTED_RANKS
        },
        "paired_comparisons": [
            paired_comparison(records, rank) for rank in EXPECTED_RANKS
        ],
        "qr_residual_energy": [
            energy_summary(records, rank) for rank in EXPECTED_RANKS
        ],
        "groups": grouped_summaries(records),
        "runs": records,
    }
    output = args.output or args.root / "stage3_aggregate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Aggregated {len(records)} Stage 3 runs: {output}")


if __name__ == "__main__":
    main()
