from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def collect_runs(root: Path) -> list[dict[str, Any]]:
    records = []
    for summary_path in sorted((root / "runs").glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        config = read_json(run_dir / "resolved_config.json")
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
        records.append(
            {
                "path": str(run_dir),
                "method": summary["method"],
                "rho": float(config["sam"]["rho"]),
                "rank": int(config["sam"]["subspace_rank"]),
                "basis_seed": (
                    int(config["sam"]["random_basis_seed"])
                    if summary["method"] == "random_subspace"
                    else None
                ),
                "train_seed": int(config["training"]["seed"]),
                "data_seed": int(config["data"]["data_seed"]),
                "validation_source": config["data"]["validation_source"],
                "best_score": float(summary["best_score"]),
                "best_epoch": int(summary["best_epoch"]),
                "epochs_completed": int(summary["epochs_completed"]),
                "global_steps": int(summary["global_steps"]),
                "seconds_per_step": float(summary["elapsed_seconds"])
                / int(summary["global_steps"]),
                "mean_loss_increase": (
                    float(np.mean(loss_increases)) if loss_increases else None
                ),
                "mean_projected_gradient_norm": (
                    float(np.mean(projected_norms)) if projected_norms else None
                ),
            }
        )
    if not records:
        raise RuntimeError(f"No completed stage2 runs found under {root / 'runs'}")
    if any(record["validation_source"] != "inner_train" for record in records):
        raise RuntimeError("Stage2 aggregate rejected a run that used official validation")
    return records


def grouped_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["method"],
            record["rho"],
            record["rank"],
            record["basis_seed"],
        )
        grouped[key].append(record)
    output = []
    for (method, rho, rank, basis_seed), runs in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        output.append(
            {
                "method": method,
                "rho": rho,
                "rank": rank,
                "basis_seed": basis_seed,
                "best_score": stats([run["best_score"] for run in runs]),
                "mean_loss_increase": stats(
                    [
                        run["mean_loss_increase"]
                        for run in runs
                        if run["mean_loss_increase"] is not None
                    ]
                ),
                "mean_projected_gradient_norm": stats(
                    [
                        run["mean_projected_gradient_norm"]
                        for run in runs
                        if run["mean_projected_gradient_norm"] is not None
                    ]
                ),
                "seconds_per_step": stats([run["seconds_per_step"] for run in runs]),
            }
        )
    return output


def paired_comparison(
    records: list[dict[str, Any]], e4_rho: float, e5_rho: float
) -> dict[str, Any]:
    e4_by_seed: dict[int, list[float]] = defaultdict(list)
    e5_by_seed: dict[int, float] = {}
    for record in records:
        if record["method"] == "random_subspace" and record["rho"] == e4_rho:
            e4_by_seed[record["train_seed"]].append(record["best_score"])
        if record["method"] == "qr_svd" and record["rho"] == e5_rho:
            e5_by_seed[record["train_seed"]] = record["best_score"]
    seeds = sorted(set(e4_by_seed) & set(e5_by_seed))
    differences = [
        e5_by_seed[seed] - float(np.mean(e4_by_seed[seed])) for seed in seeds
    ]
    return {
        "e4_rho": e4_rho,
        "e5_rho": e5_rho,
        "e4_value_per_seed": "mean_across_random_basis_seeds",
        "seeds": seeds,
        "e5_minus_e4": stats(differences) if differences else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate stage2 inner-dev rho sweeps")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = collect_runs(args.root)
    payload = {
        "selection_split": "inner_train",
        "official_validation_used": False,
        "groups": grouped_summaries(records),
        "equal_radius_direction_comparisons": [
            paired_comparison(records, rho, rho) for rho in (0.025, 0.05, 0.10)
        ],
        "effective_strength_comparisons": [
            paired_comparison(records, 0.025, 0.0625),
            paired_comparison(records, 0.05, 0.125),
        ],
        "runs": records,
    }
    output = args.output or args.root / "stage2_aggregate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Aggregated {len(records)} runs: {output}")


if __name__ == "__main__":
    main()
