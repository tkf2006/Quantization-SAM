from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"e4": "#4E79A7", "e5": "#E15759", "energy": "#59A14F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage 3 rank-search results")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


def seed_values(
    runs: list[dict[str, Any]], method: str, metric: str
) -> dict[int, dict[int, float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for run in runs:
        if run["method"] == method and run.get(metric) is not None:
            grouped[(int(run["rank"]), int(run["train_seed"]))].append(
                float(run[metric])
            )
    values: dict[int, dict[int, float]] = defaultdict(dict)
    for (rank, train_seed), observations in grouped.items():
        # E4's three basis seeds are averaged first, leaving the training seed
        # as the independent unit. E5 has one observation for each seed.
        values[rank][train_seed] = float(np.mean(observations))
    return dict(values)


def metric_stats(
    runs: list[dict[str, Any]], ranks: list[int], method: str, metric: str
) -> dict[str, dict[str, Any]]:
    values = seed_values(runs, method, metric)
    return {str(rank): sample_stats(list(values[rank].values())) for rank in ranks}


def mean_std(values: dict[int, dict[int, float]], ranks: list[int]) -> tuple[list[float], list[float]]:
    means = [float(np.mean(list(values[rank].values()))) for rank in ranks]
    stds = [float(np.std(list(values[rank].values()), ddof=1)) for rank in ranks]
    return means, stds


def plot_accuracy(
    runs: list[dict[str, Any]], aggregate: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    ranks = [int(rank) for rank in aggregate["ranks"]]
    e4 = seed_values(runs, "random_subspace", "best_score")
    e5 = seed_values(runs, "qr_svd", "best_score")
    e4_means, e4_stds = mean_std(e4, ranks)
    e5_means, e5_stds = mean_std(e5, ranks)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for values, color in ((e4, COLORS["e4"]), (e5, COLORS["e5"])):
        for seed in sorted(next(iter(values.values()))):
            axes[0].plot(
                ranks,
                [100 * values[rank][seed] for rank in ranks],
                color=color,
                alpha=0.18,
                linewidth=1,
            )
    axes[0].errorbar(
        ranks,
        100 * np.asarray(e4_means),
        yerr=100 * np.asarray(e4_stds),
        marker="s",
        capsize=4,
        linewidth=2,
        color=COLORS["e4"],
        label="E4 Random (3-basis mean)",
    )
    axes[0].errorbar(
        ranks,
        100 * np.asarray(e5_means),
        yerr=100 * np.asarray(e5_stds),
        marker="o",
        capsize=4,
        linewidth=2,
        color=COLORS["e5"],
        label="E5 QR-SVD",
    )
    axes[0].set_xticks(ranks)
    axes[0].set_xlabel("Subspace rank")
    axes[0].set_ylabel("Best inner-dev accuracy (%)")
    axes[0].set_title("Accuracy across training seeds")
    axes[0].legend(fontsize=9)

    paired_by_rank = {
        int(item["rank"]): [100 * float(value) for value in item["e5_minus_e4"]["values"]]
        for item in aggregate["paired_comparisons"]
    }
    paired_means = [float(np.mean(paired_by_rank[rank])) for rank in ranks]
    paired_stds = [float(np.std(paired_by_rank[rank], ddof=1)) for rank in ranks]
    axes[1].bar(
        ranks,
        paired_means,
        yerr=paired_stds,
        width=[2.6, 4.0, 6.0, 8.0],
        capsize=4,
        color=[COLORS["e5"] if value > 0 else "#BAB0AC" for value in paired_means],
        alpha=0.85,
    )
    for rank in ranks:
        observations = paired_by_rank[rank]
        offsets = np.linspace(-0.7, 0.7, len(observations)) * max(rank / 16, 0.4)
        axes[1].scatter(
            rank + offsets, observations, color="#222222", s=25, zorder=3
        )
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(ranks)
    axes[1].set_xlabel("Subspace rank")
    axes[1].set_ylabel("Paired E5 - E4 accuracy (pp)")
    axes[1].set_title("Direction advantage")

    by_seed_rank: dict[tuple[int, int], list[float]] = defaultdict(list)
    for run in runs:
        if run["method"] == "random_subspace":
            by_seed_rank[(int(run["rank"]), int(run["train_seed"]))].append(
                100 * float(run["best_score"])
            )
    ranges = {
        rank: [
            max(by_seed_rank[(rank, seed)]) - min(by_seed_rank[(rank, seed)])
            for seed in sorted({key[1] for key in by_seed_rank if key[0] == rank})
        ]
        for rank in ranks
    }
    range_means = [float(np.mean(ranges[rank])) for rank in ranks]
    axes[2].bar(
        ranks,
        range_means,
        width=[2.6, 4.0, 6.0, 8.0],
        color=COLORS["e4"],
        alpha=0.85,
    )
    for rank in ranks:
        offsets = np.linspace(-0.7, 0.7, len(ranges[rank])) * max(rank / 16, 0.4)
        axes[2].scatter(
            rank + offsets, ranges[rank], color="#222222", s=25, zorder=3
        )
    axes[2].set_xticks(ranks)
    axes[2].set_xlabel("Subspace rank")
    axes[2].set_ylabel("Within-seed max-min accuracy (pp)")
    axes[2].set_title("E4 random-basis sensitivity")

    fig.suptitle("Stage 3 rank search: accuracy and direction comparison", fontweight="bold")
    fig.tight_layout()
    path = output_dir / "stage3_rank_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "paired_e5_minus_e4_pp": {
            str(rank): sample_stats(paired_by_rank[rank]) for rank in ranks
        },
        "e4_basis_range_pp": {
            str(rank): sample_stats(ranges[rank]) for rank in ranks
        },
    }


def plot_mechanism_and_cost(
    runs: list[dict[str, Any]], aggregate: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    ranks = [int(rank) for rank in aggregate["ranks"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    energy = {
        int(item["rank"]): item for item in aggregate["qr_residual_energy"]
    }
    means = [100 * float(energy[rank]["mean_across_layers"]["mean"]) for rank in ranks]
    minima = [100 * float(energy[rank]["layer_minimum"]) for rank in ranks]
    maxima = [100 * float(energy[rank]["layer_maximum"]) for rank in ranks]
    axes[0, 0].plot(ranks, means, marker="o", linewidth=2, color=COLORS["energy"])
    axes[0, 0].fill_between(ranks, minima, maxima, color=COLORS["energy"], alpha=0.18)
    axes[0, 0].set_xticks(ranks)
    axes[0, 0].set_xlabel("Subspace rank")
    axes[0, 0].set_ylabel("Residual energy covered (%)")
    axes[0, 0].set_title("QR residual-energy coverage (24 matrices)")

    for method, label, marker in (
        ("random_subspace", "E4 Random", "s"),
        ("qr_svd", "E5 QR-SVD", "o"),
    ):
        values = seed_values(runs, method, "mean_projected_gradient_norm")
        metric_means, metric_stds = mean_std(values, ranks)
        color = COLORS["e4"] if method == "random_subspace" else COLORS["e5"]
        axes[0, 1].errorbar(
            ranks,
            metric_means,
            yerr=metric_stds,
            marker=marker,
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    axes[0, 1].set_xticks(ranks)
    axes[0, 1].set_xlabel("Subspace rank")
    axes[0, 1].set_ylabel("Mean projected-gradient norm")
    axes[0, 1].set_title("Gradient captured by each subspace")
    axes[0, 1].legend()

    for method, label, marker in (
        ("random_subspace", "E4 Random", "s"),
        ("qr_svd", "E5 QR-SVD", "o"),
    ):
        values = seed_values(runs, method, "mean_loss_increase")
        metric_means, metric_stds = mean_std(values, ranks)
        color = COLORS["e4"] if method == "random_subspace" else COLORS["e5"]
        axes[1, 0].errorbar(
            ranks,
            metric_means,
            yerr=metric_stds,
            marker=marker,
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    axes[1, 0].set_xticks(ranks)
    axes[1, 0].set_xlabel("Subspace rank")
    axes[1, 0].set_ylabel("Mean perturbed - clean loss")
    axes[1, 0].set_title("Effective perturbation strength")
    axes[1, 0].legend()

    for method, label, marker in (
        ("random_subspace", "E4 Random", "s"),
        ("qr_svd", "E5 QR-SVD", "o"),
    ):
        values = seed_values(runs, method, "seconds_per_step")
        metric_means, metric_stds = mean_std(values, ranks)
        color = COLORS["e4"] if method == "random_subspace" else COLORS["e5"]
        axes[1, 1].errorbar(
            ranks,
            metric_means,
            yerr=metric_stds,
            marker=marker,
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    axes[1, 1].set_xticks(ranks)
    axes[1, 1].set_xlabel("Subspace rank")
    axes[1, 1].set_ylabel("Seconds per training step")
    axes[1, 1].set_title("Measured training cost")
    axes[1, 1].legend()

    fig.suptitle("Stage 3 rank search: mechanism and cost", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "stage3_rank_mechanism_cost.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "residual_energy_percent": {
            str(rank): {
                "mean": means[index],
                "layer_minimum": minima[index],
                "layer_maximum": maxima[index],
            }
            for index, rank in enumerate(ranks)
        }
    }


def integrity_summary(run_dir: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    configs = []
    data_summaries = []
    perturbation_errors = []
    nonfinite_values = 0
    loss_increases = []
    for run in runs:
        directory = Path(run["path"])
        if not (directory / "resolved_config.json").is_file():
            # Historical aggregates retain their original outputs/... path.
            # Resolve the same run inside the relocated stage archive.
            directory = run_dir / "runs" / directory.name
        configs.append(read_json(directory / "resolved_config.json"))
        data_summaries.append(read_json(directory / "data_summary.json"))
        rho = float(run["rho"])
        with (directory / "metrics.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for value in record.values():
                    if isinstance(value, float) and not np.isfinite(value):
                        nonfinite_values += 1
                if record.get("type") != "train_step":
                    continue
                if record.get("perturbation_norm") is not None:
                    perturbation_errors.append(
                        abs(float(record["perturbation_norm"]) - rho)
                    )
                if record.get("clean_loss") is not None:
                    loss_increases.append(float(record["loss"]) - float(record["clean_loss"]))
    return {
        "all_deterministic": all(bool(run["deterministic"]) for run in runs),
        "official_validation_used": any(
            bool(item["official_validation_used"]) for item in data_summaries
        ),
        "validation_sources": sorted({run["validation_source"] for run in runs}),
        "train_example_counts": sorted({item["train_examples"] for item in data_summaries}),
        "inner_dev_example_counts": sorted(
            {item["validation_examples"] for item in data_summaries}
        ),
        "max_perturbation_radius_error": max(perturbation_errors, default=0.0),
        "nonfinite_scalar_values": nonfinite_values,
        "positive_loss_increase_fraction": float(np.mean(np.asarray(loss_increases) > 0)),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = read_json(run_dir / "stage3_aggregate.json")
    runs = aggregate["runs"]
    if len(runs) != 48:
        raise RuntimeError(f"Expected 48 completed runs, found {len(runs)}")
    if aggregate["official_validation_used"]:
        raise RuntimeError("Stage 3 aggregate reports official validation use")
    if any(not run["deterministic"] for run in runs):
        raise RuntimeError("At least one Stage 3 run was non-deterministic")

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    accuracy = plot_accuracy(runs, aggregate, output_dir)
    mechanism = plot_mechanism_and_cost(runs, aggregate, output_dir)
    ranks = [int(rank) for rank in aggregate["ranks"]]
    summary = {
        "run_dir": str(run_dir),
        "completed_runs": len(runs),
        "metrics": {
            method: {
                metric: metric_stats(runs, ranks, method, metric)
                for metric in (
                    "best_score",
                    "mean_projected_gradient_norm",
                    "mean_loss_increase",
                    "seconds_per_step",
                )
            }
            for method in ("random_subspace", "qr_svd")
        },
        **accuracy,
        **mechanism,
        "integrity": integrity_summary(run_dir, runs),
        "figures": [
            str(output_dir / "stage3_rank_accuracy.png"),
            str(output_dir / "stage3_rank_mechanism_cost.png"),
        ],
    }
    with (output_dir / "stage3_plot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
