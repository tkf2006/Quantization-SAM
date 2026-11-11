from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "lora_sam": "#59A14F",
    "random_subspace": "#4E79A7",
    "qr_svd": "#E15759",
}
LABELS = {
    "lora_sam": "E3 LoRA-SAM",
    "random_subspace": "E4 Random",
    "qr_svd": "E5 QR-SVD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage 2 rho-search results")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sample_std": None, "values": []}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def method_seed_values(
    runs: list[dict[str, Any]], method: str, metric: str
) -> dict[float, dict[int, float]]:
    grouped: dict[tuple[float, int], list[float]] = defaultdict(list)
    for run in runs:
        if run["method"] != method or run.get(metric) is None:
            continue
        grouped[(float(run["rho"]), int(run["train_seed"]))].append(
            float(run[metric])
        )
    result: dict[float, dict[int, float]] = defaultdict(dict)
    for (rho, seed), values in grouped.items():
        # E4 is averaged across random-basis seeds before the training seed is
        # treated as one independent unit. E3/E5 have one value per seed.
        result[rho][seed] = float(np.mean(values))
    return dict(result)


def summarize_grid(
    runs: list[dict[str, Any]], method: str
) -> dict[str, dict[str, Any]]:
    score = method_seed_values(runs, method, "best_score")
    loss = method_seed_values(runs, method, "mean_loss_increase")
    projection = method_seed_values(runs, method, "mean_projected_gradient_norm")
    summary = {}
    for rho in sorted(score):
        summary[f"{rho:g}"] = {
            "best_score": sample_stats(list(score[rho].values())),
            "mean_loss_increase": sample_stats(list(loss.get(rho, {}).values())),
            "mean_projected_gradient_norm": sample_stats(
                list(projection.get(rho, {}).values())
            ),
        }
    return summary


def plot_seed_curves(
    ax,
    values: dict[float, dict[int, float]],
    rhos: list[float],
    *,
    color: str,
    label: str,
    marker: str,
) -> None:
    seeds = sorted({seed for rho in rhos for seed in values[rho]})
    for seed in seeds:
        ax.plot(
            rhos,
            [100 * values[rho][seed] for rho in rhos],
            color=color,
            alpha=0.20,
            linewidth=1,
        )
    means = [100 * np.mean(list(values[rho].values())) for rho in rhos]
    stds = [
        100 * np.std(list(values[rho].values()), ddof=1)
        if len(values[rho]) > 1
        else 0.0
        for rho in rhos
    ]
    ax.errorbar(
        rhos,
        means,
        yerr=stds,
        color=color,
        marker=marker,
        linewidth=2,
        capsize=4,
        label=label,
    )


def paired_differences(
    runs: list[dict[str, Any]], e4_rho: float, e5_rho: float
) -> list[float]:
    e4 = method_seed_values(runs, "random_subspace", "best_score")[e4_rho]
    e5 = method_seed_values(runs, "qr_svd", "best_score")[e5_rho]
    seeds = sorted(set(e4) & set(e5))
    return [100 * (e5[seed] - e4[seed]) for seed in seeds]


def accuracy_figure(runs: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    e3 = method_seed_values(runs, "lora_sam", "best_score")
    e4 = method_seed_values(runs, "random_subspace", "best_score")
    e5 = method_seed_values(runs, "qr_svd", "best_score")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    e3_rhos = sorted(e3)
    plot_seed_curves(
        axes[0, 0],
        e3,
        e3_rhos,
        color=COLORS["lora_sam"],
        label=LABELS["lora_sam"],
        marker="o",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xticks(e3_rhos, [f"{rho:g}" for rho in e3_rhos])
    axes[0, 0].minorticks_off()
    axes[0, 0].set_title("E3 independent rho search")
    axes[0, 0].set_xlabel("rho")
    axes[0, 0].set_ylabel("Best inner-dev accuracy (%)")
    axes[0, 0].legend()

    equal_rhos = [0.025, 0.05, 0.1]
    plot_seed_curves(
        axes[0, 1],
        e4,
        equal_rhos,
        color=COLORS["random_subspace"],
        label="E4 mean across 3 bases",
        marker="s",
    )
    plot_seed_curves(
        axes[0, 1],
        e5,
        equal_rhos,
        color=COLORS["qr_svd"],
        label=LABELS["qr_svd"],
        marker="o",
    )
    axes[0, 1].set_xticks(equal_rhos, [f"{rho:g}" for rho in equal_rhos])
    axes[0, 1].set_title("Equal-radius direction comparison")
    axes[0, 1].set_xlabel("rho")
    axes[0, 1].set_ylabel("Best inner-dev accuracy (%)")
    axes[0, 1].legend()

    e5_rhos = sorted(e5)
    plot_seed_curves(
        axes[1, 0],
        e5,
        e5_rhos,
        color=COLORS["qr_svd"],
        label=LABELS["qr_svd"],
        marker="o",
    )
    axes[1, 0].set_xticks(e5_rhos, [f"{rho:g}" for rho in e5_rhos])
    axes[1, 0].set_title("E5 equal-radius + calibrated candidates")
    axes[1, 0].set_xlabel("rho")
    axes[1, 0].set_ylabel("Best inner-dev accuracy (%)")
    axes[1, 0].legend()

    comparisons = [
        ("equal\n.025/.025", 0.025, 0.025),
        ("equal\n.05/.05", 0.05, 0.05),
        ("equal\n.10/.10", 0.1, 0.1),
        ("calibrated\n.025/.0625", 0.025, 0.0625),
        ("calibrated\n.05/.125", 0.05, 0.125),
    ]
    all_differences = [
        paired_differences(runs, e4_rho, e5_rho)
        for _label, e4_rho, e5_rho in comparisons
    ]
    x = np.arange(len(comparisons))
    means = [np.mean(values) for values in all_differences]
    stds = [np.std(values, ddof=1) for values in all_differences]
    axes[1, 1].bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[COLORS["qr_svd"] if mean >= 0 else "#BAB0AC" for mean in means],
        alpha=0.85,
    )
    for index, values in enumerate(all_differences):
        offsets = np.linspace(-0.10, 0.10, len(values))
        axes[1, 1].scatter(
            index + offsets,
            values,
            color="#222222",
            s=24,
            zorder=3,
        )
    axes[1, 1].axhline(0, color="#333333", linewidth=1)
    axes[1, 1].set_xticks(x, [item[0] for item in comparisons])
    axes[1, 1].set_ylabel("Paired E5 - E4 accuracy (pp)")
    axes[1, 1].set_title("Direction and effective-strength comparisons")

    fig.suptitle(
        "Stage 2 rho search: inner-dev accuracy",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / "stage2_rho_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        f"e4_{e4_rho:g}_e5_{e5_rho:g}": sample_stats(values)
        for (_label, e4_rho, e5_rho), values in zip(comparisons, all_differences)
    }


def strength_figure(runs: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for method, marker in (
        ("lora_sam", "^"),
        ("random_subspace", "s"),
        ("qr_svd", "o"),
    ):
        values = method_seed_values(runs, method, "mean_loss_increase")
        rhos = sorted(values)
        means = [np.mean(list(values[rho].values())) for rho in rhos]
        stds = [np.std(list(values[rho].values()), ddof=1) for rho in rhos]
        axes[0, 0].errorbar(
            rhos,
            means,
            yerr=stds,
            color=COLORS[method],
            marker=marker,
            capsize=4,
            linewidth=2,
            label=LABELS[method],
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("rho")
    axes[0, 0].set_ylabel("Mean perturbed - clean loss")
    axes[0, 0].set_title("Effective perturbation strength")
    axes[0, 0].legend()

    for method, marker in (("random_subspace", "s"), ("qr_svd", "o")):
        values = method_seed_values(runs, method, "mean_projected_gradient_norm")
        rhos = sorted(values)
        means = [np.mean(list(values[rho].values())) for rho in rhos]
        stds = [np.std(list(values[rho].values()), ddof=1) for rho in rhos]
        axes[0, 1].errorbar(
            rhos,
            means,
            yerr=stds,
            color=COLORS[method],
            marker=marker,
            capsize=4,
            linewidth=2,
            label=LABELS[method],
        )
    axes[0, 1].set_xlabel("rho")
    axes[0, 1].set_ylabel("Mean projected-gradient norm")
    axes[0, 1].set_title("Subspace alignment along each trajectory")
    axes[0, 1].legend()

    basis_values: dict[int, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        if run["method"] == "random_subspace":
            basis_values[int(run["basis_seed"])][float(run["rho"])].append(
                100 * float(run["best_score"])
            )
    markers = ("o", "s", "^")
    for (basis_seed, by_rho), marker in zip(sorted(basis_values.items()), markers):
        rhos = sorted(by_rho)
        axes[1, 0].errorbar(
            rhos,
            [np.mean(by_rho[rho]) for rho in rhos],
            yerr=[np.std(by_rho[rho], ddof=1) for rho in rhos],
            marker=marker,
            capsize=4,
            linewidth=2,
            label=f"basis {basis_seed}",
        )
    axes[1, 0].set_xticks([0.025, 0.05, 0.1], ["0.025", "0.05", "0.10"])
    axes[1, 0].set_xlabel("rho")
    axes[1, 0].set_ylabel("Best inner-dev accuracy (%)")
    axes[1, 0].set_title("E4 sensitivity to random-basis seed")
    axes[1, 0].legend()

    e4_scores: dict[tuple[float, int], list[float]] = defaultdict(list)
    for run in runs:
        if run["method"] == "random_subspace":
            e4_scores[(float(run["rho"]), int(run["train_seed"]))].append(
                100 * float(run["best_score"])
            )
    equal_rhos = [0.025, 0.05, 0.1]
    ranges = {
        rho: [
            max(values) - min(values)
            for (item_rho, _seed), values in e4_scores.items()
            if item_rho == rho
        ]
        for rho in equal_rhos
    }
    means = [np.mean(ranges[rho]) for rho in equal_rhos]
    axes[1, 1].bar(equal_rhos, means, width=0.012, color=COLORS["random_subspace"])
    for index, rho in enumerate(equal_rhos):
        offsets = np.linspace(-0.003, 0.003, len(ranges[rho]))
        axes[1, 1].scatter(
            rho + offsets,
            ranges[rho],
            color="#222222",
            s=25,
            zorder=3,
        )
    axes[1, 1].set_xticks(equal_rhos, ["0.025", "0.05", "0.10"])
    axes[1, 1].set_xlabel("rho")
    axes[1, 1].set_ylabel("Within-seed max-min accuracy (pp)")
    axes[1, 1].set_title("E4 variability across three random bases")

    fig.suptitle(
        "Stage 2 rho search: perturbation strength and basis sensitivity",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    path = output_dir / "stage2_strength_and_basis.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        f"{rho:g}": sample_stats(values) for rho, values in ranges.items()
    }


def integrity_summary(run_dir: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    resolved_configs = []
    data_summaries = []
    perturbation_errors = []
    nonfinite_values = 0
    logged_loss_increases = 0
    positive_loss_increases = 0
    for directory in sorted((run_dir / "runs").iterdir()):
        if not directory.is_dir() or not (directory / "summary.json").exists():
            continue
        config = read_json(directory / "resolved_config.json")
        data = read_json(directory / "data_summary.json")
        resolved_configs.append(config)
        data_summaries.append(data)
        rho = float(config["sam"]["rho"])
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
                if "perturbation_norm" in record:
                    perturbation_errors.append(abs(float(record["perturbation_norm"]) - rho))
                if "clean_loss" in record:
                    logged_loss_increases += 1
                    if float(record["loss"]) - float(record["clean_loss"]) > 0:
                        positive_loss_increases += 1
    method_times: dict[str, list[float]] = defaultdict(list)
    method_best_epochs: dict[str, list[float]] = defaultdict(list)
    method_stop_epochs: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        method_times[run["method"]].append(float(run["seconds_per_step"]))
        method_best_epochs[run["method"]].append(float(run["best_epoch"]))
        method_stop_epochs[run["method"]].append(float(run["epochs_completed"]))
    return {
        "all_deterministic": all(
            config["runtime"]["deterministic"] for config in resolved_configs
        ),
        "official_validation_used": any(
            data["official_validation_used"] for data in data_summaries
        ),
        "train_example_counts": sorted({data["train_examples"] for data in data_summaries}),
        "inner_dev_example_counts": sorted(
            {data["validation_examples"] for data in data_summaries}
        ),
        "max_perturbation_radius_error": max(perturbation_errors, default=0.0),
        "nonfinite_scalar_values": nonfinite_values,
        "positive_loss_increase_fraction": (
            positive_loss_increases / logged_loss_increases
            if logged_loss_increases
            else None
        ),
        "seconds_per_step": {
            method: sample_stats(values) for method, values in method_times.items()
        },
        "best_epoch": {
            method: sample_stats(values) for method, values in method_best_epochs.items()
        },
        "stop_epoch": {
            method: sample_stats(values) for method, values in method_stop_epochs.items()
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = read_json(run_dir / "stage2_aggregate.json")
    runs = aggregate["runs"]
    if len(runs) != 54:
        raise RuntimeError(f"Expected 54 completed sweep runs, found {len(runs)}")
    if any(run["validation_source"] != "inner_train" for run in runs):
        raise RuntimeError("A Stage 2 run used a non-inner validation source")
    verification = {
        method: read_json(run_dir / "determinism" / f"{method}_verification.json")
        for method in ("e4", "e5")
    }
    if not all(item["passed"] for item in verification.values()):
        raise RuntimeError("Determinism verification did not pass")

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
    paired = accuracy_figure(runs, output_dir)
    basis_ranges = strength_figure(runs, output_dir)
    grids = {
        method: summarize_grid(runs, method)
        for method in ("lora_sam", "random_subspace", "qr_svd")
    }
    summary = {
        "run_dir": str(run_dir),
        "completed_sweep_runs": len(runs),
        "determinism_passed": {method: item["passed"] for method, item in verification.items()},
        "method_grids": grids,
        "paired_accuracy_pp": paired,
        "e4_basis_range_pp": basis_ranges,
        "integrity_and_efficiency": integrity_summary(run_dir, runs),
        "figures": [
            str(output_dir / "stage2_rho_accuracy.png"),
            str(output_dir / "stage2_strength_and_basis.png"),
        ],
    }
    with (output_dir / "stage2_plot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
