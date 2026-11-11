"""Generate the joint E2-E5 figures used by the RTE experiment record.

Run from the project root:

    python analysis/plot_e2_e5_results.py

Use explicit result directories when needed:

    python analysis/plot_e2_e5_results.py \
      --e2-e5-run-dir records/e2_e5_rte20_20260921_032917 \
      --e3-e4-run-dir records/e3_e4_rte20_20260923_032434

Dependencies: matplotlib and numpy.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
METHODS = OrderedDict(
    [
        ("e2", {"label": "E2 QLoRA", "color": "#4C78A8", "marker": "o"}),
        ("e3", {"label": "E3 LoRA-SAM", "color": "#B279A2", "marker": "^"}),
        (
            "e4",
            {"label": "E4 Random-subspace SAM", "color": "#F28E2B", "marker": "D"},
        ),
        ("e5", {"label": "E5 QR-SVD-SAM", "color": "#E45756", "marker": "s"}),
    ]
)
SEEDS = [13, 21, 42, 87, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-e5-run-dir", type=Path)
    parser.add_argument("--e3-e4-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_run(pattern: str) -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "records").glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No records/{pattern} directory found")
    return candidates[0]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_runs(e2_e5_root: Path, e3_e4_root: Path) -> dict[str, dict[int, dict]]:
    roots = {"e2": e2_e5_root, "e3": e3_e4_root, "e4": e3_e4_root, "e5": e2_e5_root}
    runs: dict[str, dict[int, dict]] = {method: {} for method in METHODS}
    for method, root in roots.items():
        for seed in SEEDS:
            directory = root / f"{method}_seed{seed}"
            runs[method][seed] = {
                "dir": directory,
                "summary": read_json(directory / "summary.json"),
                "metrics": read_jsonl(directory / "metrics.jsonl"),
            }
    return runs


def scores(runs: dict[str, dict[int, dict]], method: str) -> list[float]:
    return [100 * runs[method][seed]["summary"]["best_score"] for seed in SEEDS]


def epoch_records(run: dict) -> list[dict]:
    return [record for record in run["metrics"] if record.get("type") == "epoch"]


def train_step_records(run: dict) -> list[dict]:
    return [record for record in run["metrics"] if record.get("type") == "train_step"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def paired_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    half_width = 2.776 * statistics.stdev(values) / np.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def plot_accuracy(runs: dict[str, dict[int, dict]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    x = np.arange(len(METHODS))

    ax = axes[0]
    for seed in SEEDS:
        values = [100 * runs[method][seed]["summary"]["best_score"] for method in METHODS]
        ax.plot(x, values, color="#B7BCC5", linewidth=1.1, alpha=0.8, zorder=1)
    means = []
    for index, (method, meta) in enumerate(METHODS.items()):
        values = scores(runs, method)
        means.append(statistics.mean(values))
        ax.scatter(
            np.full(len(values), index),
            values,
            color=meta["color"],
            marker=meta["marker"],
            s=45,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.errorbar(
            index,
            statistics.mean(values),
            yerr=statistics.stdev(values),
            color="black",
            marker="D",
            markerfacecolor="white",
            capsize=5,
            linewidth=1.5,
            zorder=4,
        )
        ax.text(
            index,
            statistics.mean(values) + 0.45,
            f"{statistics.mean(values):.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x, [meta["label"] for meta in METHODS.values()], rotation=12, ha="right")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_ylim(50, 66)
    ax.set_title("Five paired seeds; diamond = mean, bar = sample SD")

    ax = axes[1]
    baseline = scores(runs, "e2")
    jitter = np.linspace(-0.13, 0.13, len(SEEDS))
    for index, method in enumerate(["e3", "e4", "e5"]):
        meta = METHODS[method]
        differences = [value - base for value, base in zip(scores(runs, method), baseline)]
        ax.scatter(
            np.full(len(differences), index) + jitter,
            differences,
            color=meta["color"],
            marker=meta["marker"],
            s=48,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        mean, lower, upper = paired_ci95(differences)
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            color="black",
            marker="D",
            markerfacecolor="white",
            capsize=6,
            linewidth=1.6,
            zorder=4,
        )
        ax.text(index, 2.45, f"mean {mean:+.2f}", ha="center", va="top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(3), [METHODS[m]["label"] for m in ["e3", "e4", "e5"]], rotation=12, ha="right")
    ax.set_ylabel("Paired change from E2 (percentage points)")
    ax.set_ylim(-6, 3)
    ax.set_title("Individual seeds and exploratory paired 95% t intervals")

    fig.suptitle("RoBERTa-base, RTE 20%: E2-E5 accuracy comparison", fontsize=15, fontweight="bold")
    fig.savefig(output_dir / "e2_e5_best_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(runs: dict[str, dict[int, dict]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, seed in zip(axes, SEEDS):
        for method, meta in METHODS.items():
            records = epoch_records(runs[method][seed])
            epochs = [record["epoch"] for record in records]
            values = [100 * record["selection_score"] for record in records]
            ax.plot(
                epochs,
                values,
                color=meta["color"],
                marker=meta["marker"],
                markersize=3.2,
                linewidth=1.35,
                label=meta["label"],
            )
            summary = runs[method][seed]["summary"]
            ax.scatter(
                summary["best_epoch"],
                100 * summary["best_score"],
                color=meta["color"],
                marker="*",
                s=75,
                zorder=5,
            )
        ax.set_title(f"Seed {seed}")
        ax.set_xlim(1, 20)
        ax.set_ylim(44, 68)
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=10)
    fig.supxlabel("Epoch")
    fig.supylabel("Validation accuracy (%)")
    fig.suptitle(
        "E2-E5 validation trajectories (* = selected best checkpoint)",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_dir / "e2_e5_validation_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_and_diagnostics(
    runs: dict[str, dict[int, dict]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    methods = list(METHODS)
    x = np.arange(len(methods))

    seconds_per_step = {
        method: [
            runs[method][seed]["summary"]["elapsed_seconds"]
            / runs[method][seed]["summary"]["global_steps"]
            for seed in SEEDS
        ]
        for method in methods
    }

    ax = axes[0, 0]
    means = [statistics.mean(seconds_per_step[method]) for method in methods]
    standard_deviations = [statistics.stdev(seconds_per_step[method]) for method in methods]
    ax.bar(x, means, yerr=standard_deviations, capsize=5, color=[METHODS[m]["color"] for m in methods], alpha=0.9)
    for index, method in enumerate(methods):
        ax.scatter(np.full(len(SEEDS), index) + np.linspace(-0.12, 0.12, len(SEEDS)), seconds_per_step[method], color="black", s=14, alpha=0.7)
        ratio = means[index] / means[0]
        ax.text(index, means[index] + standard_deviations[index] + 0.008, f"{ratio:.2f}x", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, [f"E{i}" for i in range(2, 6)])
    ax.set_ylabel("Training seconds / step")
    ax.set_title("Per-step training cost (relative to E2)")

    ax = axes[0, 1]
    mean_accuracy = [statistics.mean(scores(runs, method)) for method in methods]
    for index, method in enumerate(methods):
        ax.scatter(
            means[index],
            mean_accuracy[index],
            color=METHODS[method]["color"],
            marker=METHODS[method]["marker"],
            s=95,
            label=METHODS[method]["label"],
        )
        ax.annotate(f"E{index + 2}", (means[index], mean_accuracy[index]), xytext=(6, 6), textcoords="offset points", fontweight="bold")
    ax.set_xlabel("Mean training seconds / step")
    ax.set_ylabel("Mean best accuracy (%)")
    ax.set_title("Accuracy-cost trade-off")

    ax = axes[1, 0]
    sam_methods = ["e3", "e4", "e5"]
    gaps = {}
    norms = {}
    for method in sam_methods:
        method_gaps = []
        method_norms = []
        for seed in SEEDS:
            for record in train_step_records(runs[method][seed]):
                method_gaps.append(record["loss"] - record["clean_loss"])
                method_norms.append(record["perturbation_norm"])
        gaps[method] = method_gaps
        norms[method] = method_norms
    gap_means = [statistics.mean(gaps[method]) for method in sam_methods]
    ax.bar(range(3), gap_means, color=[METHODS[m]["color"] for m in sam_methods])
    ax.set_yscale("log")
    for index, value in enumerate(gap_means):
        ax.text(index, value * 1.12, f"{value:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(3), ["E3", "E4", "E5"])
    ax.set_ylabel("Mean perturbed loss - clean loss (log scale)")
    ax.set_title("Realized SAM loss increase; mean perturbation norm = 0.050")

    ax = axes[1, 1]
    basis = read_json(runs["e5"][SEEDS[0]]["dir"] / "qr_svd_basis.json")["layers"]
    layer_values: dict[str, dict[int, float]] = {"query": {}, "value": {}}
    for record in basis:
        match = re.search(r"layer\.(\d+)\.attention\.self\.(query|value)$", record["name"])
        if match:
            layer_values[match.group(2)][int(match.group(1))] = 100 * record["captured_energy_ratio"]
    layers = list(range(12))
    for module, color, marker in [("query", "#59A14F", "o"), ("value", "#EDC948", "s")]:
        ax.plot(layers, [layer_values[module][layer] for layer in layers], color=color, marker=marker, markersize=4, label=module)
    ax.set_xticks(layers)
    ax.set_xlabel("RoBERTa encoder layer")
    ax.set_ylabel("Rank-8 residual energy captured (%)")
    ax.set_title("E5 QR-SVD basis coverage")
    ax.legend()

    fig.suptitle("E2-E5 efficiency and SAM diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(output_dir / "e2_e5_efficiency_and_basis.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    e2_e5_root = resolve_path(args.e2_e5_run_dir) if args.e2_e5_run_dir else latest_run("e2_e5_rte20_*")
    e3_e4_root = resolve_path(args.e3_e4_run_dir) if args.e3_e4_run_dir else latest_run("e3_e4_rte20_*")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(e2_e5_root, e3_e4_root)
    style()
    plot_accuracy(runs, output_dir)
    plot_validation_curves(runs, output_dir)
    plot_efficiency_and_diagnostics(runs, output_dir)
    print(f"E2/E5 source: {e2_e5_root}")
    print(f"E3/E4 source: {e3_e4_root}")
    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
