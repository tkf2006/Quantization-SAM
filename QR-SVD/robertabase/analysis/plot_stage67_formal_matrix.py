from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SETTINGS = ("rte5", "rte20", "rte100", "mrpc20", "cola20")
SETTING_LABELS = {
    "rte5": "RTE 5%\nAccuracy",
    "rte20": "RTE 20%\nAccuracy",
    "rte100": "RTE 100%\nAccuracy",
    "mrpc20": "MRPC 20%\nCombined",
    "cola20": "CoLA 20%\nMCC",
}
METHODS = ("e2", "e3", "e4", "e5")
METHOD_LABELS = {"e2": "E2 QLoRA", "e3": "E3 LoRA-SAM", "e4": "E4 Random", "e5": "E5 QR-SVD"}
COLORS = {"e2": "#4E79A7", "e3": "#F28E2B", "e4": "#59A14F", "e5": "#E15759"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(aggregate: dict[str, Any], setting: str, method: str, name: str) -> dict[str, Any]:
    return aggregate["method_summaries"][setting][method][name]


def ordered_seed_values(item: dict[str, Any]) -> list[float]:
    return [float(value) for _seed, value in sorted(item["values_by_seed"].items(), key=lambda pair: int(pair[0]))]


def performance_figure(aggregate: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    x = np.arange(len(SETTINGS), dtype=float)
    width = 0.19

    for method_index, method in enumerate(METHODS):
        offset = (method_index - 1.5) * width
        items = [metric(aggregate, setting, method, "best_score") for setting in SETTINGS]
        axes[0, 0].bar(
            x + offset,
            [100 * item["mean"] for item in items],
            yerr=[100 * item["sample_std"] for item in items],
            width=width,
            capsize=3,
            color=COLORS[method],
            alpha=0.85,
            label=METHOD_LABELS[method],
        )
        for index, item in enumerate(items):
            values = [100 * value for value in ordered_seed_values(item)]
            axes[0, 0].scatter(
                np.full(len(values), x[index] + offset) + np.linspace(-0.025, 0.025, len(values)),
                values,
                color="#222222",
                s=12,
                alpha=0.55,
                zorder=3,
            )
    axes[0, 0].set_xticks(x, [SETTING_LABELS[item] for item in SETTINGS])
    axes[0, 0].set_ylabel("Official-validation score (%)")
    axes[0, 0].set_title("Fixed configurations across five data settings")
    axes[0, 0].legend(ncol=2, fontsize=8)

    comparisons = ("e5_minus_e4", "e5_minus_e2", "e5_minus_e3")
    comparison_labels = {"e5_minus_e4": "E5 - E4", "e5_minus_e2": "E5 - E2", "e5_minus_e3": "E5 - E3"}
    comparison_colors = {"e5_minus_e4": "#59A14F", "e5_minus_e2": "#4E79A7", "e5_minus_e3": "#F28E2B"}
    comparison_width = 0.24
    for comparison_index, name in enumerate(comparisons):
        offset = (comparison_index - 1) * comparison_width
        items = [aggregate["paired_comparisons"][setting][name] for setting in SETTINGS]
        means = [item["mean_percentage_points"] for item in items]
        ci_half = [
            0.5 * (item["exploratory_95ci_percentage_points"][1] - item["exploratory_95ci_percentage_points"][0])
            for item in items
        ]
        axes[0, 1].bar(
            x + offset,
            means,
            yerr=ci_half,
            width=comparison_width,
            capsize=3,
            color=comparison_colors[name],
            alpha=0.85,
            label=comparison_labels[name],
        )
        for index, item in enumerate(items):
            values = [100 * value for value in item["values"]]
            axes[0, 1].scatter(
                np.full(len(values), x[index] + offset) + np.linspace(-0.025, 0.025, len(values)),
                values,
                color="#222222",
                s=12,
                alpha=0.55,
                zorder=3,
            )
    axes[0, 1].axhline(0, color="#333333", linewidth=1)
    axes[0, 1].set_xticks(x, [SETTING_LABELS[item].split("\n")[0] for item in SETTINGS])
    axes[0, 1].set_ylabel("Paired primary-score difference (pp)")
    axes[0, 1].set_title("Primary comparisons; error bars are exploratory 95% CIs")
    axes[0, 1].legend(fontsize=8)

    basis_items = [aggregate["e4_basis_sensitivity"][setting] for setting in SETTINGS]
    axes[1, 0].bar(
        x,
        [100 * item["mean"] for item in basis_items],
        yerr=[100 * item["sample_std"] for item in basis_items],
        color=COLORS["e4"],
        alpha=0.85,
        capsize=4,
    )
    for index, item in enumerate(basis_items):
        values = [100 * value for value in ordered_seed_values(item)]
        axes[1, 0].scatter(
            np.full(len(values), x[index]) + np.linspace(-0.05, 0.05, len(values)),
            values,
            color="#222222",
            s=18,
            zorder=3,
        )
    axes[1, 0].set_xticks(x, [SETTING_LABELS[item].split("\n")[0] for item in SETTINGS])
    axes[1, 0].set_ylabel("Within-seed E4 basis range (pp)")
    axes[1, 0].set_title("Sensitivity to the three random bases")

    counts = []
    count_labels = []
    for setting in SETTINGS:
        for name in comparisons:
            result = aggregate["paired_comparisons"][setting][name]["wins_ties_losses"]
            counts.append([result["wins"], result["ties"], result["losses"]])
            count_labels.append(f"{setting}\n{comparison_labels[name].replace('E5 - ', 'vs ')}")
    counts_array = np.asarray(counts)
    y = np.arange(len(counts), dtype=float)
    left = np.zeros(len(counts))
    for column, (label, color) in enumerate((("win", "#59A14F"), ("tie", "#BAB0AC"), ("loss", "#E15759"))):
        axes[1, 1].barh(y, counts_array[:, column], left=left, color=color, label=label)
        left += counts_array[:, column]
    axes[1, 1].set_yticks(y, count_labels, fontsize=7)
    axes[1, 1].set_xticks(range(6))
    axes[1, 1].set_xlim(0, 5)
    axes[1, 1].set_xlabel("Number of paired training seeds")
    axes[1, 1].set_title("E5 win / tie / loss consistency")
    axes[1, 1].legend(ncol=3, fontsize=8)
    axes[1, 1].invert_yaxis()

    fig.suptitle("Stage 6-7: formal confirmation and task expansion", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "stage67_formal_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def cost_figure(aggregate: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    x = np.arange(len(SETTINGS), dtype=float)
    width = 0.19

    for method_index, method in enumerate(METHODS):
        offset = (method_index - 1.5) * width
        time_items = [metric(aggregate, setting, method, "seconds_per_step") for setting in SETTINGS]
        memory_items = [metric(aggregate, setting, method, "peak_memory_allocated_gib") for setting in SETTINGS]
        axes[0, 0].bar(
            x + offset,
            [item["mean"] for item in time_items],
            yerr=[item["sample_std"] for item in time_items],
            width=width,
            capsize=3,
            color=COLORS[method],
            alpha=0.85,
            label=METHOD_LABELS[method],
        )
        axes[1, 0].bar(
            x + offset,
            [item["mean"] for item in memory_items],
            yerr=[item["sample_std"] for item in memory_items],
            width=width,
            capsize=3,
            color=COLORS[method],
            alpha=0.85,
        )
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_xticks(x, [SETTING_LABELS[item].split("\n")[0] for item in SETTINGS])
    axes[0, 0].set_ylabel("Training seconds / step")
    axes[0, 0].set_title("Measured training-loop time")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].set_ylabel("Peak allocated CUDA memory (GiB)")
    axes[1, 0].set_title("Peak memory after model and optimizer initialization")

    for method in METHODS[1:]:
        ratios = []
        overheads = []
        for setting in SETTINGS:
            base_time = metric(aggregate, setting, "e2", "seconds_per_step")["mean"]
            method_time = metric(aggregate, setting, method, "seconds_per_step")["mean"]
            base_mem = metric(aggregate, setting, "e2", "peak_memory_allocated_gib")["mean"]
            method_mem = metric(aggregate, setting, method, "peak_memory_allocated_gib")["mean"]
            ratios.append(method_time / base_time)
            overheads.append(method_mem - base_mem)
        axes[0, 1].plot(x, ratios, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABELS[method])
        axes[1, 1].plot(x, overheads, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABELS[method])
    axes[0, 1].axhline(1, color="#333333", linewidth=1)
    axes[0, 1].set_xticks(x, [SETTING_LABELS[item].split("\n")[0] for item in SETTINGS])
    axes[0, 1].set_ylabel("Time / E2 time (x)")
    axes[0, 1].set_title("Runtime multiplier relative to QLoRA")
    axes[0, 1].legend(fontsize=8)
    axes[1, 1].axhline(0, color="#333333", linewidth=1)
    axes[1, 1].set_xticks(x, [SETTING_LABELS[item].split("\n")[0] for item in SETTINGS])
    axes[1, 1].set_ylabel("Allocated-memory overhead vs E2 (GiB)")
    axes[1, 1].set_title("Additional training memory")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Stage 6-7: training cost under fixed configurations", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "stage67_formal_cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage 6-7 formal matrix results")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/param_tuning/figures"),
    )
    args = parser.parse_args()
    aggregate_path = args.run_dir / "stage67_aggregate.json"
    aggregate = read_json(aggregate_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance_figure(aggregate, args.output_dir)
    cost_figure(aggregate, args.output_dir)
    manifest = {
        "aggregate": str(aggregate_path),
        "figures": [
            "stage67_formal_performance.png",
            "stage67_formal_cost.png",
        ],
    }
    with (args.output_dir / "stage67_plot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Saved Stage 6-7 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
