from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"random_spectrum": "#4E79A7", "qr_svd": "#E15759"}
LABELS = {
    "random_spectrum": "Random directions + QR spectrum",
    "qr_svd": "QR-SVD directions + QR spectrum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage 4/5 weighting results")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(aggregate: dict[str, Any], exponent: float, method: str, name: str):
    return aggregate["exponent_summaries"][f"{exponent:g}"][method][name]


def plot_seed_metric(
    ax,
    aggregate: dict[str, Any],
    method: str,
    name: str,
    *,
    scale: float = 1.0,
    marker: str,
) -> None:
    exponents = aggregate["singular_value_exponents"]
    values = [metric(aggregate, exponent, method, name)["values"] for exponent in exponents]
    for seed_index in range(len(values[0])):
        ax.plot(
            exponents,
            [scale * item[seed_index] for item in values],
            color=COLORS[method],
            alpha=0.18,
            linewidth=1,
        )
    ax.errorbar(
        exponents,
        [scale * metric(aggregate, exponent, method, name)["mean"] for exponent in exponents],
        yerr=[
            scale * metric(aggregate, exponent, method, name)["sample_std"]
            for exponent in exponents
        ],
        marker=marker,
        capsize=4,
        linewidth=2,
        color=COLORS[method],
        label=LABELS[method],
    )


def accuracy_figure(aggregate: dict[str, Any], output_dir: Path) -> None:
    exponents = [float(value) for value in aggregate["singular_value_exponents"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for method, marker in (("random_spectrum", "s"), ("qr_svd", "o")):
        plot_seed_metric(
            axes[0, 0],
            aggregate,
            method,
            "best_score",
            scale=100.0,
            marker=marker,
        )
    axes[0, 0].set_xticks(exponents)
    axes[0, 0].tick_params(axis="x", labelbottom=False)
    axes[0, 0].set_ylabel("Best inner-dev accuracy (%)")
    axes[0, 0].set_title("Accuracy across training seeds")
    axes[0, 0].legend(fontsize=9)

    comparisons = aggregate["paired_direction_comparisons"]
    paired_values = [
        [100 * value for value in item["qr_minus_random"]["values"]]
        for item in comparisons
    ]
    means = [float(np.mean(values)) for values in paired_values]
    stds = [float(np.std(values, ddof=1)) for values in paired_values]
    axes[0, 1].bar(
        exponents,
        means,
        yerr=stds,
        width=0.25,
        capsize=4,
        color=[COLORS["qr_svd"] if value >= 0 else "#BAB0AC" for value in means],
        alpha=0.85,
    )
    for exponent, values in zip(exponents, paired_values):
        axes[0, 1].scatter(
            exponent + np.linspace(-0.035, 0.035, len(values)),
            values,
            color="#222222",
            s=25,
            zorder=3,
        )
    axes[0, 1].axhline(0, color="#333333", linewidth=1)
    axes[0, 1].set_xticks(exponents)
    axes[0, 1].tick_params(axis="x", labelbottom=False)
    axes[0, 1].set_ylabel("Paired QR - random accuracy (pp)")
    axes[0, 1].set_title("Direction value at matched spectrum")

    effect_map = {
        (item["method"], float(item["singular_value_exponent"])): item[
            "weighted_minus_p0"
        ]
        for item in aggregate["paired_weighting_effects"]
    }
    weighted_exponents = [0.5, 1.0]
    x = np.arange(len(weighted_exponents), dtype=float)
    width = 0.32
    for offset, (raw_method, method) in zip(
        (-width / 2, width / 2),
        (("random_subspace", "random_spectrum"), ("qr_svd", "qr_svd")),
    ):
        items = [effect_map[(raw_method, exponent)] for exponent in weighted_exponents]
        axes[1, 0].bar(
            x + offset,
            [100 * item["mean"] for item in items],
            yerr=[100 * item["sample_std"] for item in items],
            width=width,
            capsize=4,
            color=COLORS[method],
            alpha=0.85,
            label=LABELS[method],
        )
        for index, item in enumerate(items):
            values = [100 * value for value in item["values"]]
            axes[1, 0].scatter(
                x[index] + offset + np.linspace(-0.035, 0.035, len(values)),
                values,
                color="#222222",
                s=20,
                zorder=3,
            )
    axes[1, 0].axhline(0, color="#333333", linewidth=1)
    axes[1, 0].set_xticks(x, ["p=1/2 vs 0", "p=1 vs 0"])
    axes[1, 0].set_ylabel("Paired accuracy change (pp)")
    axes[1, 0].set_title("Value of spectral weighting within each direction")
    axes[1, 0].legend(fontsize=8)

    random_runs = [
        run for run in aggregate["runs"] if run["method"] == "random_subspace"
    ]
    for basis_index, basis_seed in enumerate(aggregate["random_basis_seeds"]):
        means_by_p = []
        for exponent in exponents:
            values = [
                100 * run["best_score"]
                for run in random_runs
                if run["basis_seed"] == basis_seed
                and run["singular_value_exponent"] == exponent
            ]
            means_by_p.append(float(np.mean(values)))
            x_values = np.full(len(values), exponent + (basis_index - 1) * 0.025)
            x_values += np.linspace(-0.008, 0.008, len(values))
            axes[1, 1].scatter(
                x_values,
                values,
                s=18,
                alpha=0.35,
            )
        axes[1, 1].plot(
            exponents,
            means_by_p,
            marker=("o", "s", "^")[basis_index],
            linewidth=1.8,
            label=f"basis {basis_seed}",
        )
    axes[1, 1].set_xticks(exponents, ["0", "1/2", "1"])
    axes[1, 1].set_xlabel("Singular-value exponent p")
    axes[1, 1].set_ylabel("Best inner-dev accuracy (%)")
    axes[1, 1].set_title("Enhanced random-basis control")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Stage 4/5: spectral weighting and direction controls", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "stage45_weighting_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def mechanism_figure(aggregate: dict[str, Any], output_dir: Path) -> None:
    exponents = [float(value) for value in aggregate["singular_value_exponents"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for method, marker in (("random_spectrum", "s"), ("qr_svd", "o")):
        plot_seed_metric(
            axes[0, 0],
            aggregate,
            method,
            "mean_weighted_gradient_norm",
            marker=marker,
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(exponents)
    axes[0, 0].tick_params(axis="x", labelbottom=False)
    axes[0, 0].set_ylabel("Mean weighted-gradient norm")
    axes[0, 0].set_title("Weighted ascent direction before radius normalization")
    axes[0, 0].legend(fontsize=9)

    for method, marker in (("random_spectrum", "s"), ("qr_svd", "o")):
        plot_seed_metric(
            axes[0, 1],
            aggregate,
            method,
            "mean_loss_increase",
            marker=marker,
        )
    axes[0, 1].set_xticks(exponents)
    axes[0, 1].tick_params(axis="x", labelbottom=False)
    axes[0, 1].set_ylabel("Mean perturbed - clean loss")
    axes[0, 1].set_title("Effective perturbation strength at fixed rho")
    axes[0, 1].legend(fontsize=9)

    for method, marker in (("random_spectrum", "s"), ("qr_svd", "o")):
        plot_seed_metric(
            axes[1, 0],
            aggregate,
            method,
            "seconds_per_step",
            marker=marker,
        )
    axes[1, 0].set_xticks(exponents, ["0", "1/2", "1"])
    axes[1, 0].set_xlabel("Singular-value exponent p")
    axes[1, 0].set_ylabel("Seconds per training step")
    axes[1, 0].set_title("Measured training cost")
    axes[1, 0].legend(fontsize=9)

    weighting = aggregate["spectral_weighting"]
    effective = [item["effective_coordinates"] for item in weighting]
    largest_share = [100 * item["largest_coordinate_energy_share"] for item in weighting]
    axes[1, 1].plot(
        exponents,
        effective,
        marker="o",
        linewidth=2,
        color="#59A14F",
        label="Effective weighted coordinates",
    )
    axes[1, 1].set_xticks(exponents, ["0", "1/2", "1"])
    axes[1, 1].set_xlabel("Singular-value exponent p")
    axes[1, 1].set_ylabel("Effective coordinates", color="#59A14F")
    axes[1, 1].tick_params(axis="y", labelcolor="#59A14F")
    twin = axes[1, 1].twinx()
    twin.plot(
        exponents,
        largest_share,
        marker="s",
        linewidth=2,
        color="#F28E2B",
        label="Largest coordinate energy share",
    )
    twin.set_ylabel("Largest prior weight share (%)", color="#F28E2B")
    twin.tick_params(axis="y", labelcolor="#F28E2B")
    axes[1, 1].set_title("Concentration induced by the QR spectrum")

    fig.suptitle("Stage 4/5: weighting mechanism and cost", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "stage45_weighting_mechanism.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = read_json(run_dir / "stage45_aggregate.json")
    if aggregate["run_count"] != 36:
        raise RuntimeError(f"Expected 36 runs, found {aggregate['run_count']}")
    if aggregate["official_validation_used"]:
        raise RuntimeError("Official validation was used during Stage 4/5")
    if not aggregate["spectrum_verification"][
        "all_runs_share_identical_selected_spectrum"
    ]:
        raise RuntimeError("Random controls did not reuse the QR spectrum")

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
    accuracy_figure(aggregate, output_dir)
    mechanism_figure(aggregate, output_dir)
    manifest = {
        "aggregate": str(run_dir / "stage45_aggregate.json"),
        "figures": [
            str(output_dir / "stage45_weighting_accuracy.png"),
            str(output_dir / "stage45_weighting_mechanism.png"),
        ],
    }
    with (output_dir / "stage45_plot_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
