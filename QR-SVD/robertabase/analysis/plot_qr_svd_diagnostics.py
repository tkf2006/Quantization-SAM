from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYER_PATTERN = re.compile(r"layer\.(\d+)\.attention\.self\.(query|value)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot QR-SVD residual-spectrum and subspace-alignment diagnostics"
    )
    parser.add_argument("--qr-run-dir", required=True, help="Completed E5 diagnostic run")
    parser.add_argument(
        "--output-dir",
        help="Figure directory; defaults to <qr-run-dir>/diagnostic_figures",
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def layer_key(name: str) -> tuple[int, str]:
    match = LAYER_PATTERN.search(name)
    if not match:
        raise ValueError(f"Unexpected target-layer name: {name}")
    return int(match.group(1)), match.group(2)


def mean_map(records: list[dict], path: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        node = record
        for key in path:
            node = node[key]
        for name, value in node.items():
            values[name].append(float(value))
    return {name: float(np.mean(samples)) for name, samples in values.items()}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.qr_run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "diagnostic_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    basis_layers = read_json(run_dir / "qr_svd_basis.json")["layers"]
    records = read_jsonl(run_dir / "subspace_diagnostics.jsonl")
    if not records:
        raise RuntimeError("No subspace diagnostic records were found")
    if any(layer.get("cumulative_energy_by_rank") is None for layer in basis_layers):
        raise RuntimeError(
            "This run predates full-spectrum diagnostics; rebuild the QR basis and rerun E5"
        )

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    energy_by_module: dict[str, list[list[float]]] = {"query": [], "value": []}
    ranks = sorted(
        int(rank) for rank in basis_layers[0]["cumulative_energy_by_rank"]
    )
    for layer in basis_layers:
        _index, module = layer_key(layer["name"])
        values = [100 * layer["cumulative_energy_by_rank"][str(rank)] for rank in ranks]
        energy_by_module[module].append(values)
        ax.plot(ranks, values, color="#999999", alpha=0.18, linewidth=0.8)
    for module, color in (("query", "#59A14F"), ("value", "#EDC948")):
        mean_values = np.mean(energy_by_module[module], axis=0)
        ax.plot(ranks, mean_values, marker="o", color=color, linewidth=2, label=module)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, ranks)
    ax.set_xlabel("SVD rank")
    ax.set_ylabel("Residual energy captured (%)")
    ax.set_title("Quantization-residual cumulative energy")
    ax.legend()

    primary_layer_norm = mean_map(
        records, ("primary", "per_layer_projected_gradient_norm")
    )
    comparison_names = sorted(records[0]["comparisons"])
    comparison_maps = {
        comparison: mean_map(
            records,
            ("comparisons", comparison, "per_layer_projected_gradient_norm"),
        )
        for comparison in comparison_names
    }
    random_layer_norm = {
        name: float(np.mean([mapping[name] for mapping in comparison_maps.values()]))
        for name in primary_layer_norm
    }

    for column, module in enumerate(("query", "value")):
        ax = axes[0, 1] if column == 0 else axes[1, 0]
        selected = sorted(
            (
                (layer, primary_layer_norm[name], random_layer_norm[name])
                for name in primary_layer_norm
                for layer, kind in [layer_key(name)]
                if kind == module
            ),
            key=lambda item: item[0],
        )
        layers = [item[0] for item in selected]
        ax.plot(
            layers,
            [item[1] for item in selected],
            marker="o",
            label="QR-SVD",
            color="#E15759",
        )
        ax.plot(
            layers,
            [item[2] for item in selected],
            marker="s",
            label=f"Random mean (n={len(comparison_names)})",
            color="#4E79A7",
        )
        ax.set_yscale("log")
        ax.set_xticks(layers)
        ax.set_xlabel("RoBERTa encoder layer")
        ax.set_ylabel("Mean projected-gradient norm")
        ax.set_title(f"Same-batch alignment: {module}")
        ax.legend()

    ax = axes[1, 1]
    ratios = {
        comparison: [
            record["comparisons"][comparison]["norm_ratio_to_primary"]
            for record in records
        ]
        for comparison in comparison_names
    }
    ax.boxplot(
        [ratios[name] for name in comparison_names],
        labels=[name.replace("random_seed_", "seed ") for name in comparison_names],
        showmeans=True,
    )
    ax.axhline(1.0, color="#E15759", linestyle="--", linewidth=1.3, label="QR-SVD")
    ax.set_ylabel("Random projected norm / QR projected norm")
    ax.set_title("Global same-batch alignment ratio")
    ax.legend()

    fig.suptitle("QR-SVD diagnostic report", fontsize=15, fontweight="bold")
    figure_path = output_dir / "stage1_qr_svd_diagnostics.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    energy_at_configured_rank = [
        layer["captured_energy_ratio"] for layer in basis_layers
    ]
    summary = {
        "run_dir": str(run_dir),
        "logged_steps": len(records),
        "target_layers": len(basis_layers),
        "configured_rank_mean_energy": float(np.mean(energy_at_configured_rank)),
        "primary_mean_projected_gradient_norm": float(
            np.mean([record["primary"]["projected_gradient_norm"] for record in records])
        ),
        "random_to_qr_norm_ratio_mean": {
            name: float(np.mean(values)) for name, values in ratios.items()
        },
        "figure": str(figure_path),
    }
    with (output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
