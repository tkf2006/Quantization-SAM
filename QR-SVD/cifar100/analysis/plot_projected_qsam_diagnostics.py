"""Plot diagnostic curves for the rank-8 QR-SAQ and RS-SAQ experiments.

The script reads each completed run's ``experiment.log`` under
``output/cifar100/projected_qsam`` and writes two figures:

1. Layer-wise initial residual-energy capture rate.
2. Epoch-level mean projected weight-gradient energy ratio.

The gradient ratio is calculated by the trainer on every mini-batch, but the
current training code persists only its mean for each epoch.  Consequently,
the second figure is an epoch-level curve, not a per-mini-batch curve.

Example (run from the SAQ directory after downloading all six outputs)::

    python analysis/plot_projected_qsam_diagnostics.py

The script is intentionally read-only with respect to experiment outputs; it
only creates figures in ``analysis/figures/projected_qsam_diagnostics``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTED_ROOT = ROOT / "records" / "projected_qsam"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "figures" / "projected_qsam_diagnostics"

RUN_PATTERN = re.compile(r"(?P<method>qr|rs)_saq_r8_seed(?P<seed>\d+)$", re.IGNORECASE)
CAPTURE_PATTERN = re.compile(
    r"Projected QSAM basis: (?P<layer>.+?); .*?captured_residual_energy="
    r"(?P<value>[0-9.eE+-]+)"
)
GRADIENT_PATTERN = re.compile(
    r"Projected QSAM mean weight-gradient energy ratio: "
    r"(?P<value>[0-9.eE+-]+)"
)

METHODS = ("QR-SAQ", "RS-SAQ")
STYLE = {
    "QR-SAQ": {"color": "#2ca02c", "label": "QR-SAQ (rank 8)"},
    "RS-SAQ": {"color": "#ff7f0e", "label": "RS-SAQ (rank 8)"},
}


@dataclass(frozen=True)
class DiagnosticRun:
    """Diagnostics parsed from one seed run."""

    method: str
    seed: int
    run_dir: Path
    layers: tuple[str, ...]
    capture_rate: np.ndarray
    gradient_energy_ratio: np.ndarray


def parse_run(log_path: Path) -> DiagnosticRun | None:
    """Parse a completed QR-SAQ or RS-SAQ run; ignore unrelated directories."""
    match = RUN_PATTERN.search(log_path.parent.name)
    if match is None:
        return None

    method = "QR-SAQ" if match.group("method").lower() == "qr" else "RS-SAQ"
    text = log_path.read_text(encoding="utf-8", errors="replace")

    capture_by_layer: dict[str, float] = {}
    layer_order: list[str] = []
    for item in CAPTURE_PATTERN.finditer(text):
        layer = item.group("layer")
        # The basis is initialized once. Keep the first report defensively in
        # case a concatenated log contains a restarted process.
        if layer not in capture_by_layer:
            capture_by_layer[layer] = float(item.group("value"))
            layer_order.append(layer)

    gradient_values = [
        float(item.group("value")) for item in GRADIENT_PATTERN.finditer(text)
    ]
    if not layer_order:
        raise RuntimeError(f"No initial residual-energy records found in {log_path}")
    if not gradient_values:
        raise RuntimeError(f"No projected gradient-energy records found in {log_path}")

    return DiagnosticRun(
        method=method,
        seed=int(match.group("seed")),
        run_dir=log_path.parent,
        layers=tuple(layer_order),
        capture_rate=np.asarray([capture_by_layer[layer] for layer in layer_order]),
        gradient_energy_ratio=np.asarray(gradient_values),
    )


def discover_runs(projected_root: Path) -> dict[str, list[DiagnosticRun]]:
    """Discover the three QR-SAQ and three RS-SAQ runs and validate alignment."""
    if not projected_root.is_dir():
        raise FileNotFoundError(f"Projected-QSAM output directory not found: {projected_root}")

    grouped: dict[str, list[DiagnosticRun]] = {method: [] for method in METHODS}
    for log_path in sorted(projected_root.rglob("experiment.log")):
        run = parse_run(log_path)
        if run is not None:
            grouped[run.method].append(run)

    for method in METHODS:
        runs = sorted(grouped[method], key=lambda run: run.seed)
        grouped[method] = runs
        seeds = [run.seed for run in runs]
        if len(runs) != 3 or len(set(seeds)) != 3:
            raise RuntimeError(
                f"{method} requires exactly three completed seed runs; found {seeds}."
            )

    expected_layers = grouped[METHODS[0]][0].layers
    expected_epochs = len(grouped[METHODS[0]][0].gradient_energy_ratio)
    for method in METHODS:
        for run in grouped[method]:
            if run.layers != expected_layers:
                raise RuntimeError(
                    f"Layer ordering differs in {run.run_dir}; diagnostics cannot be aligned."
                )
            if len(run.gradient_energy_ratio) != expected_epochs:
                raise RuntimeError(
                    f"Epoch record count differs in {run.run_dir}; expected "
                    f"{expected_epochs}, got {len(run.gradient_energy_ratio)}."
                )
    return grouped


def mean_and_std(runs: list[DiagnosticRun], attribute: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stacked seed values and their mean/sample standard deviation."""
    values = np.vstack([getattr(run, attribute) for run in runs])
    return values, values.mean(axis=0), values.std(axis=0, ddof=1)


def plot_capture_rate(grouped: dict[str, list[DiagnosticRun]], output_dir: Path, dpi: int) -> None:
    """Plot per-layer residual energy retained by the initialized subspace."""
    layers = grouped["QR-SAQ"][0].layers
    x = np.arange(len(layers))
    figure, axis = plt.subplots(figsize=(13.5, 5.4))

    for method in METHODS:
        values, mean, std = mean_and_std(grouped[method], "capture_rate")
        style = STYLE[method]
        for row in values:
            axis.plot(x, row * 100.0, color=style["color"], alpha=0.20, linewidth=0.8)
        axis.plot(x, mean * 100.0, color=style["color"], linewidth=2.2, label=style["label"])
        axis.fill_between(
            x, (mean - std) * 100.0, (mean + std) * 100.0,
            color=style["color"], alpha=0.14,
        )

    axis.set_title("Initial quantization-residual energy captured by the rank-8 subspace")
    axis.set_xlabel("Quantized layer")
    axis.set_ylabel("Captured residual energy (%)")
    axis.set_xticks(x, layers, rotation=55, ha="right", fontsize=8)
    axis.set_xlim(-0.3, len(layers) - 0.7)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(loc="upper right", frameon=True)
    figure.tight_layout()
    figure.savefig(output_dir / "initial_residual_energy_capture_by_layer.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "initial_residual_energy_capture_by_layer.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_gradient_energy_ratio(grouped: dict[str, list[DiagnosticRun]], output_dir: Path, dpi: int) -> None:
    """Plot epoch means of the per-mini-batch projected-gradient energy ratio."""
    epochs = np.arange(len(grouped["QR-SAQ"][0].gradient_energy_ratio))
    figure, axis = plt.subplots(figsize=(9.0, 5.2))

    for method in METHODS:
        values, mean, std = mean_and_std(grouped[method], "gradient_energy_ratio")
        style = STYLE[method]
        for row in values:
            axis.plot(epochs, row * 100.0, color=style["color"], alpha=0.20, linewidth=0.8)
        axis.plot(epochs, mean * 100.0, color=style["color"], linewidth=2.2, label=style["label"])
        axis.fill_between(
            epochs, (mean - std) * 100.0, (mean + std) * 100.0,
            color=style["color"], alpha=0.14,
        )

    for epoch in (80, 120):
        axis.axvline(epoch, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
    axis.set_title("Epoch mean of projected weight-gradient energy ratio")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Projected / raw weight-gradient energy (%)")
    axis.set_xlim(0, len(epochs) - 1)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(loc="upper right", frameon=True)
    figure.tight_layout()
    figure.savefig(output_dir / "projected_gradient_energy_ratio_by_epoch.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "projected_gradient_energy_ratio_by_epoch.pdf", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-root", type=Path, default=DEFAULT_PROJECTED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = discover_runs(args.projected_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_capture_rate(grouped, args.output_dir, args.dpi)
    plot_gradient_energy_ratio(grouped, args.output_dir, args.dpi)
    print(f"Saved diagnostic figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
