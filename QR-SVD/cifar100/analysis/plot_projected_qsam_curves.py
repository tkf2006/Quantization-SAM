"""Plot the SAQ, random-subspace SAQ, and QR-SAQ experiment curves.

Run from the SAQ directory after all six projected-QSAM runs finish:

    python analysis/plot_projected_qsam_curves.py

The script expects the original SAQ logs under ``output/cifar100/qresnet20/w4a4``
and the new RS-SAQ / QR-SAQ logs under ``output/cifar100/projected_qsam``.
It writes PNG and PDF figures to ``analysis/figures/projected_qsam``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTED_ROOT = ROOT / "records" / "projected_qsam"
DEFAULT_SAQ_ROOT = ROOT.parents[1] / "SAQ-Reproduction" / "records"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "figures" / "projected_qsam"

COLUMNS = (
    "epoch",
    "train_top1_error",
    "train_loss",
    "test_top1_error",
    "test_loss",
    "train_top5_error",
    "test_top5_error",
)

METHOD_STYLE = {
    "SAQ": {"label": "SAQ", "color": "#1f77b4"},
    "RS-SAQ": {"label": "RS-SAQ (rank 8)", "color": "#ff7f0e"},
    "QR-SAQ": {"label": "QR-SAQ (rank 8)", "color": "#2ca02c"},
}
METHOD_ORDER = tuple(METHOD_STYLE)


@dataclass
class Run:
    method: str
    seed: int
    run_dir: Path
    data: Dict[str, np.ndarray]


def read_setting(settings_path: Path, key: str) -> str | None:
    """Read ``key: value`` from the plain-text settings file."""
    if not settings_path.is_file():
        return None
    match = re.search(
        rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$",
        settings_path.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1).strip() if match else None


def classify_method(opt_type: str | None) -> str | None:
    if opt_type is None:
        return None
    if "RS_QSAM" in opt_type:
        return "RS-SAQ"
    if "QR_QSAM" in opt_type:
        return "QR-SAQ"
    if opt_type == "QSAM_SGD":
        return "SAQ"
    return None


def load_log(log_path: Path) -> Dict[str, np.ndarray]:
    """Load QSAM's whitespace-separated log, including its one-line variant."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values = np.fromstring(text, sep=" ")
    if values.size == 0 or values.size % len(COLUMNS) != 0:
        raise ValueError(
            f"{log_path} has {values.size} values; expected a non-zero multiple "
            f"of {len(COLUMNS)}."
        )
    matrix = values.reshape(-1, len(COLUMNS))
    data = {name: matrix[:, index] for index, name in enumerate(COLUMNS)}
    epochs = data["epoch"].astype(int)
    if not np.array_equal(epochs, np.arange(len(epochs))):
        raise ValueError(f"{log_path} has missing, duplicate, or unordered epochs.")
    data["epoch"] = epochs
    data["train_top1"] = 100.0 - data["train_top1_error"]
    data["test_top1"] = 100.0 - data["test_top1_error"]
    data["train_top5"] = 100.0 - data["train_top5_error"]
    data["test_top5"] = 100.0 - data["test_top5_error"]
    return data


def discover_runs(roots: Iterable[Path]) -> List[Run]:
    runs: List[Run] = []
    ignored: List[Path] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Experiment root does not exist: {root}")
        for log_path in sorted(root.rglob("log.txt")):
            run_dir = log_path.parent
            settings_path = run_dir / "settings.log"
            method = classify_method(read_setting(settings_path, "opt_type"))
            seed_text = read_setting(settings_path, "seed")
            if method is None or seed_text is None:
                ignored.append(run_dir)
                continue
            try:
                seed = int(seed_text)
                data = load_log(log_path)
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"Cannot load {run_dir}: {error}") from error
            runs.append(Run(method=method, seed=seed, run_dir=run_dir, data=data))

    if ignored:
        print("Ignored logs without a recognized QSAM method or seed:", file=sys.stderr)
        for run_dir in ignored:
            print(f"  {run_dir}", file=sys.stderr)
    return runs


def group_and_validate(runs: Iterable[Run]) -> Dict[str, List[Run]]:
    grouped: Dict[str, List[Run]] = {method: [] for method in METHOD_ORDER}
    for run in runs:
        if run.method in grouped:
            grouped[run.method].append(run)

    for method, method_runs in grouped.items():
        method_runs.sort(key=lambda run: run.seed)
        seeds = [run.seed for run in method_runs]
        if len(method_runs) != 3 or len(set(seeds)) != 3:
            raise RuntimeError(
                f"{method} requires exactly three distinct seed runs; found seeds {seeds}."
            )
        lengths = {len(run.data["epoch"]) for run in method_runs}
        if lengths != {200}:
            raise RuntimeError(
                f"{method} requires 200 epochs per seed; found lengths {sorted(lengths)}."
            )
    return grouped


def metric_matrix(runs: List[Run], metric: str) -> tuple[np.ndarray, np.ndarray]:
    epochs = runs[0].data["epoch"]
    values = np.vstack([run.data[metric] for run in runs])
    if not all(np.array_equal(run.data["epoch"], epochs) for run in runs[1:]):
        raise RuntimeError(f"Epoch axes differ within {runs[0].method}.")
    return epochs, values


def add_lr_markers(axis: plt.Axes) -> None:
    for epoch in (80, 120):
        axis.axvline(epoch, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
    axis.text(80, 0.02, "LR ↓", rotation=90, transform=axis.get_xaxis_transform(),
              ha="right", va="bottom", color="0.35", fontsize=8)
    axis.text(120, 0.02, "LR ↓", rotation=90, transform=axis.get_xaxis_transform(),
              ha="right", va="bottom", color="0.35", fontsize=8)


def plot_mean_curve(
    axis: plt.Axes,
    grouped: Dict[str, List[Run]],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for method in METHOD_ORDER:
        epochs, values = metric_matrix(grouped[method], metric)
        style = METHOD_STYLE[method]
        for row in values:
            axis.plot(epochs, row, color=style["color"], alpha=0.18, linewidth=0.8)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axis.plot(epochs, mean, color=style["color"], linewidth=2.0, label=style["label"])
        axis.fill_between(epochs, mean - std, mean + std, color=style["color"], alpha=0.14)

    add_lr_markers(axis)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0, 199)
    axis.grid(alpha=0.22, linewidth=0.6)


def save_figure(figure: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_test_top1(grouped: Dict[str, List[Run]], output_dir: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    plot_mean_curve(
        axis,
        grouped,
        metric="test_top1",
        ylabel="Test Top-1 accuracy (%)",
        title="CIFAR-100 W4A4: test Top-1 accuracy",
    )
    axis.legend(loc="lower right", frameon=True)
    save_figure(figure, output_dir, "test_top1_accuracy", dpi)


def plot_losses(grouped: Dict[str, List[Run]], output_dir: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    plot_mean_curve(
        axes[0],
        grouped,
        metric="train_loss",
        ylabel="Training loss",
        title="Training loss",
    )
    plot_mean_curve(
        axes[1],
        grouped,
        metric="test_loss",
        ylabel="Test loss",
        title="Test loss",
    )
    axes[1].legend(loc="upper right", frameon=True)
    save_figure(figure, output_dir, "loss_curves", dpi)


def plot_per_seed_top1(grouped: Dict[str, List[Run]], output_dir: Path, dpi: int) -> None:
    runs_by_method_seed = {
        method: {run.seed: run for run in runs} for method, runs in grouped.items()
    }
    seeds = sorted(runs_by_method_seed["SAQ"])
    figure, axes = plt.subplots(1, len(seeds), figsize=(14.0, 4.4), sharey=True)
    for axis, seed in zip(np.atleast_1d(axes), seeds):
        for method in METHOD_ORDER:
            run = runs_by_method_seed[method][seed]
            style = METHOD_STYLE[method]
            axis.plot(
                run.data["epoch"],
                run.data["test_top1"],
                color=style["color"],
                linewidth=1.5,
                label=style["label"],
            )
        add_lr_markers(axis)
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel("Epoch")
        axis.set_xlim(0, 199)
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Test Top-1 accuracy (%)")
    axes[-1].legend(loc="lower right", frameon=True)
    save_figure(figure, output_dir, "test_top1_by_seed", dpi)


def plot_late_stability(grouped: Dict[str, List[Run]], output_dir: Path, dpi: int) -> None:
    """Show each seed's mean Test Top-1 over epochs 120--199."""
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    offsets = np.array([-0.08, 0.0, 0.08])
    for index, method in enumerate(METHOD_ORDER):
        _, values = metric_matrix(grouped[method], "test_top1")
        late_values = values[:, 120:200].mean(axis=1)
        style = METHOD_STYLE[method]
        x = np.full(late_values.shape, index, dtype=float) + offsets
        axis.scatter(x, late_values, color=style["color"], s=42, zorder=3)
        axis.errorbar(
            index,
            late_values.mean(),
            yerr=late_values.std(ddof=1),
            color=style["color"],
            capsize=6,
            linewidth=1.8,
            marker="_",
            markersize=16,
            zorder=4,
        )

    axis.set_xticks(range(len(METHOD_ORDER)), [METHOD_STYLE[m]["label"] for m in METHOD_ORDER])
    axis.set_ylabel("Mean Test Top-1, epochs 120–199 (%)")
    axis.set_title("Late-stage Test Top-1 stability")
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    save_figure(figure, output_dir, "late_test_top1_stability", dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-root", type=Path, default=DEFAULT_PROJECTED_ROOT)
    parser.add_argument("--saq-root", type=Path, default=DEFAULT_SAQ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display figures after saving them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs((args.saq_root, args.projected_root))
    grouped = group_and_validate(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_test_top1(grouped, args.output_dir, args.dpi)
    plot_losses(grouped, args.output_dir, args.dpi)
    plot_per_seed_top1(grouped, args.output_dir, args.dpi)
    plot_late_stability(grouped, args.output_dir, args.dpi)

    print("Saved figures:")
    for figure_path in sorted(args.output_dir.glob("*.png")):
        print(f"  {figure_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
