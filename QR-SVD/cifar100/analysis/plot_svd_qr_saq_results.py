"""Summarize and plot the completed SVD-RS-SAQ / SVD-QR-SAQ runs.

The current first-round design contains three SVD-RS-SAQ seeds and one
SVD-QR-SAQ seed.  A single-seed method is always drawn as an individual curve;
the script never fabricates a standard deviation for it.  Original SAQ runs
are included as a historical reference when ``--saq-root`` is available.

Run from the SAQ directory::

    python analysis/plot_svd_qr_saq_results.py

Figures and machine-readable CSV summaries are written to
``analysis/figures/svd_qr_saq`` by default.  Experiment output directories are
read-only.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RS_ROOT = ROOT / "records" / "svd_random"
DEFAULT_QR_ROOT = ROOT / "records" / "svd_residual"
DEFAULT_SAQ_ROOT = ROOT.parents[1] / "SAQ-Reproduction" / "records"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "figures" / "svd_qr_saq"

COLUMNS = (
    "epoch", "train_top1_error", "train_loss", "test_top1_error", "test_loss",
    "train_top5_error", "test_top5_error",
)
METHOD_ORDER = ("SAQ", "SVD-RS-SAQ", "SVD-QR-SAQ")
STYLE = {
    "SAQ": {"color": "#4C78A8"},
    "SVD-RS-SAQ": {"color": "#F58518"},
    "SVD-QR-SAQ": {"color": "#54A24B"},
}

CAPTURE_RE = re.compile(
    r"Projected QSAM basis: (?P<layer>.+?); rank=(?P<rank>\d+); "
    r"residual_norm=(?P<norm>[0-9.eE+-]+); "
    r"captured_residual_energy=(?P<ratio>[0-9.eE+-]+)"
)
GRADIENT_RE = re.compile(
    r"Projected QSAM mean weight-gradient energy ratio: (?P<ratio>[0-9.eE+-]+)"
)
TIMESTAMP_RE = re.compile(r"(?m)^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
TRAIN_TIME_RE = re.compile(r"Epoch: \[\d+\] Total time: (?P<time>\d+:\d{2}:\d{2})")
TEST_TIME_RE = re.compile(r"Test: Total time: (?P<time>\d+:\d{2}:\d{2})")
MEMORY_RE = re.compile(r"max mem: (?P<mib>[0-9.]+)")


@dataclass
class Capture:
    layer: str
    rank: int
    residual_norm: float
    ratio: float


@dataclass
class Run:
    method: str
    seed: int
    run_dir: Path
    data: dict[str, np.ndarray]
    captures: list[Capture]
    gradient_ratio: np.ndarray
    wall_hours: float
    mean_train_epoch_seconds: float
    mean_test_epoch_seconds: float
    peak_cuda_allocated_mib: float


def setting(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    match = re.search(
        rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1).strip() if match else None


def classify(opt_type: str | None) -> str | None:
    if opt_type == "SVD_RS_QSAM_SGD":
        return "SVD-RS-SAQ"
    if opt_type == "SVD_QR_QSAM_SGD":
        return "SVD-QR-SAQ"
    if opt_type == "QSAM_SGD":
        return "SAQ"
    return None


def hms_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    values = np.fromstring(path.read_text(encoding="utf-8", errors="replace"), sep=" ")
    if values.size == 0 or values.size % len(COLUMNS):
        raise ValueError(f"{path} contains {values.size} values; expected 7 per epoch")
    matrix = values.reshape(-1, len(COLUMNS))
    data = {name: matrix[:, index] for index, name in enumerate(COLUMNS)}
    data["epoch"] = data["epoch"].astype(int)
    if not np.array_equal(data["epoch"], np.arange(len(data["epoch"]))):
        raise ValueError(f"{path} has missing, duplicate, or unordered epochs")
    for split in ("train", "test"):
        for topk in ("top1", "top5"):
            data[f"{split}_{topk}"] = 100.0 - data[f"{split}_{topk}_error"]
    return data


def parse_experiment_log(path: Path) -> tuple[list[Capture], np.ndarray, float, float, float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    captures: list[Capture] = []
    seen: set[str] = set()
    for item in CAPTURE_RE.finditer(text):
        layer = item.group("layer")
        if layer not in seen:
            seen.add(layer)
            captures.append(Capture(layer, int(item.group("rank")), float(item.group("norm")), float(item.group("ratio"))))

    gradient = np.asarray([float(item.group("ratio")) for item in GRADIENT_RE.finditer(text)])
    timestamps = [datetime.strptime(item.group("time"), "%Y-%m-%d %H:%M:%S,%f") for item in TIMESTAMP_RE.finditer(text)]
    wall_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0 if len(timestamps) >= 2 else np.nan
    train_times = [hms_seconds(item.group("time")) for item in TRAIN_TIME_RE.finditer(text)]
    test_times = [hms_seconds(item.group("time")) for item in TEST_TIME_RE.finditer(text)]
    memories = [float(item.group("mib")) for item in MEMORY_RE.finditer(text)]
    return (
        captures,
        gradient,
        wall_hours,
        float(np.mean(train_times)) if train_times else np.nan,
        float(np.mean(test_times)) if test_times else np.nan,
        float(np.max(memories)) if memories else np.nan,
    )


def discover(roots: Iterable[Path]) -> list[Run]:
    runs: list[Run] = []
    for root in roots:
        if not root.is_dir():
            continue
        for log_path in sorted(root.rglob("log.txt")):
            run_dir = log_path.parent
            method = classify(setting(run_dir / "settings.log", "opt_type"))
            seed_value = setting(run_dir / "settings.log", "seed")
            experiment_log = run_dir / "experiment.log"
            if method is None or seed_value is None or not experiment_log.is_file():
                continue
            parsed = parse_experiment_log(experiment_log)
            runs.append(Run(method, int(seed_value), run_dir, read_metrics(log_path), *parsed))
    return runs


def group_and_validate(runs: list[Run]) -> dict[str, list[Run]]:
    grouped = {method: [] for method in METHOD_ORDER}
    for run in runs:
        grouped[run.method].append(run)
    expected_counts = {"SAQ": 3, "SVD-RS-SAQ": 3}
    for method in METHOD_ORDER:
        grouped[method].sort(key=lambda run: run.seed)
        seeds = [run.seed for run in grouped[method]]
        expected = expected_counts.get(method)
        count_is_invalid = len(seeds) != expected if expected is not None else not (1 <= len(seeds) <= 3)
        if count_is_invalid or len(set(seeds)) != len(seeds):
            expectation = str(expected) if expected is not None else "1--3"
            raise RuntimeError(f"{method}: expected {expectation} unique run(s), found seeds {seeds}")
        if any(len(run.data["epoch"]) != 200 for run in grouped[method]):
            raise RuntimeError(f"{method}: every completed run must contain epochs 0--199")
    expected_layers = tuple(item.layer for item in grouped["SVD-RS-SAQ"][0].captures)
    for method in ("SVD-RS-SAQ", "SVD-QR-SAQ"):
        for run in grouped[method]:
            if tuple(item.layer for item in run.captures) != expected_layers:
                raise RuntimeError(f"Diagnostic layer order differs in {run.run_dir}")
            if len(run.gradient_ratio) != 200:
                raise RuntimeError(f"Expected 200 gradient-ratio epochs in {run.run_dir}")
    return grouped


def stack(runs: list[Run], metric: str) -> np.ndarray:
    return np.vstack([run.data[metric] for run in runs])


def method_label(method: str, count: int) -> str:
    return f"{method} (n={count})"


def plot_curve(axis: plt.Axes, runs: list[Run], metric: str, method: str) -> None:
    values = stack(runs, metric)
    epochs = runs[0].data["epoch"]
    style = STYLE[method]
    for row in values:
        axis.plot(epochs, row, color=style["color"], alpha=0.22 if len(runs) > 1 else 0.95, linewidth=0.8)
    mean = values.mean(axis=0)
    axis.plot(epochs, mean, color=style["color"], linewidth=2.0, label=method_label(method, len(runs)))
    if len(runs) > 1:
        std = values.std(axis=0, ddof=1)
        axis.fill_between(epochs, mean - std, mean + std, color=style["color"], alpha=0.13)


def decorate_curve(axis: plt.Axes, ylabel: str, title: str) -> None:
    for epoch in (80, 120):
        axis.axvline(epoch, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
    axis.set(xlabel="Epoch", ylabel=ylabel, title=title, xlim=(0, 199))
    axis.grid(alpha=0.22, linewidth=0.6)


def save(figure: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_accuracy(grouped: dict[str, list[Run]], output_dir: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for method in METHOD_ORDER:
        plot_curve(axes[0], grouped[method], "test_top1", method)
        plot_curve(axes[1], grouped[method], "test_top5", method)
    decorate_curve(axes[0], "Test Top-1 accuracy (%)", "Test Top-1 accuracy")
    decorate_curve(axes[1], "Test Top-5 accuracy (%)", "Test Top-5 accuracy")
    axes[0].legend(loc="lower right")
    save(figure, output_dir, "test_accuracy_curves", dpi)


def plot_losses(grouped: dict[str, list[Run]], output_dir: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for method in METHOD_ORDER:
        plot_curve(axes[0], grouped[method], "train_loss", method)
        plot_curve(axes[1], grouped[method], "test_loss", method)
    decorate_curve(axes[0], "Training loss", "Training loss")
    decorate_curve(axes[1], "Test loss", "Test loss")
    axes[1].legend(loc="upper right")
    save(figure, output_dir, "loss_curves", dpi)


def plot_generalization(grouped: dict[str, list[Run]], output_dir: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for method in METHOD_ORDER:
        plot_curve(axes[0], grouped[method], "train_top1", method)
        gaps = stack(grouped[method], "train_top1") - stack(grouped[method], "test_top1")
        proxy = [Run(**{**run.__dict__, "data": {**run.data, "gap": row}}) for run, row in zip(grouped[method], gaps)]
        plot_curve(axes[1], proxy, "gap", method)
    decorate_curve(axes[0], "Training Top-1 accuracy (%)", "Training accuracy")
    decorate_curve(axes[1], "Train - test Top-1 (percentage points)", "Generalization gap")
    axes[1].legend(loc="upper left")
    save(figure, output_dir, "training_and_generalization_curves", dpi)


def plot_late_stability(grouped: dict[str, list[Run]], output_dir: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for index, method in enumerate(METHOD_ORDER):
        values = stack(grouped[method], "test_top1")[:, 120:200].mean(axis=1)
        offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(1)
        axis.scatter(index + offsets, values, s=46, color=STYLE[method]["color"], zorder=3)
        if len(values) > 1:
            axis.errorbar(index, values.mean(), yerr=values.std(ddof=1), color=STYLE[method]["color"], capsize=6, marker="_", markersize=18)
    axis.set_xticks(range(len(METHOD_ORDER)), [method_label(m, len(grouped[m])) for m in METHOD_ORDER])
    axis.set_ylabel("Mean Test Top-1, epochs 120--199 (%)")
    axis.set_title("Late-stage accuracy by run")
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    save(figure, output_dir, "late_stage_accuracy", dpi)


def plot_diagnostics(grouped: dict[str, list[Run]], output_dir: Path, dpi: int) -> None:
    svd_methods = ("SVD-RS-SAQ", "SVD-QR-SAQ")
    layers = [item.layer for item in grouped["SVD-RS-SAQ"][0].captures]
    x = np.arange(len(layers))
    figure, axis = plt.subplots(figsize=(13.5, 5.5))
    for method in svd_methods:
        values = np.vstack([[item.ratio for item in run.captures] for run in grouped[method]]) * 100
        for row in values:
            axis.plot(x, row, color=STYLE[method]["color"], alpha=0.22 if len(values) > 1 else 0.95, linewidth=0.8)
        mean = values.mean(axis=0)
        axis.plot(x, mean, color=STYLE[method]["color"], linewidth=2, label=method_label(method, len(grouped[method])))
        if len(values) > 1:
            std = values.std(axis=0, ddof=1)
            axis.fill_between(x, mean - std, mean + std, color=STYLE[method]["color"], alpha=0.13)
    axis.set(title="Initial residual energy captured by 8 paired SVD modes", xlabel="Quantized layer", ylabel="Captured residual energy (%)")
    axis.set_xticks(x, layers, rotation=55, ha="right", fontsize=8)
    axis.set_xlim(-0.3, len(layers) - 0.7)
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend()
    save(figure, output_dir, "initial_residual_energy_capture", dpi)

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    epochs = np.arange(200)
    for method in svd_methods:
        values = np.vstack([run.gradient_ratio for run in grouped[method]]) * 100
        for row in values:
            axis.plot(epochs, row, color=STYLE[method]["color"], alpha=0.22 if len(values) > 1 else 0.95, linewidth=0.8)
        mean = values.mean(axis=0)
        axis.plot(epochs, mean, color=STYLE[method]["color"], linewidth=2, label=method_label(method, len(grouped[method])))
        if len(values) > 1:
            std = values.std(axis=0, ddof=1)
            axis.fill_between(epochs, mean - std, mean + std, color=STYLE[method]["color"], alpha=0.13)
    decorate_curve(axis, "Projected / raw gradient energy (%)", "Projected weight-gradient energy ratio")
    axis.set_ylim(bottom=0)
    axis.legend()
    save(figure, output_dir, "projected_gradient_energy_ratio", dpi)


def plot_resources(grouped: dict[str, list[Run]], output_dir: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for index, method in enumerate(METHOD_ORDER):
        runs = grouped[method]
        times = np.asarray([run.wall_hours for run in runs])
        memories = np.asarray([run.peak_cuda_allocated_mib for run in runs])
        offsets = np.linspace(-0.08, 0.08, len(runs)) if len(runs) > 1 else np.zeros(1)
        axes[0].scatter(index + offsets, times, color=STYLE[method]["color"], s=46)
        axes[1].scatter(index + offsets, memories, color=STYLE[method]["color"], s=46)
        if len(runs) > 1:
            axes[0].errorbar(index, times.mean(), yerr=times.std(ddof=1), color=STYLE[method]["color"], capsize=5, marker="_")
            axes[1].errorbar(index, memories.mean(), yerr=memories.std(ddof=1), color=STYLE[method]["color"], capsize=5, marker="_")
    labels = [method_label(m, len(grouped[m])) for m in METHOD_ORDER]
    for axis in axes:
        axis.set_xticks(range(len(labels)), labels, rotation=12)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].set(title="Observed run wall time", ylabel="Hours")
    axes[1].set(title="Peak CUDA memory allocated", ylabel="MiB")
    save(figure, output_dir, "resource_cost", dpi)


def run_row(run: Run) -> dict[str, object]:
    test_top1 = run.data["test_top1"]
    test_top5 = run.data["test_top5"]
    best_top1_epoch = int(np.argmax(test_top1))
    best_top5_epoch = int(np.argmax(test_top5))
    return {
        "method": run.method, "seed": run.seed,
        "best_test_top1": float(test_top1[best_top1_epoch]), "best_test_top1_epoch": best_top1_epoch,
        "best_test_top5": float(test_top5[best_top5_epoch]), "best_test_top5_epoch": best_top5_epoch,
        "final_test_top1": float(test_top1[-1]), "final_test_top5": float(test_top5[-1]),
        "final_test_loss": float(run.data["test_loss"][-1]), "final_train_top1": float(run.data["train_top1"][-1]),
        "final_generalization_gap": float(run.data["train_top1"][-1] - test_top1[-1]),
        "late_test_top1_mean": float(test_top1[120:200].mean()),
        "late_test_top1_epoch_std": float(test_top1[120:200].std(ddof=1)),
        "late_test_loss_mean": float(run.data["test_loss"][120:200].mean()),
        "wall_hours": run.wall_hours, "mean_train_epoch_seconds": run.mean_train_epoch_seconds,
        "mean_test_epoch_seconds": run.mean_test_epoch_seconds, "peak_cuda_allocated_mib": run.peak_cuda_allocated_mib,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(grouped: dict[str, list[Run]], output_dir: Path) -> None:
    rows = [run_row(run) for method in METHOD_ORDER for run in grouped[method]]
    write_csv(output_dir / "run_metrics.csv", rows)

    numeric_fields = [key for key in rows[0] if key not in {"method", "seed", "best_test_top1_epoch", "best_test_top5_epoch"}]
    method_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {"method": method, "n": len(selected)}
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in selected])
            summary[f"{field}_mean"] = float(values.mean())
            summary[f"{field}_sample_std"] = float(values.std(ddof=1)) if len(values) > 1 else ""
        method_rows.append(summary)
    write_csv(output_dir / "method_summary.csv", method_rows)

    diagnostic_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    for method in ("SVD-RS-SAQ", "SVD-QR-SAQ"):
        for run in grouped[method]:
            denominator = sum(item.residual_norm ** 2 for item in run.captures)
            global_capture = sum(item.residual_norm ** 2 * item.ratio for item in run.captures) / denominator
            for item in run.captures:
                diagnostic_rows.append({"method": method, "seed": run.seed, "layer": item.layer, "rank": item.rank, "residual_norm": item.residual_norm, "captured_residual_energy": item.ratio})
            gradient_rows.append({
                "method": method, "seed": run.seed,
                "layer_macro_mean_residual_capture": float(np.mean([item.ratio for item in run.captures])),
                "global_residual_capture": global_capture,
                "gradient_ratio_epoch0": float(run.gradient_ratio[0]),
                "gradient_ratio_all_epoch_mean": float(run.gradient_ratio.mean()),
                "gradient_ratio_late_mean": float(run.gradient_ratio[120:200].mean()),
                "gradient_ratio_final": float(run.gradient_ratio[-1]),
            })
    write_csv(output_dir / "residual_capture_by_layer.csv", diagnostic_rows)
    write_csv(output_dir / "diagnostic_summary.csv", gradient_rows)

    # Exact rank-8 mode storage for this qsampreresnet20 architecture.
    shapes = [(16, 27)] + [(16, 144)] * 6 + [(32, 16), (32, 144)] + [(32, 288)] * 5 + [(64, 32), (64, 288)] + [(64, 576)] * 5 + [(100, 64)]
    factorized_elements = sum(min(8, m, n) * (m + n) for m, n in shapes)
    dense_elements = sum(min(8, m, n) * m * n for m, n in shapes)
    write_csv(output_dir / "mode_storage.csv", [{
        "quantized_layers": len(shapes), "rank": 8,
        "factorized_U_V_elements": factorized_elements,
        "factorized_fp32_mib": factorized_elements * 4 / 2**20,
        "dense_mode_elements": dense_elements,
        "dense_fp32_mib": dense_elements * 4 / 2**20,
        "factorized_vs_dense_reduction_percent": (1 - factorized_elements / dense_elements) * 100,
    }])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rs-root", type=Path, default=DEFAULT_RS_ROOT)
    parser.add_argument("--qr-root", type=Path, default=DEFAULT_QR_ROOT)
    parser.add_argument("--saq-root", type=Path, default=DEFAULT_SAQ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = group_and_validate(discover((args.saq_root, args.rs_root, args.qr_root)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_accuracy(grouped, args.output_dir, args.dpi)
    plot_losses(grouped, args.output_dir, args.dpi)
    plot_generalization(grouped, args.output_dir, args.dpi)
    plot_late_stability(grouped, args.output_dir, args.dpi)
    plot_diagnostics(grouped, args.output_dir, args.dpi)
    plot_resources(grouped, args.output_dir, args.dpi)
    write_summaries(grouped, args.output_dir)
    print("Loaded runs:")
    for method in METHOD_ORDER:
        print(f"  {method}: seeds {[run.seed for run in grouped[method]]}")
    print(f"Saved figures and CSV summaries to: {args.output_dir}")


if __name__ == "__main__":
    main()
