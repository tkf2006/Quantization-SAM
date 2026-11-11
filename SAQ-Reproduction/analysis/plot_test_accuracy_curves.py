"""Plot test Top-1 accuracy for the four completed 200-epoch experiments.

Usage:
    python analysis/plot_test_accuracy_curves.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "records"
FIGURE_PATH = ROOT / "reports" / "figures" / "test_top1_accuracy_200epoch.png"

EXPERIMENTS = [
    ("SAQ (seed 1)", "saq_seed1/experiment.log", "-"),
    ("SAQ (seed 2)", "saq_seed2/experiment.log", "-"),
    ("SAQ (seed 3)", "saq_seed3/experiment.log", "-"),
    ("QAT-SGD (seed 1)", "qat_sgd_seed1/experiment.log", "--"),
]

EPOCH_RE = re.compile(r"Epoch: \[(\d+)\]")
TEST_TOP1_RE = re.compile(r"Testing Error: ([\d.]+) Loss: [\d.]+, Top5 Error: [\d.]+")


def parse_test_top1(log_path: Path) -> tuple[list[int], list[float]]:
    """Return the test Top-1 accuracy recorded after each training epoch."""
    scores: dict[int, float] = {}
    current_epoch: int | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        test_match = TEST_TOP1_RE.search(line)
        if test_match and current_epoch is not None:
            scores[current_epoch] = 100.0 - float(test_match.group(1))

    if len(scores) != 200:
        raise ValueError(
            f"Expected 200 test accuracies in {log_path.name}, found {len(scores)}."
        )
    epochs = sorted(scores)
    return epochs, [scores[epoch] for epoch in epochs]


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    for label, filename, linestyle in EXPERIMENTS:
        epochs, accuracies = parse_test_top1(OUTPUT_DIR / filename)
        ax.plot(epochs, accuracies, label=label, linewidth=1.7, linestyle=linestyle)

    for epoch in (80, 120):
        ax.axvline(epoch, color="gray", linewidth=0.9, linestyle=":", alpha=0.8)
        ax.text(
            epoch + 0.8,
            0.03,
            "LR decay",
            color="gray",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
            va="bottom",
        )

    ax.set_title("CIFAR-100 4-bit QAT: Test Top-1 Accuracy over 200 Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Top-1 accuracy (%)")
    ax.set_xlim(0, 199)
    ax.set_ylim(58, 70)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
