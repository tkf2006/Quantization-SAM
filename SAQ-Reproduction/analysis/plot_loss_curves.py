"""Plot epoch-level training-loss curves from the four 200-epoch experiment logs.

Usage (from any working directory):
    python analysis/plot_loss_curves.py

The figure is written next to this script as ``loss_curves_200epoch.png``.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "records"
FIGURE_PATH = ROOT / "reports" / "figures" / "loss_curves_200epoch.png"

# Display label, log filename, line style.  The three SAQ runs are distinct
# random seeds; the SGD curve is the non-SAM QAT baseline.
EXPERIMENTS = [
    ("SAQ (seed 1)", "saq_seed1/experiment.log", "-"),
    ("SAQ (seed 2)", "saq_seed2/experiment.log", "-"),
    ("SAQ (seed 3)", "saq_seed3/experiment.log", "-"),
    ("QAT-SGD (seed 1)", "qat_sgd_seed1/experiment.log", "--"),
]

EPOCH_RE = re.compile(r"Epoch: \[(\d+)\]")
TRAIN_LOSS_RE = re.compile(r"Training Error: [\d.]+ Loss: ([\d.]+)")


def parse_training_losses(log_path: Path) -> tuple[list[int], list[float]]:
    """Extract one aggregate training loss per epoch from a training log."""
    losses: dict[int, float] = {}
    current_epoch: int | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        loss_match = TRAIN_LOSS_RE.search(line)
        if loss_match and current_epoch is not None:
            losses[current_epoch] = float(loss_match.group(1))

    if len(losses) != 200:
        raise ValueError(
            f"Expected 200 epoch losses in {log_path.name}, found {len(losses)}."
        )

    epochs = sorted(losses)
    return epochs, [losses[epoch] for epoch in epochs]


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    for label, filename, linestyle in EXPERIMENTS:
        epochs, losses = parse_training_losses(OUTPUT_DIR / filename)
        ax.plot(epochs, losses, label=label, linewidth=1.8, linestyle=linestyle)

    # The logged multi-step schedule changes the learning rate after epochs 80
    # and 120; marking these points makes loss-shape changes interpretable.
    for epoch in (80, 120):
        ax.axvline(epoch, color="gray", linewidth=0.9, linestyle=":", alpha=0.8)
        ax.text(
            epoch + 0.8,
            0.97,
            "LR decay",
            color="gray",
            fontsize=9,
            transform=ax.get_xaxis_transform(),
            va="top",
        )

    ax.set_title("CIFAR-100 4-bit QAT: Training Loss over 200 Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training cross-entropy loss")
    ax.set_xlim(0, 199)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
