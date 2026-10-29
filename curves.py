"""Training-curve plot for the MobileNet baseline (matplotlib only, so the report can redraw it without TensorFlow)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_curves(history: dict[str, list[float]], stage2_start: int, path: Path) -> None:
    """Loss and accuracy per epoch for train and validation; a dashed line marks where fine-tuning starts."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        if stage2_start:
            ax.axvline(stage2_start + 0.5, color="grey", linestyle="--", label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
