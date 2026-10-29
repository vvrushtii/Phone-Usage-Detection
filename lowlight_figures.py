"""Figures for the low-light experiment (used by ml/lowlight_experiment.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ml import detections as det_cache  # noqa: E402
from ml import enhance, make_lowlight  # noqa: E402

CONDITION_LABELS = {"original": "original", "dim": "dim", "dark": "dark", "real_dark": "real dark photos"}
METHOD_LABELS = {"none": "no enhancement", "global_he": "global HE", "clahe": "CLAHE", "clahe_smooth": "CLAHE + blur"}
PICK_BRIGHTNESS = (100, 140)  # a normally lit example, so the darkened version shows what the darkening removes


def _example_path() -> Path:
    """First test image (by name) with a confident phone and a normal brightness."""
    dets = det_cache.get_detections("fine_tuned", make_lowlight.test_images())
    for path in make_lowlight.test_images():
        confident = any(row[4] >= 0.8 for row in dets[path.stem]["phone"])
        if confident and PICK_BRIGHTNESS[0] <= enhance.to_gray_mean(cv2.imread(str(path))) <= PICK_BRIGHTNESS[1]:
            return path
    raise RuntimeError("no suitable example image found")


def enhancement_examples(out: Path, params: dict) -> None:
    """One image as original / dark / dark + global HE / dark + CLAHE / dark + CLAHE + blur, with grey histograms."""
    path = _example_path()
    dark = make_lowlight.load_condition(path, "dark")
    panels = [("original", cv2.imread(str(path))), ("dark", dark),
              ("dark + global HE", enhance.enhance(dark, "global_he")),
              ("dark + CLAHE", enhance.enhance(dark, "clahe", params)),
              ("dark + CLAHE + blur", enhance.enhance(dark, "clahe_smooth", params))]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 5.6), gridspec_kw={"height_ratios": [3, 1.6]})
    for col, (title, img) in enumerate(panels):
        axes[0, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"{title}\nmean brightness {enhance.to_gray_mean(img):.0f}", fontsize=9)
        axes[0, col].axis("off")
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axes[1, col].hist(grey.ravel(), bins=64, range=(0, 255), color="#495057")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlim(0, 255)
        axes[1, col].set_xlabel("grey level", fontsize=8)
        axes[1, col].tick_params(labelsize=7)
    axes[1, 0].set_ylabel("pixels (log)", fontsize=8)
    fig.suptitle(f"Enhancing a dark image (CLAHE clip {params['clip_limit']}, tiles {tuple(params['tile_grid'])})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def map_chart(cells: dict, counts: dict, out: Path) -> None:
    """Grouped bars: mAP50 per lighting condition, one bar per enhancement method; phone on the left, head on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    conditions = list(CONDITION_LABELS)
    width = 0.2
    for ax, cls in zip(axes, ("phone", "head")):
        for i, method in enumerate(enhance.METHODS):
            vals = [cells[c][method][cls] for c in conditions]
            bars = ax.bar(np.arange(len(conditions)) + (i - 1.5) * width, vals, width, label=METHOD_LABELS[method])
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.2f}",
                        ha="center", fontsize=7)
        ax.set_xticks(range(len(conditions)), [f"{CONDITION_LABELS[c]}\n(n={counts[c]})" for c in conditions])
        ax.set_title(f"{cls} mAP50, test split")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("mAP50")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
