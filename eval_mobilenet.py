"""Bring the MobileNetV2 baseline into the Phase 3 evaluation (used by ml/evaluate.py).

The model is trained and run by ml/mobilenet_baseline.py in the TensorFlow environment; this module only reads the
held-out predictions it saved, so evaluation stays in the main environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from ml import detections as det_cache
from ml import rules

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/mobilenet"
PREDICTIONS = RESULTS / "predictions_held_out.csv"
METRICS = RESULTS / "metrics.json"
GRID_ROWS, GRID_COLS = 2, 4
COLORS = {"head": "#1c7ed6", "phone": "#e8590c"}
LABEL = {0: "not using", 1: "using"}


def load(held_out: pd.DataFrame) -> tuple[pd.DataFrame, dict] | None:
    """Predictions in held-out order plus the training summary, or None when the baseline has not been run."""
    if not PREDICTIONS.exists() or not METRICS.exists():
        return None
    table = pd.read_csv(PREDICTIONS).set_index("image_name").loc[held_out["image_name"]].reset_index()
    return table, json.loads(METRICS.read_text())


def disagreement_figure(held_out: pd.DataFrame, paths: list[Path], mobilenet: pd.DataFrame, rules_pred: list[int],
                        cache_ft: dict, cfg: dict, out: Path) -> dict:
    """8 held-out images where MobileNet and B-rules disagree: 4 where only MobileNet is right, 4 where only B-rules is.

    Within each group the images MobileNet was most sure about come first, so the grid is not cherry-picked."""
    truth = held_out["using_phone"].to_numpy()
    mob = mobilenet["pred"].to_numpy()
    prob = mobilenet["prob"].to_numpy()
    disagree = [i for i in range(len(truth)) if mob[i] != rules_pred[i]]
    only_mobilenet = sorted((i for i in disagree if mob[i] == truth[i]), key=lambda i: (-abs(prob[i] - 0.5), i))
    only_rules = sorted((i for i in disagree if rules_pred[i] == truth[i]), key=lambda i: (-abs(prob[i] - 0.5), i))
    picks = only_mobilenet[: GRID_COLS] + only_rules[: GRID_COLS]
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(3.2 * GRID_COLS, 4.3 * GRID_ROWS))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, picks):
        ax.imshow(cv2.cvtColor(cv2.imread(str(paths[i])), cv2.COLOR_BGR2RGB))
        for d in rules.filter_dets(det_cache.as_rule_dets(cache_ft[paths[i].stem]), cfg):
            x1, y1, x2, y2 = d["box"]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=COLORS[d["cls"]], lw=2))
        ax.set_title(f"label: {LABEL[truth[i]]}\nMobileNet: {LABEL[mob[i]]} (p={prob[i]:.2f})\n"
                     f"B-rules: {LABEL[rules_pred[i]]}", fontsize=8.5)
    fig.suptitle("MobileNet and B-rules disagree. Top row: MobileNet is right. Bottom row: B-rules is right. "
                 "Boxes are the detector's (orange = phone, blue = head).", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {"disagreements": len(disagree), "only_mobilenet_right": len(only_mobilenet), "only_rules_right": len(only_rules),
            "grid_images": [paths[i].name for i in picks]}
