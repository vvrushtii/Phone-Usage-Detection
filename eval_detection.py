"""Detection metrics on the detector's TEST split: mAP50, mAP50-95, mean IoU of true positives, mAP50 by box size.

Used by ml/evaluate.py. The COCO-pretrained model is scored on `phone` only (COCO has no head class). mAP is computed
with torchmetrics on the cached low-confidence detections; IoU is measured at each model's own operating threshold.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

IOU_MATCH = 0.5  # a detection is a true positive when it overlaps a ground-truth box at least this much
SIZE_NAMES = ("small", "medium", "large")
SIZE_LIMITS = (32**2, 96**2)  # COCO's definition, in pixels of box area


def read_ground_truth(label_file: Path, width: int, height: int) -> dict[str, list[list[float]]]:
    """Ground-truth boxes in pixels, per class. Exact duplicate boxes (a labelling slip) are counted once."""
    boxes: dict[str, set[tuple]] = {"phone": set(), "head": set()}
    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes["phone" if int(cls) == 0 else "head"].add((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return {k: [list(b) for b in sorted(v)] for k, v in boxes.items()}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between (N, 4) and (M, 4) boxes."""
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_at_threshold(preds: list[list[list[float]]], gts: list[list[list[float]]], conf: float) -> dict:
    """Greedy one-to-one matching per image (highest confidence first) at a fixed confidence threshold."""
    ious, tp, fp, fn = [], 0, 0, 0
    for p, g in zip(preds, gts):
        p = sorted((r for r in p if r[4] >= conf), key=lambda r: -r[4])
        free = np.ones(len(g), dtype=bool)
        garr = np.array(g, dtype=float).reshape(-1, 4)
        for row in p:
            if free.any():
                scores = _iou(np.array([row[:4]]), garr)[0]
                scores[~free] = -1
                j = int(scores.argmax())
                if scores[j] >= IOU_MATCH:
                    free[j] = False
                    tp += 1
                    ious.append(float(scores[j]))
                    continue
            fp += 1
        fn += int(free.sum())
    return {
        "operating_conf": conf, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
        "mean_iou_of_true_positives": float(np.mean(ious)) if ious else None,
    }


def to_tensors(preds, gts):
    p_list, t_list = [], []
    for p, g in zip(preds, gts):
        p_arr = torch.tensor(p, dtype=torch.float32).reshape(-1, 5)
        p_list.append({"boxes": p_arr[:, :4], "scores": p_arr[:, 4], "labels": torch.zeros(len(p), dtype=torch.int64)})
        t_list.append({"boxes": torch.tensor(g, dtype=torch.float32).reshape(-1, 4),
                       "labels": torch.zeros(len(g), dtype=torch.int64)})
    return p_list, t_list


def _clean(x: torch.Tensor) -> float | None:
    v = float(x)
    return None if v < 0 else v  # torchmetrics reports -1 when nothing falls in that group


def map_scores(preds, gts) -> dict:
    """mAP50, mAP50-95 and mAP50 by box size for one class (labels are irrelevant, so everything is class 0)."""
    p_list, t_list = to_tensors(preds, gts)
    full = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    at50 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[IOU_MATCH])
    for m in (full, at50):
        m.update(p_list, t_list)
    r_full, r_50 = full.compute(), at50.compute()
    areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for g in gts for b in g])
    counts = (int((areas < SIZE_LIMITS[0]).sum()), int(((areas >= SIZE_LIMITS[0]) & (areas < SIZE_LIMITS[1])).sum()),
              int((areas >= SIZE_LIMITS[1]).sum()))
    return {
        "map50": _clean(r_full["map_50"]), "map50_95": _clean(r_full["map"]),
        "map50_by_size": {n: {"map50": _clean(r_50[f"map_{n}"]), "ground_truth_boxes": c} for n, c in zip(SIZE_NAMES, counts)},
        "ground_truth_boxes": int(len(areas)),
    }


def evaluate_class(preds, gts, operating_conf: float) -> dict:
    return {**map_scores(preds, gts), **match_at_threshold(preds, gts, operating_conf)}


def plot_by_size(results: dict[str, dict], path: Path) -> None:
    """Grouped bars: phone mAP50 by box size, one bar per model."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.38
    for i, (name, res) in enumerate(results.items()):
        vals = [res["map50_by_size"][s]["map50"] or 0.0 for s in SIZE_NAMES]
        bars = ax.bar(np.arange(3) + (i - 0.5) * width, vals, width, label=name)
        for bar, s in zip(bars, SIZE_NAMES):
            n = res["map50_by_size"][s]["ground_truth_boxes"]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.2f}\n(n={n})",
                    ha="center", fontsize=8)
    ax.set_xticks(range(3), [f"{s}\n(area {lim})" for s, lim in zip(SIZE_NAMES, ("< 32²", "32²–96²", "> 96²"))])
    ax.set_ylabel("phone mAP50")
    ax.set_ylim(0, 1.12)
    ax.set_title("Phone detection by box size, test split")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
