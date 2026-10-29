"""Binary classification metrics for "using phone" (positive) vs "not using" (negative). Plain Python, so the
threshold sweeps in tune_rules.py stay fast."""
from __future__ import annotations

from typing import Sequence


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics(c: dict[str, int]) -> dict[str, float]:
    """Accuracy first (the syllabus headline), then precision, recall, F1 and specificity."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "specificity": _ratio(tn, tn + fp),
    }
