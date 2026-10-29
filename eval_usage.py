"""Usage classification on the HELD-OUT split of label.csv, plus failure analysis of the rules.

Used by ml/evaluate.py. Methods compared on the same images:
    A-coco       COCO-pretrained yolo26n: any phone detected -> using
    A-ft         fine-tuned detector:     any phone detected -> using
    A2-authors   fine-tuned detector, the FPI-Det authors' own decision logic (simple_classify_images.py)
    B-rules      fine-tuned detector + ml/rules.py (head-phone geometry)
    C-mobilenet  MobileNetV2 classifier on the whole image (ml/mobilenet_baseline.py, predictions read from disk)
Each "any phone" baseline uses the phone confidence that worked best on the tune split, so the comparison is fair.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml import detections as det_cache
from ml import rules
from ml.usage_metrics import confusion, metrics

ROOT = Path(__file__).resolve().parents[1]
AUTHORS_DIR = ROOT / "third_party/FPI-Det/classification_task"
METHODS = ("A-coco", "A-ft", "A2-authors", "B-rules", "C-mobilenet")  # C-mobilenet is added when Phase 4 has run
# The authors' script looks for pure red (heads) and pure blue (phones) pixels, in OpenCV's BGR order.
AUTHORS_HEAD_BGR, AUTHORS_PHONE_BGR = (0, 0, 255), (255, 0, 0)
AUTHORS_USING = 0  # what their classify_image returns when it finds a head and a phone
GRID_ROWS, GRID_COLS = 3, 4
COLORS = {"head": "#1c7ed6", "phone": "#e8590c"}
# how close our own metrics must be to the authors' calculate_metrics.py output (it prints 4 decimals)
AUTHORS_TOLERANCE = 1e-4


# --- the authors' decision logic -------------------------------------------------------------------

def _load_authors_module():
    spec = importlib.util.spec_from_file_location("fpi_simple_classify", AUTHORS_DIR / "simple_classify_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # only defines functions; its main block is guarded
    return module


def _draw_for_authors(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    canvas = img.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), AUTHORS_HEAD_BGR if d["cls"] == "head" else AUTHORS_PHONE_BGR, 2)
    return canvas


def authors_predictions(paths: list[Path], dets_per_image: list[list[dict]], tmp: Path) -> list[int]:
    """Run the authors' classify_image on images with our boxes drawn in their colours; 1 = using phone."""
    module = _load_authors_module()
    blank = np.full((200, 200, 3), 128, dtype=np.uint8)
    probe = tmp / "probe.png"
    for dets, expect in (([], 1), ([{"cls": "head", "box": [20, 20, 80, 80]}], 1),
                         ([{"cls": "head", "box": [20, 20, 80, 80]}, {"cls": "phone", "box": [100, 100, 150, 150]}], 0)):
        cv2.imwrite(str(probe), _draw_for_authors(blank, dets))  # PNG: lossless, so the pure colours survive
        assert module.classify_image(str(probe)) == expect, "adapter for the authors' script does not work"
    out = []
    for path, dets in zip(paths, dets_per_image):
        cv2.imwrite(str(probe), _draw_for_authors(cv2.imread(str(path)), dets))
        out.append(int(module.classify_image(str(probe)) == AUTHORS_USING))
    return out


def authors_metrics(y_true: list[int], y_pred: list[int], names: list[str], tmp: Path) -> dict[str, float]:
    """The authors' calculate_metrics.py, run unchanged on our predictions (it reads label.csv and 8x.csv)."""
    work = tmp / "authors_metrics"
    work.mkdir(exist_ok=True)
    shutil.copy(AUTHORS_DIR / "calculate_metrics.py", work)
    pd.DataFrame({"image_name": names, "class_id": y_true}).to_csv(work / "label.csv", index=False)
    pd.DataFrame({"image_name": names, "class_id": y_pred}).to_csv(work / "8x.csv", index=False)
    run = subprocess.run([sys.executable, "calculate_metrics.py"], cwd=work, capture_output=True, text=True, check=True)
    found = {k.lower(): float(v) for k, v in re.findall(r"^(\w[\w ]*): ([0-9.]+)$", run.stdout, re.M)}
    found["f1"] = found.pop("f1 score")  # their label for it
    return found


# --- predictions ---------------------------------------------------------------------------------

def predict_methods(
    held_out: pd.DataFrame, cache_ft: dict, cache_coco: dict, cfg: dict, baselines: dict, tmp: Path,
) -> tuple[dict[str, list[int]], list[Path]]:
    paths = [det_cache.IMAGES / r.split / r.image_name for r in held_out.itertuples()]
    ft = [det_cache.as_rule_dets(cache_ft[p.stem]) for p in paths]
    coco = [det_cache.as_rule_dets(cache_coco[p.stem]) for p in paths]

    def any_phone(dets: list[list[dict]], conf: float) -> list[int]:
        return [int(any(d["cls"] == "phone" and d["conf"] >= conf for d in ds)) for ds in dets]

    pred = {
        "A-coco": any_phone(coco, baselines["A-coco"]["phone_conf"]),
        "A-ft": any_phone(ft, baselines["A-ft"]["phone_conf"]),
        "A2-authors": authors_predictions(paths, [rules.filter_dets(ds, cfg) for ds in ft], tmp),
        "B-rules": [int(rules.classify_image(ds, cfg)["using"]) for ds in ft],
    }
    return pred, paths


# --- reporting ---------------------------------------------------------------------------------

def plot_confusion(c: dict[str, int], title: str, path: Path) -> None:
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    pct = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center",
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["not using", "using"])
    ax.set_yticks([0, 1], ["not using", "using"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def classification_report(
    held_out: pd.DataFrame, pred: dict[str, list[int]], cfg: dict, baselines: dict, out_dir: Path, tmp: Path,
    extra_thresholds: dict[str, dict] | None = None,
) -> dict:
    y = held_out["using_phone"].tolist()
    names = held_out["image_name"].tolist()
    thresholds = {
        "A-coco": {"phone_conf": baselines["A-coco"]["phone_conf"]},
        "A-ft": {"phone_conf": baselines["A-ft"]["phone_conf"]},
        "A2-authors": {"phone_conf": cfg["conf"]["phone"], "head_conf": cfg["conf"]["head"]},
        "B-rules": cfg,
        **(extra_thresholds or {}),
    }
    report = {}
    for method in (m for m in METHODS if m in pred):
        c = confusion(y, pred[method])
        ours = metrics(c)
        theirs = authors_metrics(y, pred[method], names, tmp)
        gap = max(abs(ours[k] - theirs[k]) for k in ("accuracy", "precision", "recall", "f1"))
        report[method] = {
            "thresholds": thresholds[method], "confusion": c, "metrics": ours, "authors_calculate_metrics": theirs,
            "max_gap_to_authors_script": gap, "agrees_with_authors_script": bool(gap <= AUTHORS_TOLERANCE),
        }
        plot_confusion(c, f"{method} (held-out, n={len(y)})", out_dir / f"confusion_{method}.png")
    return report


# --- failure analysis of B-rules ---------------------------------------------------------------------

def failure_analysis(held_out: pd.DataFrame, paths: list[Path], cache_ft: dict, cfg: dict, out_dir: Path) -> dict:
    """Why B-rules is wrong, in numbers, and two grids of the most confident mistakes."""
    fps, fns = [], []
    fn_kind: Counter = Counter()
    fp_kind: Counter = Counter()
    for row, path in zip(held_out.itertuples(), paths):
        dets = rules.filter_dets(det_cache.as_rule_dets(cache_ft[path.stem]), cfg)
        result = rules.classify_image(dets, cfg)
        phone_conf = {tuple(d["box"]): d["conf"] for d in dets if d["cls"] == "phone"}
        n_heads = sum(d["cls"] == "head" for d in dets)
        if result["using"] and row.using_phone == 0:
            states = [p["state"] for p in result["persons"] if p["state"] != rules.FOCUSED]
            fp_kind["ON_CALL" if rules.ON_CALL in states else "ON_PHONE"] += 1
            score = max(phone_conf[tuple(b)] for p in result["persons"] if p["state"] != rules.FOCUSED for b in p["phones"])
            fps.append((score, path, dets, "/".join(sorted(set(states)))))
        elif not result["using"] and row.using_phone == 1:
            if not phone_conf:
                fn_kind["no phone detected"] += 1
            elif n_heads == 0:
                fn_kind["phone found, no head detected"] += 1
            else:
                fn_kind["head and phone found, rules say not using"] += 1
            fns.append((max(phone_conf.values(), default=0.0), path, dets, "FOCUSED"))
    for kind, items, name in (("fp", fps, "failures_fp.png"), ("fn", fns, "failures_fn.png")):
        items.sort(key=lambda t: (-t[0], t[1].name))
        _plot_failures(items[: GRID_ROWS * GRID_COLS], kind, out_dir / name)
    return {
        "false_positives": len(fps), "false_negatives": len(fns),
        "false_positive_by_state": dict(fp_kind), "false_negative_by_cause": dict(fn_kind),
        "grid_ranking": "false positives: confidence of the phone that triggered 'using'; "
                        "false negatives: confidence of the most confident phone detected (0 if none)",
        "fp_grid_images": [p.name for _, p, _, _ in sorted(fps, key=lambda t: (-t[0], t[1].name))[:GRID_ROWS * GRID_COLS]],
        "fn_grid_images": [p.name for _, p, _, _ in sorted(fns, key=lambda t: (-t[0], t[1].name))[:GRID_ROWS * GRID_COLS]],
    }


def _plot_failures(items: list, kind: str, path: Path) -> None:
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(3 * GRID_COLS, 3.6 * GRID_ROWS))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (score, img_path, dets, state) in zip(axes.flat, items):
        ax.imshow(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB))
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=COLORS[d["cls"]], lw=2))
        ax.set_title(f"predicted {state}\nphone conf {score:.2f}", fontsize=9)
    what = "predicted using, label says not using" if kind == "fp" else "predicted not using, label says using"
    fig.suptitle(f"B-rules mistakes ({what}); orange = phone, blue = head", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
