"""Reporting helpers for the YOLO training run, shared by the local catch-up script and the Colab notebook.

Not run directly. Used by ml/phase2_catchup.py (CPU) and ml/train_colab.ipynb (GPU), so both produce
identical tables and figures.
"""
from __future__ import annotations

import random
import statistics
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ultralytics import YOLO

PHONE, HEAD = 0, 1
COCO_PHONE_NAME = "cell phone"
N_EXAMPLES = 6
EXAMPLE_SEED = 42
PREDICT_CONF = 0.25  # Ultralytics' default; the real thresholds are tuned in Phase 3
VAL_BATCH = 16
ROW_HEIGHT_IN = 4.2
COLOR = {"phone": "#e8590c", "head": "#1c7ed6"}
AUGMENTATION_KEYS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
    "mosaic", "mixup", "cutmix", "copy_paste", "auto_augment", "erasing", "close_mosaic",
)
SETTING_KEYS = ("imgsz", "batch", "patience", "seed", "deterministic", "optimizer", "cache", "workers")


def sample_figure(images_dir: Path, labels_dir: Path, n: int = 8, seed: int = EXAMPLE_SEED):
    """Random labelled images with their ground-truth boxes: a sanity check before spending GPU time."""
    stems = [p.stem for p in sorted(images_dir.glob("*.jpg")) if (labels_dir / f"{p.stem}.txt").read_text().strip()]
    picks = random.Random(seed).sample(stems, n)
    cols = n // 2
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 7))
    for ax, stem in zip(axes.flat, picks):
        img = cv2.cvtColor(cv2.imread(str(images_dir / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)
        ax.axis("off")
        for line in (labels_dir / f"{stem}.txt").read_text().splitlines():
            cls, cx, cy, bw, bh = line.split()
            name = "phone" if int(cls) == PHONE else "head"
            ax.add_patch(plt.Rectangle(((float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h),
                                       float(bw) * w, float(bh) * h, fill=False, ec=COLOR[name], lw=2))
    fig.suptitle("Training images with their boxes (orange = phone, blue = head)", fontsize=12)
    fig.tight_layout()
    return fig


def per_class_val(best: Path, data_yaml: Path, val_images: Path, imgsz: int, device: str | int) -> dict:
    """Run a trained model over the val split and return per-class precision, recall, mAP50, mAP50-95."""
    with tempfile.TemporaryDirectory() as tmp:  # keeps Ultralytics' run folders out of the repo
        m = YOLO(best).val(
            data=str(data_yaml), split="val", imgsz=imgsz, batch=VAL_BATCH, device=device, workers=0,
            plots=False, verbose=False, project=tmp, name="val",
        )
    per_class = {}
    for i, class_id in enumerate(m.ap_class_index):  # class_result takes a position in this list, not a class id
        p, r, ap50, ap = m.class_result(i)
        per_class[m.names[class_id]] = {
            "precision": float(p), "recall": float(r), "map50": float(ap50), "map50_95": float(ap),
        }
    return {
        "split": "val",
        "imgsz": imgsz,
        "images": len(list(val_images.glob("*.jpg"))),
        "overall": {k: float(v) for k, v in m.results_dict.items()},
        "per_class": per_class,
    }


def print_table(val: dict) -> None:
    print(f"\nval split: {val['images']} images, imgsz {val['imgsz']}")
    print(f"{'class':>6} {'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}")
    for name, m in val["per_class"].items():
        print(f"{name:>6} {m['precision']:7.3f} {m['recall']:7.3f} {m['map50']:7.3f} {m['map50_95']:9.3f}")


def _read_boxes(label_file: Path) -> list[tuple[int, float]]:
    """(class, relative area) for each ground-truth box in one label file."""
    rows = [line.split() for line in label_file.read_text().splitlines() if line.strip()]
    return [(int(r[0]), float(r[3]) * float(r[4])) for r in rows]


def choose_examples(val_images: Path, val_labels: Path) -> list[str]:
    """Pick val images with a head and a small phone: the cases COCO weights are expected to miss."""
    boxes = {p.stem: _read_boxes(val_labels / f"{p.stem}.txt") for p in sorted(val_images.glob("*.jpg"))}
    median_phone = statistics.median(a for bs in boxes.values() for c, a in bs if c == PHONE)
    candidates = [
        s for s, bs in boxes.items()
        if any(c == HEAD for c, _ in bs) and any(c == PHONE for c, _ in bs)
        and min(a for c, a in bs if c == PHONE) <= median_phone
    ]
    return random.Random(EXAMPLE_SEED).sample(candidates, N_EXAMPLES)


def _detect(model: YOLO, img, imgsz: int, device: str | int, only_name: str | None) -> list[tuple[list[float], float, str]]:
    r = model.predict(img, conf=PREDICT_CONF, imgsz=imgsz, device=device, verbose=False)[0]
    out = []
    for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
        name = r.names[cls]
        if only_name is None or name == only_name:
            out.append((box, conf, "phone" if name == COCO_PHONE_NAME else name))
    return out


def before_after(
    best: Path, coco_weights: Path | str, val_images: Path, val_labels: Path, out_png: Path,
    imgsz: int, device: str | int,
) -> dict[str, int]:
    """Save COCO-pretrained vs fine-tuned detections on the same val images; return phone hits per model."""
    stems = choose_examples(val_images, val_labels)
    imgs = [cv2.imread(str(val_images / f"{s}.jpg")) for s in stems]
    ratios = [im.shape[1] / im.shape[0] for im in imgs]  # column widths follow image shapes, no wasted space
    fig, axes = plt.subplots(
        2, N_EXAMPLES, figsize=(sum(ratios) * ROW_HEIGHT_IN + 1.2, 2 * ROW_HEIGHT_IN + 0.8),
        gridspec_kw={"width_ratios": ratios},
    )
    coco, tuned = YOLO(coco_weights), YOLO(best)
    hits = {"COCO": 0, "fine-tuned": 0}
    for col, img in enumerate(imgs):
        rows = (
            ("COCO", _detect(coco, img, imgsz, device, COCO_PHONE_NAME)),
            ("fine-tuned", _detect(tuned, img, imgsz, device, None)),
        )
        for row, (label, dets) in enumerate(rows):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            hits[label] += any(n == "phone" for _, _, n in dets)
            for (x1, y1, x2, y2), conf, name in dets:
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=COLOR[name], lw=2))
                ax.text(x1, max(y1 - 3, 8), f"{name} {conf:.2f}", color=COLOR[name], fontsize=8, weight="bold",
                        bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 1})
        axes[0, col].set_title(f"val image {col + 1}", fontsize=10)
    axes[0, 0].set_ylabel("COCO yolo26n\n(cell phone only)", fontsize=11)
    axes[1, 0].set_ylabel("fine-tuned best.pt\n(phone + head)", fontsize=11)
    fig.suptitle(
        f"Before vs after fine-tuning: the same {N_EXAMPLES} val images with small phones "
        f"(both models at confidence ≥ {PREDICT_CONF})", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return hits


def build_summary(
    run_dir: Path, val: dict, *, gpu: str | None, torch_version: str | None,
    ultralytics_trained_with: str | None, notes: list[str],
) -> dict:
    """Describe a finished training run from the files it left behind. Unknown fields stay None."""
    results = pd.read_csv(run_dir / "results.csv")
    results.columns = [c.strip() for c in results.columns]
    args = yaml.safe_load((run_dir / "args.yaml").read_text())
    return {
        "model": args["model"],
        "ultralytics_trained_with": ultralytics_trained_with,
        "torch_trained_with": torch_version,
        "gpu": gpu,
        "train_hours": float(results["time"].iloc[-1]) / 3600,
        "epochs_requested": args["epochs"],
        "epochs_run": int(len(results)),
        "best_epoch": int(results["metrics/mAP50-95(B)"].idxmax()) + 1,
        "settings": {k: args[k] for k in SETTING_KEYS},
        "augmentation": {k: args[k] for k in AUGMENTATION_KEYS},
        "final_val": val,
        "notes": notes,
    }
