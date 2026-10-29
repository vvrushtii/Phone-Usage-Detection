"""Phase 2b: regenerate the Phase 2 outputs that the Colab run did not produce.

Run from the repo root:
    python ml/phase2_catchup.py

Needs   models/best.pt, data/fpi_yolo/ (val split), and in results/train/: results.csv, args.yaml,
        results.png (the files downloaded from the Colab run). phase2_summary.json is used if present.
Writes  results/train/val_per_class.json   per-class P / R / mAP50 / mAP50-95 of best.pt on val
        results/train/before_after.png     COCO yolo26n vs fine-tuned best.pt on 6 val images
        results/train/train_summary.json   what was trained, how long, final val metrics
Runs on CPU. Nothing is retrained. models/yolo26n.pt (COCO weights) is downloaded on first run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are only saved, never shown
import ultralytics  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_report as tr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "results/train"
BEST = ROOT / "models/best.pt"
COCO_WEIGHTS = ROOT / "models/yolo26n.pt"
DATA_YAML = ROOT / "data/fpi_yolo/fpi.yaml"
VAL_IMAGES = ROOT / "data/fpi_yolo/images/val"
VAL_LABELS = ROOT / "data/fpi_yolo/labels/val"
REQUIRED = [BEST, DATA_YAML, VAL_LABELS, TRAIN_DIR / "results.csv", TRAIN_DIR / "args.yaml", TRAIN_DIR / "results.png"]
DEVICE = "cpu"
COLAB_CHECK_TOLERANCE = 0.01  # CPU vs GPU, torch versions and batch size shift mAP slightly


def check_inputs() -> None:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing inputs:\n  " + "\n  ".join(missing) + "\nDownload them from the Colab run folder on Drive "
            "(MyDrive/<DRIVE_DIR>/runs/<run name>/) into results/train/, and best.pt into models/."
        )


def compare_with_colab(val: dict, colab: dict) -> dict:
    """Largest difference between the metrics Colab recorded on GPU and the ones recomputed here on CPU."""
    diffs = [
        abs(val["per_class"][cls][k] - m[k])
        for cls, m in colab.get("val_per_class", {}).items() for k in ("precision", "recall", "map50", "map50_95")
    ]
    if not diffs:
        return {"max_abs_diff": None, "within_tolerance": None, "tolerance": COLAB_CHECK_TOLERANCE}
    return {"max_abs_diff": max(diffs), "within_tolerance": max(diffs) <= COLAB_CHECK_TOLERANCE,
            "tolerance": COLAB_CHECK_TOLERANCE}


def main() -> None:
    check_inputs()
    args = yaml.safe_load((TRAIN_DIR / "args.yaml").read_text())
    colab_file = TRAIN_DIR / "phase2_summary.json"
    colab = json.loads(colab_file.read_text()) if colab_file.exists() else {}

    val = tr.per_class_val(BEST, DATA_YAML, VAL_IMAGES, args["imgsz"], DEVICE)
    (TRAIN_DIR / "val_per_class.json").write_text(json.dumps(val, indent=2))
    print("best.pt on val, CPU")
    tr.print_table(val)

    hits = tr.before_after(BEST, COCO_WEIGHTS, VAL_IMAGES, VAL_LABELS, TRAIN_DIR / "before_after.png",
                           args["imgsz"], DEVICE)
    print(f"\nbefore_after.png: phone found in {hits['COCO']}/{tr.N_EXAMPLES} images by COCO, "
          f"{hits['fine-tuned']}/{tr.N_EXAMPLES} by fine-tuned")

    notes = [
        "Trained with the first version of ml/train_colab.ipynb, before the Phase 2b notebook update.",
        f"epochs={args['epochs']} and batch={args['batch']} (fixed) were used; docs/02-train-colab.md "
        "now specifies EPOCHS=60 and AutoBatch. The model was not retrained.",
        "Optimizer was left on 'auto'; which optimizer Ultralytics actually chose is not recorded in any saved file.",
        "train_hours includes the one-off disk-cache build in epoch 1.",
        "best.pt is chosen by Ultralytics on the val split, so val is not a clean held-out set for this model.",
    ]
    summary = tr.build_summary(
        TRAIN_DIR, val, gpu=colab.get("gpu"), torch_version=colab.get("torch"),
        ultralytics_trained_with=colab.get("ultralytics"), notes=notes,
    )
    summary["val_metrics_source"] = f"recomputed locally on CPU with ultralytics {ultralytics.__version__}"
    summary["local_vs_colab_val"] = compare_with_colab(val, colab)
    (TRAIN_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))

    check = summary["local_vs_colab_val"]
    if check["max_abs_diff"] is not None:
        print(f"local vs Colab val metrics: max abs diff {check['max_abs_diff']:.4f} "
              f"({'within' if check['within_tolerance'] else 'OUTSIDE'} tolerance {check['tolerance']})")
    print("wrote val_per_class.json, before_after.png, train_summary.json in results/train/")


if __name__ == "__main__":
    main()
