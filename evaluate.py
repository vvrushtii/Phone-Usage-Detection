"""Regenerate everything in results/eval/ from the saved model, the tuned rules and the splits.

Run from the repo root, after `python ml/tune_rules.py`:
    python ml/evaluate.py              (add --refresh to rerun the detectors instead of using data/cache/)

Writes detection.json, detection_by_size.png, classification.json, confusion_<method>.png, failures_fp.png,
failures_fn.png, speed.json and summary.md. When Phase 4 has been run it also adds the low-light and MobileNet
sections and the C-mobilenet row, and writes results/mobilenet/examples.png. Detection is scored on the detector's TEST split; usage classification on
the HELD-OUT part of label.csv. Nothing here changes shared/rules.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_cpu_info  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import detections as det_cache  # noqa: E402
from ml import eval_detection as ed  # noqa: E402
from ml import eval_mobilenet as em  # noqa: E402
from ml import eval_usage as eu  # noqa: E402
from ml import summary_sections  # noqa: E402
from ml.report_values import rendered_note  # noqa: E402
from ml import rules, splits  # noqa: E402
from ml.tune_rules import GRID  # noqa: E402

OUT = ROOT / "results/eval"
TEST_IMAGES = det_cache.IMAGES / "test"
TEST_LABELS = ROOT / "data/fpi_yolo/labels/test"
SPEED_WARMUP, SPEED_RUNS = 20, 200
FAILURE_NOTES = OUT / "failure_notes.md"  # written by hand after looking at the failure grids; kept across runs


def load_inputs() -> tuple[dict, dict]:
    """The tuned rules and baseline thresholds, after checking that rules.json is still what tuning produced."""
    summary_path = OUT / "tuning_summary.json"
    if not summary_path.exists() or not (OUT / "baseline_thresholds.json").exists():
        raise SystemExit("Run `python ml/tune_rules.py` first: there are no tuned thresholds yet.")
    cfg = rules.load_config()
    tuned = json.loads(summary_path.read_text())["best"]
    changed = [f"{s}.{k}" for s, k in GRID if abs(cfg[s][k] - tuned[f"{s}.{k}"]) > 1e-9]
    if changed:
        raise SystemExit(f"shared/rules.json no longer matches the tuning result for {changed}. Rerun tune_rules.py.")
    return cfg, json.loads((OUT / "baseline_thresholds.json").read_text())


def detection_report(cfg: dict, baselines: dict, cache_ft: dict, cache_coco: dict) -> dict:
    stems = sorted(p.stem for p in TEST_IMAGES.glob("*.jpg"))
    gts = [ed.read_ground_truth(TEST_LABELS / f"{s}.txt", *cache_ft[s]["size"]) for s in stems]
    phone_gt, head_gt = [g["phone"] for g in gts], [g["head"] for g in gts]
    res = {
        "split": "test", "images": len(stems),
        "phone": {
            "yolo26n_coco": ed.evaluate_class([cache_coco[s]["phone"] for s in stems], phone_gt,
                                              baselines["A-coco"]["phone_conf"]),
            "fine_tuned": ed.evaluate_class([cache_ft[s]["phone"] for s in stems], phone_gt, cfg["conf"]["phone"]),
        },
        "head": {"fine_tuned": ed.evaluate_class([cache_ft[s]["head"] for s in stems], head_gt, cfg["conf"]["head"]),
                 "yolo26n_coco": "not applicable: COCO has no head class"},
    }
    ed.plot_by_size(res["phone"], OUT / "detection_by_size.png")
    return res


def measure_speed() -> dict:
    """Mean time for one 640 px image on this CPU: preprocessing, inference and postprocessing, batch of one."""
    paths = sorted(TEST_IMAGES.glob("*.jpg"))[: SPEED_WARMUP + SPEED_RUNS]
    images = [cv2.imread(str(p)) for p in paths]  # read first so disk speed is not part of the timing
    out = {"cpu": get_cpu_info(), "images_timed": SPEED_RUNS, "warmup_runs": SPEED_WARMUP, "imgsz": det_cache.IMGSZ}
    for key in ("coco", "fine_tuned"):
        model = YOLO(det_cache.WEIGHTS[key])
        for img in images[:SPEED_WARMUP]:
            model.predict(img, imgsz=det_cache.IMGSZ, device="cpu", verbose=False)
        times = []
        for img in images[SPEED_WARMUP:]:
            t0 = time.perf_counter()
            model.predict(img, imgsz=det_cache.IMGSZ, device="cpu", verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        out[key] = {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
                    "p95_ms": float(np.percentile(times, 95))}
    return out


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_summary(cfg: dict, baselines: dict, det: dict, cls: dict, fail: dict, speed: dict, split_info: dict,
                  mobilenet_figure: dict | None) -> None:
    h = split_info["held_out"]
    lines = [
        "# Evaluation summary", "",
        f"Detection is scored on the detector's **test** split ({det['images']} images). Usage classification is scored on "
        f"the **held-out** part of label.csv ({h['images']} images, {h['using_phone']} using a phone, "
        f"{h['using_rate']:.1%}). Thresholds were chosen on the separate tune split ({split_info['tune']['images']} images).", "",
        "## Detection (test split)", "",
        "| Model | Class | mAP50 | mAP50-95 | Mean IoU of true positives | Precision | Recall | At confidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, cls_name, r in (("COCO yolo26n", "phone", det["phone"]["yolo26n_coco"]),
                               ("Fine-tuned", "phone", det["phone"]["fine_tuned"]),
                               ("Fine-tuned", "head", det["head"]["fine_tuned"])):
        lines.append(f"| {model} | {cls_name} | {_f(r['map50'])} | {_f(r['map50_95'])} | {_f(r['mean_iou_of_true_positives'])} | "
                     f"{_f(r['precision'])} | {_f(r['recall'])} | {r['operating_conf']} |")
    lines += ["", "COCO has no head class, so only the fine-tuned model is scored on heads. Precision, recall and IoU "
              "are measured at each model's tuned confidence; mAP uses all detections.", "",
              "### Phone mAP50 by box size", "", "| Model | Small (< 32²) | Medium (32²–96²) | Large (> 96²) |", "|---|---|---|---|"]
    for model, r in (("COCO yolo26n", det["phone"]["yolo26n_coco"]), ("Fine-tuned", det["phone"]["fine_tuned"])):
        cells = [f"{_f(r['map50_by_size'][s]['map50'])} (n={r['map50_by_size'][s]['ground_truth_boxes']})"
                 for s in ed.SIZE_NAMES]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += ["", f"## Usage classification (held-out, n={h['images']})", "",
              "| Method | Accuracy | Precision | Recall | F1 | Specificity | Thresholds |", "|---|---|---|---|---|---|---|"]
    for m in (m for m in eu.METHODS if m in cls):
        r = cls[m]
        thr = "rules.json" if m == "B-rules" else ", ".join(f"{k} {v}" for k, v in r["thresholds"].items())
        met = r["metrics"]
        lines.append(f"| {m} | {met['accuracy']:.3f} | {met['precision']:.3f} | {met['recall']:.3f} | {met['f1']:.3f} | "
                     f"{met['specificity']:.3f} | {thr} |")
    agree = all(r["agrees_with_authors_script"] for r in cls.values())
    lines += [
        "", "Positive class = using phone. Every number was also recomputed with the FPI-Det authors' "
        f"`calculate_metrics.py`; {f'all {len(cls)} methods agree' if agree else 'THESE DISAGREE, check classification.json'}.",
        "`A2-authors` uses the authors' decision logic (a head and a phone both found = using) on our detections. Their script "
        "returns 0 for \"using\" while label.csv marks \"using\" as 1 (checked against the boxes in Phase 1), so their numeric "
        "encoding is flipped here; the logic is unchanged. label.csv has no behaviour categories, so there is no per-category recall.",
    ]
    ft, rl = cls["A-ft"]["metrics"], cls["B-rules"]["metrics"]
    gate = rl["accuracy"] > ft["accuracy"] and rl["f1"] > ft["f1"]
    lines += ["", f"**Sanity check (B-rules must beat A-ft on accuracy and F1): {'PASS' if gate else 'FAIL'}** "
              f"(accuracy {rl['accuracy']:.3f} vs {ft['accuracy']:.3f}, F1 {rl['f1']:.3f} vs {ft['f1']:.3f}).", "",
              f"## Speed (CPU: {speed['cpu']}, {speed['imgsz']} px, batch of 1, mean of {speed['images_timed']} images after "
              f"{speed['warmup_runs']} warm-up runs)", "", "| Model | Mean ms | Median ms | 95th percentile ms |", "|---|---|---|---|"]
    for name, key in (("COCO yolo26n", "coco"), ("Fine-tuned", "fine_tuned")):
        s = speed[key]
        lines.append(f"| {name} | {s['mean_ms']:.1f} | {s['median_ms']:.1f} | {s['p95_ms']:.1f} |")
    lines += ["", "## Failure analysis of B-rules (held-out)", "",
              f"- False positives: {fail['false_positives']} ({fail['false_positive_by_state']}).",
              f"- False negatives: {fail['false_negatives']}, by cause: " + "; ".join(
                  f"{k}: {v}" for k, v in sorted(fail["false_negative_by_cause"].items(), key=lambda kv: -kv[1])) + ".",
              f"- Grids (`failures_fp.png`, `failures_fn.png`) show the most confident mistakes. Ranking: {fail['grid_ranking']}.", ""]
    if FAILURE_NOTES.exists():
        lines += ["### Observations", "", rendered_note("results/eval/failure_notes.md"), ""]
    lines += summary_sections.lowlight_lines() + summary_sections.mobilenet_lines(cls, mobilenet_figure) + summary_sections.app_speed_lines()
    lines += ["## Caveats", "",
              "- `best.pt` was chosen on the detector's val split, and about 42% of label.csv are val images, so held-out numbers "
              "are slightly optimistic.",
              "- FPI-Det is mostly surveillance-angle crops; results say little about a frontal laptop webcam.", ""]
    (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="rerun the detectors instead of using data/cache/")
    refresh = parser.parse_args().refresh

    cfg, baselines = load_inputs()
    held_out = splits.load_split("held_out")
    split_info = json.loads((splits.SPLIT_DIR / "summary.json").read_text())

    images = det_cache.label_csv_images()
    cache_ft = det_cache.get_detections("fine_tuned", images, refresh=refresh)
    cache_coco = det_cache.get_detections("coco", images, refresh=refresh)

    det = detection_report(cfg, baselines, cache_ft, cache_coco)
    (OUT / "detection.json").write_text(json.dumps(det, indent=2))
    print("detection done")

    with tempfile.TemporaryDirectory() as tmp:
        pred, paths = eu.predict_methods(held_out, cache_ft, cache_coco, cfg, baselines, Path(tmp))
        mobilenet = em.load(held_out)
        extra = {}
        if mobilenet is not None:
            pred["C-mobilenet"] = mobilenet[0]["pred"].astype(int).tolist()
            extra["C-mobilenet"] = {"threshold": mobilenet[1]["threshold"]}
        cls = eu.classification_report(held_out, pred, cfg, baselines, OUT, Path(tmp), extra)
    mobilenet_figure = None if mobilenet is None else em.disagreement_figure(
        held_out, paths, mobilenet[0], pred["B-rules"], cache_ft, cfg, em.RESULTS / "examples.png")
    if mobilenet_figure is not None:
        (em.RESULTS / "disagreement.json").write_text(json.dumps(mobilenet_figure, indent=2))
    fail = eu.failure_analysis(held_out, paths, cache_ft, cfg, OUT)
    (OUT / "classification.json").write_text(json.dumps({"split": "held_out", "images": len(held_out),
                                                         "methods": cls, "failures": fail}, indent=2))
    print("classification done")

    speed = measure_speed()
    (OUT / "speed.json").write_text(json.dumps(speed, indent=2))
    write_summary(cfg, baselines, det, cls, fail, speed, split_info, mobilenet_figure)
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
