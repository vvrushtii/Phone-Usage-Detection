"""The Phase 4 and 5 sections of results/eval/summary.md (used by ml/evaluate.py).

Numbers come from the JSON files the two experiments wrote; the plain-language observations come from
results/lowlight/notes.md and results/mobilenet/notes.md, which are written by hand after looking at the figures.
Each section is skipped when its experiment has not been run.
"""
from __future__ import annotations

import json
from pathlib import Path

from ml import enhance
from ml.report_values import rendered_note

ROOT = Path(__file__).resolve().parents[1]
LOWLIGHT = ROOT / "results/lowlight"
MOBILENET = ROOT / "results/mobilenet"
APP_SPEED = ROOT / "results/eval/app_speed.json"
METHOD_NAMES = {"none": "none", "global_he": "global HE", "clahe": "CLAHE", "clahe_smooth": "CLAHE + blur"}
CONDITION_NAMES = {"original": "original", "dim": "dim", "dark": "dark", "real_dark": "real dark photos"}


def _notes(folder: Path) -> list[str]:
    """The hand-written observations, with their {{placeholders}} filled from the results files."""
    path = folder / "notes.md"
    return ["### Observations", "", rendered_note(str(path.relative_to(ROOT))), ""] if path.exists() else []


def lowlight_lines() -> list[str]:
    path = LOWLIGHT / "lowlight.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    choice, app = d["val_choice"], d["app"]
    lines = [
        "## Low-light experiment (Unit II)", "",
        "Does enhancing a dark frame before detection help? Dim and dark copies of the test images were made by lowering "
        "brightness, applying gamma and adding noise (`ml/make_lowlight.py`). The \"real dark photos\" are the 15% of original "
        "test images with the lowest brightness. The CLAHE settings and the method were chosen on "
        f"{choice['val_images']} val images darkened the same way, not on test.", "",
        "### Detector mAP50 on the test split (phone / head)", "",
        "| Condition (images) | " + " | ".join(METHOD_NAMES[m] for m in enhance.METHODS) + " |",
        "|---|" + "---|" * len(enhance.METHODS),
    ]
    for cond, name in CONDITION_NAMES.items():
        cells = " | ".join(f"{d['test_map50'][cond][m]['phone']:.3f} / {d['test_map50'][cond][m]['head']:.3f}"
                           for m in enhance.METHODS)
        lines.append(f"| {name} ({d['test_counts'][cond]}) | {cells} |")
    p = choice["params"]
    by_method = choice["phone_map50_by_method"]
    lines += ["", f"Chosen on val: CLAHE clip limit {p['clip_limit']}, tiles {tuple(p['tile_grid'])}. Phone mAP50 on the val sample "
              "(mean of dim and dark): " + ", ".join(f"{METHOD_NAMES[m]} {by_method[m]:.3f}" for m in enhance.METHODS)
              + ". Full CLAHE grid in `results/lowlight/lowlight.json`.", "",
              f"### Usage classification with B-rules (held-out, n={d['usage_heldout']['images']})", "",
              "| Lighting / enhancement | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for key, m in d["usage_heldout"]["rows"].items():
        cond, method = key.split("/")
        lines.append(f"| {cond} / {METHOD_NAMES[method]} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    b = d["brightness"]
    lines += ["", f"Mean brightness (0 to 255) of test images: original median {b['original']['p50']:.0f}, dim median "
              f"{b['dim']['p50']:.0f}, dark median {b['dark']['p50']:.0f}; the real dark photos reach up to {b['real_dark']['max']:.0f}.", "",
              f"**App setting chosen from this** (`shared/rules.json`, `app`): enhancement `{app['enhance_method']}`, used when "
              f"mean frame brightness is below {app['low_light_threshold']}.", ""]
    return lines + _notes(LOWLIGHT)


def mobilenet_lines(cls: dict, figure: dict | None) -> list[str]:
    path = MOBILENET / "metrics.json"
    if not path.exists() or "C-mobilenet" not in cls:
        return []
    m = json.loads(path.read_text())
    lines = [
        "## MobileNetV2 baseline (Unit IV)", "",
        f"A MobileNetV2 (ImageNet weights) looks at the whole image shrunk to 224 x 224 and answers \"using phone or not\". "
        f"It was trained on the tune split only ({m['train_images']} train / {m['val_images']} validation images; class ratio "
        f"{m['class_ratio']:.2f}:1, so class weights were used). Stage 1 trained a new head on the frozen backbone "
        f"({m['stage1_epochs_run']} epochs), stage 2 fine-tuned the top {m['stage2']['unfrozen_layers']} layers "
        f"({m['stage2_epochs_run']} epochs). Decision threshold {m['threshold']} (best on validation: {m['best_threshold']}, "
        f"{m['val_accuracy_at_best']:.3f} vs {m['val_accuracy_at_0.5']:.3f} at 0.5). Its row is C-mobilenet in the classification "
        "table above; curves are in `results/mobilenet/training_curves.png`.", ""]
    if figure:
        lines += [f"MobileNet and B-rules disagree on {figure['disagreements']} held-out images: MobileNet alone is right on "
                  f"{figure['only_mobilenet_right']}, B-rules alone on {figure['only_rules_right']} "
                  "(`results/mobilenet/examples.png`).", ""]
    return lines + _notes(MOBILENET)


def app_speed_lines() -> list[str]:
    """Speed of the Focus Coach pipeline, from `python -m app.benchmark`."""
    if not APP_SPEED.exists():
        return []
    runs = json.loads(APP_SPEED.read_text())["runs"]
    lines = ["## Focus Coach app speed", "",
             f"CPU: {runs[0]['cpu']}. {runs[0]['frames_timed']} webcam-sized frames (1280 x 720) after 20 warm-up frames, through the "
             "whole app pipeline: enhancement check, YOLO26n on the CPU, rules, smoothing and drawing the HUD. "
             "Camera capture and the window are not included. The target is at least 8 FPS.", "",
             "| Detector size | Mean ms per frame | Median ms | 95th percentile ms | FPS (mean) |", "|---|---|---|---|---|"]
    lines += [f"| {r['imgsz']} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | {r['fps_mean']:.1f} |" for r in runs]
    return lines + [""]
