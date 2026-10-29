"""Does image enhancement help the detector in low light? (Unit II experiment)

Run from the repo root, after `python ml/make_lowlight.py`:
    python ml/lowlight_experiment.py

1. Choose the enhancement method and the CLAHE settings on a sample of the VAL split (darkened in memory).
2. Score the fine-tuned detector on the TEST split under {original, dim, dark, real-dark} x {none, global_he, clahe,
   clahe_smooth}: phone and head mAP50 per cell.
3. Score B-rules usage accuracy on the held-out label.csv images under {original, dark} x {none, chosen method}.
4. Decide the app setting from the evidence and write it to shared/rules.json under "app".
Detector outputs are cached in data/cache/lowlight/, so a rerun only redoes what changed.
Writes results/lowlight/lowlight.json, enhancement_examples.png and lowlight_map.png.
"""
from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import detections as det_cache  # noqa: E402
from ml import enhance, make_lowlight, rules, splits  # noqa: E402
from ml import eval_detection as ed  # noqa: E402
from ml import lowlight_figures  # noqa: E402
from ml.usage_metrics import confusion, metrics  # noqa: E402

warnings.filterwarnings("ignore", message="Encountered more than 100 detections")
RESULTS = ROOT / "results/lowlight"
CACHE = ROOT / "data/cache/lowlight"
LABELS = ROOT / "data/fpi_yolo/labels"
RULES_JSON = ROOT / "shared/rules.json"
SEED = 42
VAL_SAMPLE = 400  # val images used to choose the method and CLAHE settings (the full val split is slow on a CPU)
CLIP_LIMITS = (2.0, 3.0, 4.0)
TILE_GRIDS = ((8, 8), (16, 16))
LOW_LIGHT = ("dim", "dark")
METHODS = enhance.METHODS
MIN_GAIN = 0.005  # an enhancement must beat "none" by at least half a mAP50 point to be used
CHUNK = 128


# --- running the detector under a condition and method -------------------------------------------------------

def _params_tag(method: str, params: dict) -> str:
    if method in ("none", "global_he"):
        return ""
    tag = f"__c{params['clip_limit']}_t{params['tile_grid'][0]}"
    return tag + (f"_k{params['smooth_ksize']}" if method == "clahe_smooth" else "")


def cached_detections(model: YOLO, wanted: dict, condition: str, method: str, params: dict, paths: list[Path]) -> dict:
    """Detections for `paths` under (condition, method, params); only the images not yet cached are run."""
    if condition == "original" and method == "none":
        return det_cache.get_detections("fine_tuned", paths)  # the Phase 3 cache is the same run
    CACHE.mkdir(parents=True, exist_ok=True)
    file = CACHE / f"{condition}__{method}{_params_tag(method, params)}.json"
    meta = {"weights_bytes": det_cache.FINE_TUNED.stat().st_size, "imgsz": det_cache.IMGSZ, "conf": det_cache.CACHE_CONF}
    cached: dict = {}
    if file.exists():
        blob = json.loads(file.read_text())
        if blob["meta"] == meta:
            cached = blob["dets"]
    missing = [p for p in paths if p.stem not in cached]
    for start in range(0, len(missing), CHUNK):
        chunk = missing[start:start + CHUNK]
        images = [enhance.enhance(make_lowlight.load_condition(p, condition), method, params) for p in chunk]
        results = model.predict(images, conf=det_cache.CACHE_CONF, imgsz=det_cache.IMGSZ, batch=det_cache.BATCH,
                                device="cpu", classes=list(wanted), verbose=False)
        for path, r in zip(chunk, results):
            rows: dict[str, list] = {"phone": [], "head": []}
            for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
                rows[wanted[cls]].append([round(v, 1) for v in box] + [round(conf, 4)])
            h, w = r.orig_shape
            cached[path.stem] = {"size": [w, h], **rows}
        print(f"  {condition}/{method}{_params_tag(method, params)}: {start + len(chunk)}/{len(missing)}",
              file=sys.stderr, flush=True)
    if missing:
        file.write_text(json.dumps({"meta": meta, "dets": cached}))
    return {p.stem: cached[p.stem] for p in paths}


def map50(preds: list, gts: list) -> float:
    p_list, t_list = ed.to_tensors(preds, gts)
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[ed.IOU_MATCH])
    metric.update(p_list, t_list)
    return float(metric.compute()["map"])


def class_map50(dets: dict, paths: list[Path], split: str) -> dict[str, float]:
    """Phone and head mAP50 of cached detections against the labels of the given split."""
    gts = {"phone": [], "head": []}
    preds = {"phone": [], "head": []}
    for p in paths:
        entry = dets[p.stem]
        truth = ed.read_ground_truth(LABELS / split / f"{p.stem}.txt", *entry["size"])
        for cls in gts:
            gts[cls].append(truth[cls])
            preds[cls].append(entry[cls])
    return {cls: map50(preds[cls], gts[cls]) for cls in gts}


def _clahe_params(clip: float, tile: tuple[int, int]) -> dict:
    return {**enhance.DEFAULT_PARAMS, "clip_limit": clip, "tile_grid": tile}


# --- step 1: choose on val ------------------------------------------------------------------------------------

def choose_on_val(model: YOLO, wanted: dict) -> dict:
    val_paths = sorted((det_cache.IMAGES / "val").glob("*.jpg"))
    sample = sorted(random.Random(SEED).sample(val_paths, VAL_SAMPLE))

    def score(method: str, params: dict) -> float:
        """Phone mAP50 averaged over the dim and dark versions of the val sample."""
        return float(np.mean([class_map50(cached_detections(model, wanted, c, method, params, sample), sample, "val")["phone"]
                              for c in LOW_LIGHT]))

    grid = [{"clip_limit": c, "tile_grid": t, "phone_map50": score("clahe", _clahe_params(c, t))}
            for c in CLIP_LIMITS for t in TILE_GRIDS]
    best = max(grid, key=lambda g: g["phone_map50"])  # first on ties, so the result is deterministic
    params = _clahe_params(best["clip_limit"], tuple(best["tile_grid"]))
    by_method = {m: score(m, params) for m in METHODS}
    top = max(("global_he", "clahe", "clahe_smooth"), key=lambda m: by_method[m])
    use = by_method[top] > by_method["none"] + MIN_GAIN
    return {"val_images": len(sample), "clahe_grid": grid, "params": params, "phone_map50_by_method": by_method,
            "best_method": top, "app_method": top if use else "none"}


# --- step 2: test grid ----------------------------------------------------------------------------------------

def test_grid(model: YOLO, wanted: dict, params: dict) -> tuple[dict, dict]:
    paths = make_lowlight.test_images()
    real_dark = {n[:-4] for n in make_lowlight.load_real_dark()["image_name"]}
    real_paths = [p for p in paths if p.stem in real_dark]
    cells: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        for condition in ("original", "dim", "dark"):
            dets = cached_detections(model, wanted, condition, method, params, paths)
            cells.setdefault(condition, {})[method] = class_map50(dets, paths, "test")
            if condition == "original":  # real-dark photos are the darkest originals, so reuse their run
                cells.setdefault("real_dark", {})[method] = class_map50(dets, real_paths, "test")
    counts = {"original": len(paths), "dim": len(paths), "dark": len(paths), "real_dark": len(real_paths)}
    return cells, counts


# --- step 3: usage accuracy -----------------------------------------------------------------------------------

def usage_grid(model: YOLO, wanted: dict, params: dict, method: str) -> dict:
    held_out = splits.load_split("held_out")
    paths = [det_cache.IMAGES / r.split / r.image_name for r in held_out.itertuples()]
    y = held_out["using_phone"].tolist()
    cfg = rules.load_config()
    out = {}
    for condition, m in (("original", "none"), ("original", method), ("dark", "none"), ("dark", method)):
        dets = cached_detections(model, wanted, condition, m, params, paths)
        pred = [int(rules.classify_image(det_cache.as_rule_dets(dets[p.stem]), cfg)["using"]) for p in paths]
        out[f"{condition}/{m}"] = metrics(confusion(y, pred))
    return {"images": len(paths), "method": method, "rows": out}


# --- step 4: the app setting ----------------------------------------------------------------------------------

def brightness_table(paths: list[Path]) -> dict:
    """Mean grey level of every test image under each condition, summarised."""
    table = {}
    for condition in ("original", "dim", "dark"):
        values = np.array([enhance.to_gray_mean(make_lowlight.load_condition(p, condition)) for p in paths])
        table[condition] = {f"p{q}": float(np.percentile(values, q)) for q in (5, 15, 50, 85, 95)}
        table[condition]["mean"] = float(values.mean())
    real = make_lowlight.load_real_dark()["brightness"]
    table["real_dark"] = {"max": float(real.max()), "median": float(real.median()), "min": float(real.min())}
    return table


def app_setting(choice: dict, cells: dict, brightness: dict) -> dict:
    """Method from the val choice. Threshold: the brightness range where enhancement helped, i.e. the dim frames (95th
    percentile of their mean brightness). The real dark photos (up to ~75) were not helped, so they stay untouched."""
    method = choice["app_method"]
    return {
        "low_light_threshold": round(brightness["dim"]["p95"]),
        "enhance_method": method,
        "enhance_params": {"clip_limit": choice["params"]["clip_limit"], "tile_grid": list(choice["params"]["tile_grid"]),
                           "smooth_ksize": choice["params"]["smooth_ksize"]},
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = YOLO(det_cache.FINE_TUNED)
    wanted = {i: det_cache.NAME_MAP["fine_tuned"][n] for i, n in model.names.items() if n in det_cache.NAME_MAP["fine_tuned"]}

    choice = choose_on_val(model, wanted)
    print("val choice:", json.dumps({k: choice[k] for k in ("params", "phone_map50_by_method", "best_method", "app_method")}))
    cells, counts = test_grid(model, wanted, choice["params"])
    usage = usage_grid(model, wanted, choice["params"], choice["best_method"])
    brightness = brightness_table(make_lowlight.test_images())
    app = app_setting(choice, cells, brightness)

    cfg = json.loads(RULES_JSON.read_text())
    cfg["app"] = {**cfg.get("app", {}), **app}  # keep the other app settings (imgsz)
    RULES_JSON.write_text(json.dumps(cfg, indent=2) + "\n")
    (RESULTS / "lowlight.json").write_text(json.dumps(
        {"val_choice": choice, "test_map50": cells, "test_counts": counts, "usage_heldout": usage,
         "brightness": brightness, "app": app}, indent=2))
    lowlight_figures.enhancement_examples(RESULTS / "enhancement_examples.png", choice["params"])
    lowlight_figures.map_chart(cells, counts, RESULTS / "lowlight_map.png")
    print("app setting:", json.dumps(app))


if __name__ == "__main__":
    main()
