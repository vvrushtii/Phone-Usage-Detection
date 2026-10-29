"""Run a YOLO model once over a set of images and cache the raw detections on disk.

Thresholds are swept later in pure Python, so the model only has to run once. Detections are kept down to a very
low confidence (CACHE_CONF) so any later threshold can be applied on top.

Run from the repo root to fill the cache for both models on every label.csv image:
    python ml/detections.py            (add --refresh to rerun the models)
Cache files: data/cache/dets_<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data/cache"
IMAGES = ROOT / "data/fpi_yolo/images"
FINE_TUNED = ROOT / "models/best.pt"
COCO = ROOT / "models/yolo26n.pt"
IMGSZ = 640
CACHE_CONF = 0.001  # the usual setting for measuring mAP; any higher threshold can be applied later
BATCH = 16
COCO_PHONE = "cell phone"
# class names in each model's output -> the names used everywhere else
NAME_MAP = {"fine_tuned": {"phone": "phone", "head": "head"}, "coco": {COCO_PHONE: "phone"}}
WEIGHTS = {"fine_tuned": FINE_TUNED, "coco": COCO}

# {stem: {"size": [w, h], "phone": [[x1, y1, x2, y2, conf], ...], "head": [...]}}
Detections = dict[str, dict]


def label_csv_images() -> list[Path]:
    """Every image that has a usage label: the val and test images of the detector split."""
    return sorted(p for split in ("val", "test") for p in (IMAGES / split).glob("*.jpg"))


def _meta(model_key: str) -> dict:
    w = WEIGHTS[model_key]
    return {"weights": w.name, "weights_bytes": w.stat().st_size, "imgsz": IMGSZ, "conf": CACHE_CONF}


def _run(model_key: str, paths: list[Path], device: str) -> Detections:
    model = YOLO(WEIGHTS[model_key])
    wanted = {i: NAME_MAP[model_key][n] for i, n in model.names.items() if n in NAME_MAP[model_key]}
    out: Detections = {}
    for start in range(0, len(paths), BATCH * 8):
        chunk = paths[start:start + BATCH * 8]
        results = model.predict(
            [str(p) for p in chunk], conf=CACHE_CONF, imgsz=IMGSZ, batch=BATCH, device=device,
            classes=list(wanted), verbose=False,
        )
        for path, r in zip(chunk, results):
            rows: dict[str, list] = {"phone": [], "head": []}
            for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
                rows[wanted[cls]].append([round(v, 1) for v in box] + [round(conf, 4)])
            h, w = r.orig_shape
            out[path.stem] = {"size": [w, h], **rows}
        print(f"  {model_key}: {min(start + len(chunk), len(paths))}/{len(paths)}", file=sys.stderr, flush=True)
    return out


def get_detections(model_key: str, paths: list[Path], device: str = "cpu", refresh: bool = False) -> Detections:
    """Detections for the given images, from the cache when it is valid and complete, else computed and cached."""
    cache_file = CACHE_DIR / f"dets_{model_key}.json"
    meta = _meta(model_key)
    cached: Detections = {}
    if cache_file.exists() and not refresh:
        blob = json.loads(cache_file.read_text())
        if blob["meta"] == meta:
            cached = blob["dets"]
    missing = [p for p in paths if p.stem not in cached]
    if missing:
        cached.update(_run(model_key, missing, device))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"meta": meta, "dets": cached}))
    return {p.stem: cached[p.stem] for p in paths}


def as_rule_dets(entry: dict, min_conf: dict[str, float] | None = None) -> list[dict]:
    """Cached entry -> the detection dicts that ml/rules.py takes, optionally dropping low confidences early."""
    floor = min_conf or {}
    return [
        {"cls": cls, "box": row[:4], "conf": row[4]}
        for cls in ("phone", "head") for row in entry[cls] if row[4] >= floor.get(cls, 0.0)
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and rerun the models")
    refresh = ap.parse_args().refresh
    images = label_csv_images()
    for key in ("fine_tuned", "coco"):
        dets = get_detections(key, images, refresh=refresh)
        print(f"{key}: {len(dets)} images cached, {sum(len(d['phone']) for d in dets.values())} phone boxes")
