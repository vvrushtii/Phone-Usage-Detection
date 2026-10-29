"""How fast does the Focus Coach pipeline run on this computer?

Run from the repo root:
    python -m app.benchmark                 (add --imgsz 480 to test the faster setting)
Feeds 220 test photos, each placed on a 1280 x 720 canvas like a webcam frame, through the same code the app uses
(enhancement check, YOLO26n on the CPU, rules, smoothing, drawing the HUD). The first 20 are warm-up; the next 200 are
timed. Camera capture and the window are not included. Writes results/eval/app_speed.json, which evaluate.py puts in
results/eval/summary.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.torch_utils import get_cpu_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.capture import Frame  # noqa: E402
from app.focus_coach import FocusCoach  # noqa: E402
from ml import rules  # noqa: E402

TEST_IMAGES = ROOT / "data/fpi_yolo/images/test"
CANVAS = (1280, 720)  # width, height of a typical webcam frame
WARMUP, TIMED = 20, 200
FRAME_MS = 100.0
OUT = ROOT / "results/eval/app_speed.json"


def on_canvas(img: np.ndarray) -> np.ndarray:
    """The photo scaled to fit a webcam-sized canvas, grey around it."""
    w, h = CANVAS
    scale = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    canvas = np.full((h, w, 3), 114, np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--imgsz", type=int, default=None, help="detector input size (default from shared/rules.json)")
    imgsz = ap.parse_args().imgsz
    paths = sorted(TEST_IMAGES.glob("*.jpg"))[: WARMUP + TIMED]
    frames = [on_canvas(cv2.imread(str(p))) for p in paths]  # read first so disk speed is not timed
    app = FocusCoach(rules.load_config(), imgsz)
    app.session.start(0.0)
    times = []
    for i, image in enumerate(frames):
        t0 = time.perf_counter()
        app.handle(Frame(image, i * FRAME_MS, i))
        if i >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
    result = {"cpu": get_cpu_info(), "imgsz": app.pipeline.imgsz, "frame_size": list(CANVAS), "frames_timed": len(times),
              "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
              "p95_ms": float(np.percentile(times, 95)), "fps_mean": 1000 / float(np.mean(times))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        previous = json.loads(OUT.read_text())
        runs = {str(r["imgsz"]): r for r in previous.get("runs", [])}
    else:
        runs = {}
    runs[str(result["imgsz"])] = result
    OUT.write_text(json.dumps({"runs": sorted(runs.values(), key=lambda r: -r["imgsz"])}, indent=2))
    print(f"{result['cpu']}, imgsz {result['imgsz']}: {result['mean_ms']:.1f} ms per frame = {result['fps_mean']:.1f} FPS "
          f"(median {result['median_ms']:.1f} ms, 95th percentile {result['p95_ms']:.1f} ms)")


if __name__ == "__main__":
    main()
