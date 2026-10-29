"""Make the example files for the demo from real FPI-Det photos: demo/examples/images/*.jpg and demo/examples/scripted_clip.mp4.

Run from the repo root (needs data/, models/best.pt and the detection cache from `python ml/detections.py`):
    python demo/make_examples.py
The four images come from the held-out photos: one per state and one tricky case (a photo labelled "not using" that the
rules call "using"). The clip is built from three of those photos with the same script as report/make_sample_session.py:
focused, a 5 s phone pickup, a 0.5 s glance, a 5 s call, then a covered camera. It is a scripted clip, not a recording of
a person; record your own for the viva (app/make_demo_clip.md) and save it as demo/examples/demo.mp4.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.focus_coach import FocusCoach  # noqa: E402
from ml import detections as dc  # noqa: E402
from ml import rules, splits  # noqa: E402
from report import make_sample_session as clip_script  # noqa: E402

OUT = ROOT / "demo/examples"
CONF = 0.8  # clear single-person photos for the examples
STATES = (("on_call", rules.ON_CALL), ("on_phone", rules.ON_PHONE), ("focused", rules.FOCUSED))


def pick_images() -> dict[str, Path]:
    held = splits.load_split("held_out")
    paths = [dc.IMAGES / r.split / r.image_name for r in held.itertuples()]
    cache = dc.get_detections("fine_tuned", paths)
    cfg = rules.load_config()
    found: dict[str, Path] = {}
    for path in paths:
        dets = [d for d in dc.as_rule_dets(cache[path.stem]) if d["conf"] >= CONF]
        heads = [d for d in dets if d["cls"] == "head"]
        phones = [d for d in dets if d["cls"] == "phone"]
        if len(heads) != 1 or len(phones) > 1:
            continue
        state = rules.classify_image(dets, cfg, "largest")["persons"][0]["state"]
        if (state == rules.FOCUSED) == bool(phones):
            continue
        for name, wanted in STATES:
            if state == wanted and name not in found:
                found[name] = path
    classification = json.loads((ROOT / "results/eval/classification.json").read_text())["failures"]
    found["tricky_labelled_not_using"] = next(p for p in paths if p.name == classification["fp_grid_images"][0])
    return found


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for i, (name, path) in enumerate(pick_images().items(), 1):
        shutil.copy(path, OUT / "images" / f"{i:02d}_{name}.jpg")
    app = FocusCoach(rules.load_config())
    photos = clip_script.pick_photos(app)
    raw = OUT / "scripted_clip_raw.avi"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), clip_script.FPS, (clip_script.SIZE, clip_script.SIZE))
    for state, seconds in clip_script.SCRIPT:
        image = np.zeros((clip_script.SIZE, clip_script.SIZE, 3), np.uint8) if state == "COVER" else photos[state]
        for _ in range(int(seconds * clip_script.FPS)):
            writer.write(image)
    writer.release()
    final = OUT / "scripted_clip.mp4"
    done = subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-g", "1",
                           "-movflags", "+faststart", str(final)], capture_output=True, text=True)
    if done.returncode != 0:
        shutil.copy(raw, final)  # no ffmpeg here: keep the OpenCV file, it still works with the app
    raw.unlink()
    print("wrote", sorted(p.name for p in OUT.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
