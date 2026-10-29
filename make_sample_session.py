"""Make the sample session report for the report: run the app's code on a scripted clip built from real photos.

Run from the repo root:
    python report/make_sample_session.py
It picks three held-out photos (one focused, one with a phone in front, one with a phone at the ear), builds a 38 s clip
(focused, 5 s phone, 0.5 s glance, 5 s call, covered camera), runs the same FocusCoach and Session code as the app on its
frames at the clip's own timestamps, and writes the app's report.png to report/figures/fig10_session_report.png.
This is a scripted test clip, NOT a real study session; the report says so.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.capture import Frame  # noqa: E402
from app.focus_coach import FocusCoach  # noqa: E402
from app.report import write_report  # noqa: E402
from ml import detections as dc  # noqa: E402
from ml import rules, splits  # noqa: E402

OUT = ROOT / "report/figures/fig10_session_report.png"
FPS = 10
SIZE = 640
CONF = 0.6  # only photos where the detector is confident, so the clip shows the app, not detector luck
SCRIPT = [("FOCUSED", 6), ("ON_PHONE", 5), ("FOCUSED", 6), ("ON_PHONE", 0.5), ("FOCUSED", 6), ("ON_CALL", 5),
          ("FOCUSED", 4), ("COVER", 6)]


def square(img: np.ndarray) -> np.ndarray:
    scale = SIZE / max(img.shape[:2])
    small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    y, x = (SIZE - small.shape[0]) // 2, (SIZE - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


def pick_photos(app: FocusCoach) -> dict[str, np.ndarray]:
    """First held-out photo (by name) for each state that the whole pipeline also classifies that way."""
    held = splits.load_split("held_out")
    cfg = app.cfg
    cache = dc.get_detections("fine_tuned", [dc.IMAGES / r.split / r.image_name for r in held.itertuples()])
    found: dict[str, np.ndarray] = {}
    for r in held.itertuples():
        path = dc.IMAGES / r.split / r.image_name
        dets = [d for d in dc.as_rule_dets(cache[path.stem]) if d["conf"] >= CONF]
        heads = [d for d in dets if d["cls"] == "head"]
        phones = [d for d in dets if d["cls"] == "phone"]
        if len(heads) != 1:
            continue
        state = rules.classify_image(dets, cfg, "largest")["persons"][0]["state"]
        if state in found or (state == "FOCUSED") == bool(phones) or (state != "FOCUSED" and len(phones) != 1):
            continue
        img = square(cv2.imread(str(path)))
        if app.pipeline.process(img, "off").state == state:
            found[state] = img
        if len(found) == 3:
            return found
    raise RuntimeError("could not find a photo for every state")


def main() -> None:
    app = FocusCoach(rules.load_config())
    photos = pick_photos(app)
    app.session.start(0.0)
    t_ms, index = 0.0, 0
    for state, seconds in SCRIPT:
        image = np.zeros((SIZE, SIZE, 3), np.uint8) if state == "COVER" else photos[state]
        for _ in range(int(seconds * FPS)):
            app.handle(Frame(image.copy(), t_ms, index))
            t_ms += 1000 / FPS
            index += 1
    app.session.end(t_ms)
    with tempfile.TemporaryDirectory() as tmp:
        folder = write_report(app.session, {"source": "scripted test clip"}, root=Path(tmp))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(folder / "report.png", OUT)
        print((folder / "events.csv").read_text())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
