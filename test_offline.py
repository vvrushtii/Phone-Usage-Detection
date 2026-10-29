"""app/offline.py on a short synthetic clip: the output files exist and can be read. Needs models/best.pt."""
import csv
import json

import cv2
import numpy as np
import pytest

from app.offline import run_to_folder
from app.pipeline import WEIGHTS

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="models/best.pt is not available")


def make_clip(path, seconds=3, fps=10, size=(160, 120)):
    rng = np.random.default_rng(0)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(seconds * fps):
        out.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    out.release()


def test_offline_run_writes_all_outputs(tmp_path):
    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    result = run_to_folder(clip, tmp_path / "out")
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"events.csv", "summary.json", "report.png", "annotated.mp4"}
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["source"] == "clip.mp4" and summary["pickups"] == 0
    assert next(csv.reader((tmp_path / "out/events.csv").open())) == ["start", "end", "type", "duration_s", "new_pickup"]
    assert cv2.imread(str(tmp_path / "out/report.png")) is not None
    video = cv2.VideoCapture(str(tmp_path / "out/annotated.mp4"))
    assert video.get(cv2.CAP_PROP_FRAME_COUNT) >= 25  # 3 s at 10 fps
    video.release()
    assert result.finished and result.progress == 1.0


def test_missing_video_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_to_folder(tmp_path / "nope.mp4", tmp_path / "out")
