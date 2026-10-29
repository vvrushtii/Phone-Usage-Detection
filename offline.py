"""Run Focus Coach on a recorded video, without a window: the same pipeline as the desktop app, fed from a file.

Run from the repo root:
    python -m app.offline path/to/clip.mp4 --out sessions/offline_clip/
Writes events.csv, summary.json, report.png and annotated.mp4 to the output folder. The Colab demo (demo/gradio_app.py)
calls the same functions, so a clip gives the same events in both places.

Frames are sampled at `sample_fps` and stamped with their position in the video (frame index / video fps), so the
result does not depend on how fast this computer is.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.capture import Frame  # noqa: E402
from app.focus_coach import FocusCoach  # noqa: E402
from app.report import write_report  # noqa: E402
from app.session import Session  # noqa: E402
from ml import rules  # noqa: E402

DEFAULT_SAMPLE_FPS = 10.0
DEFAULT_VIDEO_FPS = 30.0


@dataclass
class OfflineResult:
    """Filled in while the frame iterator runs; complete (finished=True) when it is exhausted."""
    source: str
    frames_total: int = 0
    frames_done: int = 0
    finished: bool = False
    session: Session | None = None
    coach: FocusCoach | None = field(default=None, repr=False)

    @property
    def progress(self) -> float:
        return min(1.0, self.frames_done / self.frames_total) if self.frames_total else 0.0

    def meta(self, sample_fps: float) -> dict:
        return {"source": self.source, "imgsz": self.coach.pipeline.imgsz, "sample_fps": sample_fps,
                **self.coach.stats(), "settings": self.coach.cfg}


def process_video(path: Path, cfg: dict | None = None, sample_fps: float = DEFAULT_SAMPLE_FPS, imgsz: int | None = None,
                  coach: FocusCoach | None = None) -> tuple[Iterator[np.ndarray], OfflineResult]:
    """Annotated frames of a video, plus a result object that holds the session once the frames have all been read.

    Usage: `frames, result = process_video(p)`; loop over `frames`; then `result.session` has the events and summary.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    coach = coach or FocusCoach(cfg or rules.load_config(), imgsz)
    result = OfflineResult(source=path.name, coach=coach)

    def frames() -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {path}. Try an .mp4 encoded with H.264.")
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS
        result.frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        next_sample_s, last_t_ms, index = 0.0, 0.0, 0
        coach.session = Session(coach.cfg["smoothing"])
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                t_s = index / fps
                result.frames_done = index + 1
                if t_s + 1e-9 >= next_sample_s:
                    next_sample_s += 1.0 / sample_fps
                    last_t_ms = t_s * 1000
                    if not coach.session.running:
                        coach.session.start(last_t_ms)
                    coach.last_t_ms = last_t_ms
                    yield coach.handle(Frame(image, last_t_ms, index))
                index += 1
        finally:
            cap.release()
        if coach.session.running:
            coach.session.end(last_t_ms)
        result.session, result.finished = coach.session, True

    return frames(), result


def run_to_folder(path: Path, out_dir: Path, cfg: dict | None = None, sample_fps: float = DEFAULT_SAMPLE_FPS,
                  imgsz: int | None = None, coach: FocusCoach | None = None, write_video: bool = True,
                  on_progress: Callable[[float], None] | None = None) -> OfflineResult:
    """The whole job: process the video and write events.csv, summary.json, report.png (and annotated.mp4) to out_dir."""
    frames, result = process_video(path, cfg, sample_fps, imgsz, coach)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    for image in frames:
        if write_video:
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (w, h))
            writer.write(image)
        if on_progress:
            on_progress(result.progress)
    if writer is not None:
        writer.release()
    if result.session is None or result.session.wall_start is None:
        raise ValueError(f"No frames could be read from {path}")
    write_report(result.session, result.meta(sample_fps), folder=out_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path, help="path to the video file")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default sessions/offline_<video name>)")
    ap.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS, help="frames per second to analyse (default 10)")
    ap.add_argument("--imgsz", type=int, default=None, help="detector input size (default from shared/rules.json)")
    ap.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    args = ap.parse_args(argv)
    out = args.out or ROOT / "sessions" / f"offline_{args.video.stem}"
    try:
        result = run_to_folder(args.video, out, sample_fps=args.sample_fps, imgsz=args.imgsz, write_video=not args.no_video,
                               on_progress=lambda p: print(f"\r{p:4.0%}", end="", flush=True))
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    s = result.session.summary()
    print(f"\n{s['pickups']} pickups, {s['phone_s']} s on phone, {s['away_s']} s away, {s['percent_focused']}% focused. Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
