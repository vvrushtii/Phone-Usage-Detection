"""Focus Coach: a webcam (or a recorded video) watches you study and counts how often you pick up your phone.

Run from the repo root:
    python -m app.focus_coach                          webcam 0
    python -m app.focus_coach --source demo.mp4        a recorded clip instead of the webcam
Keys: S start / pause / resume   E end session and show the report   D debug boxes   C enhancement auto/on/off   Q quit
Options: --imgsz 480 (faster)   --save-video out.mp4 (writes the annotated picture; off by default)
         --autostart (begin the session at once)   --no-display (no windows, for scripted runs; implies --autostart)
Privacy: frames are processed in memory and never saved unless --save-video is given. Only event times and totals are
written, to sessions/<date_time>/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import hud  # noqa: E402
from app.capture import CaptureError, Frame, FrameSource, parse_source  # noqa: E402
from app.pipeline import ENHANCE_MODES, FramePipeline  # noqa: E402
from app.report import show_report, write_report  # noqa: E402
from app.session import Session  # noqa: E402
from ml import rules  # noqa: E402

WINDOW = "Focus Coach"
FPS_SMOOTHING = 0.9   # weight of the old value in the running FPS average
SAVED_VIDEO_FPS = 15.0
NO_FRAME_TIMEOUT_S = 1.0


class FocusCoach:
    """The pipeline, the session and the drawing, for one frame at a time (also used by app/benchmark.py)."""

    def __init__(self, cfg: dict, imgsz: int | None = None, pipeline: FramePipeline | None = None) -> None:
        self.cfg = cfg
        self.pipeline = pipeline or FramePipeline(cfg, imgsz)  # a shared pipeline lets many users reuse one loaded model
        self.keys_hint = hud.KEYS_HINT                          # the Colab demo replaces these two texts with button hints
        self.start_hint = hud.START_HINT
        self.session = Session(cfg["smoothing"])
        self.debug = False
        self.enhance_mode = "auto"
        self.fps = 0.0
        self.last_t_ms = 0.0
        self.frames = 0
        self.enhanced_frames = 0
        self.infer_total_ms = 0.0
        self._last_wall: float | None = None
        self._first_wall: float | None = None

    def handle(self, frame: Frame) -> np.ndarray:
        """Run one frame through the pipeline, update the session and return the picture to show."""
        now = time.perf_counter()
        if self._last_wall is not None:
            inst = 1.0 / max(now - self._last_wall, 1e-6)
            self.fps = inst if self.fps == 0 else FPS_SMOOTHING * self.fps + (1 - FPS_SMOOTHING) * inst
        self._last_wall = now
        self._first_wall = self._first_wall or now
        self.last_t_ms = frame.t_ms

        result = self.pipeline.process(frame.image, self.enhance_mode)
        self.frames += 1
        self.enhanced_frames += result.enhanced
        self.infer_total_ms += result.infer_ms
        self.session.update(frame.t_ms, result.state)

        image = result.image
        if self.debug:
            hud.draw_boxes(image, result.dets, result.user_head)
        s = self.session
        if s.running and s.paused:
            hud.draw_idle_banner(image, "PAUSED   (S to resume)")
        elif s.running:
            hud.draw_banner(image, s.state)
        else:
            hud.draw_idle_banner(image, "Not started")
        if s.running:
            hud.draw_counters(image, s.elapsed_ms, s.pickups, s.phone_ms, s.current_streak_ms)
        else:
            hud.draw_tip(image, self.start_hint)
        hud.draw_footer(image, self.fps, result.infer_ms, self._enhance_text(result.enhanced), self.keys_hint)
        return image

    def _enhance_text(self, applied: bool) -> str:
        method = {"clahe_smooth": "CLAHE", "clahe": "CLAHE", "global_he": "histogram equalization"}.get(
            self.cfg["app"]["enhance_method"], "enhancement")
        if self.enhance_mode == "off":
            return "Enhancement: off"
        if self.enhance_mode == "on":
            return f"Enhancement: {method} on (manual)"
        return f"Low light: {method} on" if applied else "Lighting OK"

    def cycle_enhancement(self) -> None:
        self.enhance_mode = ENHANCE_MODES[(ENHANCE_MODES.index(self.enhance_mode) + 1) % len(ENHANCE_MODES)]

    def stats(self) -> dict:
        wall = time.perf_counter() - self._first_wall if self._first_wall else 0.0
        return {"frames_processed": self.frames, "mean_fps": round(self.frames / wall, 1) if wall else 0.0,
                "mean_inference_ms": round(self.infer_total_ms / self.frames, 1) if self.frames else 0.0,
                "enhanced_frame_share": round(self.enhanced_frames / self.frames, 3) if self.frames else 0.0}


def finish_session(app: FocusCoach, source_text: str, display: bool) -> Path:
    """End the running session, write its files and (with a display) show the report. Starts a fresh session object."""
    app.session.end(app.last_t_ms)
    meta = {"source": source_text, "imgsz": app.pipeline.imgsz, **app.stats(), "settings": app.cfg}
    folder = write_report(app.session, meta)
    print(f"Session saved to {folder}")
    if display:
        show_report(folder / "report.png")
    app.session = Session(app.cfg["smoothing"])
    return folder


def ask_to_save(image: np.ndarray) -> str:
    """The quit prompt. Returns 'save', 'discard' or 'cancel'."""
    hud.draw_message(image, ["A session is running.", "Y = save the report and quit", "N = quit without saving",
                             "any other key = keep going"])
    cv2.imshow(WINDOW, image)
    key = cv2.waitKey(0) & 0xFF
    return {ord("y"): "save", ord("Y"): "save", ord("n"): "discard", ord("N"): "discard"}.get(key, "cancel")


def run(args: argparse.Namespace) -> int:
    cfg = rules.load_config()
    display = not args.no_display
    print("Loading the model...")
    try:
        app = FocusCoach(cfg, args.imgsz)
        source = FrameSource(parse_source(args.source)).start()
    except (CaptureError, FileNotFoundError) as error:
        print(f"Error: {error}")
        return 1
    print("Running. Press S in the window to start a session." if display else "Running without a window.")
    writer: cv2.VideoWriter | None = None
    if display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    autostart = args.autostart or not display
    try:
        while True:
            try:
                frame = source.read(NO_FRAME_TIMEOUT_S)
            except CaptureError as error:
                print(f"Error: {error}")
                break
            if frame is None:
                if source.ended:
                    break
                continue
            if autostart:
                app.session.start(frame.t_ms)
                autostart = False
            image = app.handle(frame)
            if args.save_video:
                if writer is None:
                    h, w = image.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             source.fps if not source.is_camera else SAVED_VIDEO_FPS, (w, h))
                writer.write(image)
            if not display:
                continue
            cv2.imshow(WINDOW, image)
            code = cv2.waitKey(1) & 0xFF
            key = chr(code).lower() if code != 255 else ""
            closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            s = app.session
            if key == "s":
                if not s.running:
                    s.start(frame.t_ms)
                elif s.paused:
                    s.resume(frame.t_ms)
                else:
                    s.pause(frame.t_ms)
            elif key == "e" and s.running:
                finish_session(app, args.source, display)
            elif key == "d":
                app.debug = not app.debug
            elif key == "c":
                app.cycle_enhancement()
            elif key == "q" or closed:
                choice = ask_to_save(image) if s.running and not closed else "save"
                if choice == "cancel":
                    continue
                if s.running and choice == "save":
                    finish_session(app, args.source, display)
                break
        if app.session.running:  # the video ended or the camera stopped while a session was running
            finish_session(app, args.source, display)
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("--source", default="0", help="camera number (default 0) or path to a video file")
    ap.add_argument("--imgsz", type=int, default=None, help="detector input size; default from shared/rules.json (640)")
    ap.add_argument("--save-video", default=None, metavar="PATH", help="write the annotated picture to this .mp4")
    ap.add_argument("--autostart", action="store_true", help="start the session immediately")
    ap.add_argument("--no-display", action="store_true", help="no windows (implies --autostart)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
