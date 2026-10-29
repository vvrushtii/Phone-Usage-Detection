"""Focus Coach in the browser (Gradio): a new front end for the same model, rules, smoothing and report code.

Run locally (no public link) from the repo root:
    python -m demo.gradio_app                 --port 7861 to change the port; and open http://127.0.0.1:7860
In Colab, use demo/focus_coach_colab.ipynb. Three tabs:
    Live     webcam frames travel from the browser to this computer and back (about 5 per second)
    Video    upload a clip; it is processed with app/offline.py (the same code as `python -m app.offline`)
    Image    upload one photo; shows the boxes, each person's state and the image-level verdict, with no smoothing
Frames are processed in memory. Uploaded files and reports live in a temporary folder of this session.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import hud  # noqa: E402
from app.capture import Frame  # noqa: E402
from app.focus_coach import FocusCoach  # noqa: E402
from app.offline import run_to_folder  # noqa: E402
from app.pipeline import FramePipeline, FrameResult, WEIGHTS  # noqa: E402
from app.report import write_report  # noqa: E402
from ml import rules  # noqa: E402

RULES_PATH = ROOT / "shared/rules.json"
CLASSIFICATION = ROOT / "results/eval/classification.json"
EXAMPLES = ROOT / "demo/examples"
STREAM_EVERY_S = 0.4        # about 2.5 frames per second from the browser: more than a tunnel and a CPU can keep up with only builds a backlog
STREAM_TIME_LIMIT_S = 900   # one stream is cut after this long; it restarts by itself. No concurrency limit, so a second open tab cannot block the stream
PRIVACY_NOTE = "Frames are sent to this Colab session for processing and are not stored."
LIVE_START_HINT = "Click Record on the webcam to start"
LIVE_KEYS_HINT = "End session: button below"
WEBCAM_SIZE = {"width": 480, "height": 360}  # small frames travel faster to and from the server
MIN_FRAME_GAP_MS = 300      # frames that queued up behind a slow one are skipped, so the picture never falls behind


class SharedPipeline:
    """One loaded model for all users; only one frame is in the detector at a time."""

    def __init__(self, pipeline: FramePipeline) -> None:
        self._pipeline, self._lock = pipeline, threading.Lock()
        self.imgsz, self.cfg = pipeline.imgsz, pipeline.cfg

    def process(self, frame: np.ndarray, enhance_mode: str = "auto") -> FrameResult:
        with self._lock:
            return self._pipeline.process(frame, enhance_mode)


def to_browser_mp4(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so every browser can play it. Returns False when ffmpeg is not available."""
    if shutil.which("ffmpeg") is None:
        return False
    run = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                          str(dst)], capture_output=True, text=True)
    return run.returncode == 0 and dst.exists()


def footer_text(model_path: Path) -> str:
    """Model, device and the project's headline accuracy, read from results/ (never typed here)."""
    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (works, but is slower than a GPU)"
    text = f"Model: {model_path.name} (YOLO26n fine-tuned on FPI-Det) | Device: {device}"
    if CLASSIFICATION.exists():
        data = json.loads(CLASSIFICATION.read_text())
        m = data["methods"]["B-rules"]["metrics"]
        text += f" | Detection + rules on {data['images']:,} held-out photos: accuracy {m['accuracy']:.3f}, F1 {m['f1']:.3f}"
    return text


# --- tab 1: live ----------------------------------------------------------------------------------------------

def new_coach(cfg: dict, pipeline: SharedPipeline) -> FocusCoach:
    coach = FocusCoach(cfg, pipeline=pipeline)
    coach.debug = True
    coach.start_hint, coach.keys_hint = LIVE_START_HINT, LIVE_KEYS_HINT
    return coach


def status_line(coach: FocusCoach | None) -> str:
    if coach is None or not coach.session.running:
        return "Not started"
    return "PAUSED" if coach.session.paused else hud.STATE_LABELS[coach.session.state]


def counters_line(coach: FocusCoach | None) -> str:
    if coach is None or not coach.session.running:
        return "Time 00:00 | Pickups 0 | Phone time 00:00 | Focus streak 00:00"
    s = coach.session
    return (f"Time {hud.clock_text(s.elapsed_ms)} | Pickups {s.pickups} | Phone time {hud.clock_text(s.phone_ms)} | "
            f"Focus streak {hud.clock_text(s.current_streak_ms)}")


def make_live_handlers(cfg: dict, pipeline: SharedPipeline):
    def live_frame(image: np.ndarray | None, coach: FocusCoach | None, force_enhance: bool):
        if image is None:
            return None, status_line(coach), counters_line(coach), "", coach
        coach = coach or new_coach(cfg, pipeline)
        coach.enhance_mode = "on" if force_enhance else "auto"
        now_ms = time.monotonic() * 1000  # the server's clock: when the frame arrived
        if coach.frames and now_ms - coach.last_t_ms < MIN_FRAME_GAP_MS:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), coach
        if not coach.session.running:  # Record starts the session; no separate Start button
            coach.session.start(now_ms)
        out = coach.handle(Frame(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), now_ms, coach.frames))
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), status_line(coach), counters_line(coach), f"{coach.fps:.1f} frames/s", coach

    def live_end(coach: FocusCoach | None):
        if coach is None or not coach.session.running:
            return coach, None, None, "No session is running. Click Record on the webcam first."
        coach.session.end(coach.last_t_ms)
        folder = write_report(coach.session, {"source": "browser webcam", "imgsz": pipeline.imgsz, **coach.stats(), "settings": cfg},
                              root=Path(tempfile.mkdtemp(prefix="focus_coach_")))
        s = coach.session.summary()
        message = (f"Session ended: {s['pickups']} pickups, {s['phone_s']} s on phone, {s['away_s']} s away, "
                   f"{s['percent_focused']}% focused, "
                   f"longest focus {s['longest_focus_streak_s']} s. While Record is on, a new session starts by itself.")
        coach.session = type(coach.session)(cfg["smoothing"])  # a fresh session; the next frame starts it
        return coach, str(folder / "report.png"), str(folder / "events.csv"), message

    return live_frame, live_end


# --- tab 2: video ---------------------------------------------------------------------------------------------

def make_video_handler(cfg: dict, pipeline: SharedPipeline):
    def process_upload(path: str | None, progress=gr.Progress()):
        if not path:
            raise gr.Error("Upload a video first (an .mp4 works best).")
        out = Path(tempfile.mkdtemp(prefix="focus_coach_video_"))
        try:
            result = run_to_folder(Path(path), out, coach=new_coach(cfg, pipeline), on_progress=lambda p: progress(p, desc="Analysing"))
        except (FileNotFoundError, ValueError) as error:
            raise gr.Error(str(error)) from error
        playable = out / "annotated_h264.mp4"
        video = playable if to_browser_mp4(out / "annotated.mp4", playable) else out / "annotated.mp4"
        s = result.session.summary()
        message = (f"{s['pickups']} pickups, {s['phone_s']} s on phone, {s['percent_focused']}% focused, "
                   f"longest focus {s['longest_focus_streak_s']} s, {s['away_s']} s away.")
        if video.name == "annotated.mp4":
            message += " (ffmpeg was not found, so the video may not play in every browser; use the download.)"
        return str(video), str(out / "report.png"), [str(out / "events.csv"), str(out / "summary.json"), str(video)], message

    return process_upload


# --- tab 3: image ---------------------------------------------------------------------------------------------

def make_image_handler(cfg: dict, pipeline: SharedPipeline):
    def classify_photo(image: np.ndarray | None):
        if image is None:
            raise gr.Error("Upload a photo first.")
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = pipeline.process(bgr, "auto")
        verdict = rules.classify_image(result.dets, cfg, mode="all")  # every person in the photo, no smoothing
        canvas = result.image.copy()
        for d in result.dets:
            if d["cls"] == "phone":
                x1, y1, x2, y2 = (int(v) for v in d["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), hud.BOX_COLORS["phone"], 2)
        for person in verdict["persons"]:
            x1, y1, x2, y2 = (int(v) for v in person["head"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), hud.STATE_COLORS[person["state"]], 3)
            label_y = y1 - 6 if y1 > 24 else y1 + 20  # inside the box when the head touches the top edge
            cv2.putText(canvas, person["state"].replace("_", " "), (x1 + 4, label_y), hud.FONT, 0.55,
                        hud.STATE_COLORS[person["state"]], 2, cv2.LINE_AA)
        states = [p["state"] for p in verdict["persons"]]
        if not states:
            text = "No head found, so the rules cannot decide: NOT USING"
        else:
            text = ("USING a phone" if verdict["using"] else "NOT using a phone") + f" | people: {', '.join(states)}"
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), text

    return classify_photo


# --- the page -------------------------------------------------------------------------------------------------

def build_demo(model_path: Path = WEIGHTS, rules_path: Path = RULES_PATH) -> gr.Blocks:
    cfg = json.loads(Path(rules_path).read_text())
    pipeline = SharedPipeline(FramePipeline(cfg, weights=Path(model_path)))
    cfg = pipeline.cfg  # with the app's phone confidence, so the photo tab applies the same rule as the pipeline
    live_frame, live_end = make_live_handlers(cfg, pipeline)
    process_upload = make_video_handler(cfg, pipeline)
    classify_photo = make_image_handler(cfg, pipeline)

    with gr.Blocks(title="Focus Coach") as demo:
        gr.Markdown("# Focus Coach — Mobile Phone Usage Detection\n"
                    "A webcam, a YOLO26n detector for heads and phones, simple rules and time smoothing. "
                    "If the live tab is laggy, use the Video tab: it is the reliable way to see the full pipeline.")
        with gr.Tab("Live (webcam)"):
            gr.Markdown("**1.** Click the webcam box and allow the camera. **2.** Click **Record**: the session starts and the right "
                        "picture shows what Focus Coach sees, live. **3.** Use your phone for a few seconds. "
                        "**4.** Click **End session** for the report.")
            coach_state = gr.State(None)
            with gr.Row():
                webcam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Your webcam",
                                  webcam_options=gr.WebcamOptions(constraints={"video": WEBCAM_SIZE}))
                annotated = gr.Image(label="Focus Coach (live)", type="numpy", streaming=True, format="jpeg")
            gr.Markdown(PRIVACY_NOTE)
            force = gr.Checkbox(label="Force low-light enhancement (default: automatic, from shared/rules.json)", value=False)
            with gr.Row():
                status, counters, fps = (gr.Textbox(label="State", interactive=False), gr.Textbox(label="Counters", interactive=False),
                                         gr.Textbox(label="Speed", interactive=False))
            end = gr.Button("End session", variant="primary")
            note = gr.Textbox(label="Message", interactive=False)
            report_image, events_file = gr.Image(label="Session report", type="filepath"), gr.File(label="events.csv")
            webcam.stream(live_frame, [webcam, coach_state, force], [annotated, status, counters, fps, coach_state],
                          stream_every=STREAM_EVERY_S, time_limit=STREAM_TIME_LIMIT_S, concurrency_limit=None)
            end.click(live_end, [coach_state], [coach_state, report_image, events_file, note])
        with gr.Tab("Video (upload)"):
            gr.Markdown("Upload a short clip of someone studying. The whole clip is analysed and you get an annotated video, a report and downloads.")
            video_in = gr.Video(sources=["upload"], label="Video")
            run = gr.Button("Analyse video", variant="primary")
            video_out = gr.Video(label="Annotated video")
            video_message = gr.Textbox(label="Summary", interactive=False)
            report_out, files_out = gr.Image(label="Session report", type="filepath"), gr.File(label="Downloads", file_count="multiple")
            run.click(process_upload, [video_in], [video_out, report_out, files_out, video_message])
            clips = [p for p in (EXAMPLES / "demo.mp4", EXAMPLES / "scripted_clip.mp4") if p.exists()]  # your own clip first
            if clips:
                gr.Examples(examples=[[str(p)] for p in clips], inputs=[video_in], label="Example clips")
        with gr.Tab("Image (upload)"):
            gr.Markdown("Upload one photo. Each person's state is shown, and the verdict is for the whole image. No time smoothing here.")
            image_in = gr.Image(label="Photo", type="numpy")
            image_run = gr.Button("Analyse photo", variant="primary")
            image_out, verdict_out = gr.Image(label="Result", type="numpy"), gr.Textbox(label="Verdict", interactive=False)
            image_run.click(classify_photo, [image_in], [image_out, verdict_out])
            images = sorted((EXAMPLES / "images").glob("*.jpg"))
            if images:
                gr.Examples(examples=[[str(p)] for p in images], inputs=[image_in], label="Examples from the held-out photos")
        gr.Markdown(footer_text(Path(model_path)))
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    build_demo().queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
