"""Everything the Focus Coach window draws, with OpenCV (syllabus Unit I: drawing on images).

Functions draw onto the frame they are given. Sizes scale with the frame height so the text is readable on any camera.
Colours are BGR.
"""
from __future__ import annotations

import cv2
import numpy as np

from ml.rules import FOCUSED, ON_CALL, ON_PHONE
from ml.smoothing import AWAY

FONT = cv2.FONT_HERSHEY_SIMPLEX
STATE_COLORS = {FOCUSED: (80, 170, 60), ON_PHONE: (0, 170, 255), ON_CALL: (50, 50, 220), AWAY: (128, 128, 128)}
STATE_LABELS = {FOCUSED: "FOCUSED", ON_PHONE: "ON PHONE", ON_CALL: "ON CALL", AWAY: "AWAY"}
BOX_COLORS = {"head": (220, 130, 30), "phone": (30, 90, 230)}
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
TIP = "Make sure the camera sees your face and the desk in front of you."
PRIVACY = "Video stays on this computer. Only event times and totals are saved."


def _scale(frame: np.ndarray) -> float:
    return frame.shape[0] / 720


def _text(frame, text, org, size, color=WHITE, thickness=2) -> None:
    """Text with a dark outline so it reads on any background."""
    s = _scale(frame)
    thick, r = max(1, int(thickness * s)), max(1, round(1.5 * s))
    # the outline is the same text shifted a few pixels in every direction; a thicker stroke would space the letters
    # differently (OpenCV 5 text layout depends on thickness) and show up as a ghost
    for dx in (-r, 0, r):
        for dy in (-r, 0, r):
            if dx or dy:
                cv2.putText(frame, text, (org[0] + dx, org[1] + dy), FONT, size * s, BLACK, thick, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, size * s, color, thick, cv2.LINE_AA)


def _measure(frame, text, size, thickness=2) -> tuple[int, int]:
    """Width and height of the text as _text would draw it."""
    (w, h), _ = cv2.getTextSize(text, FONT, size * _scale(frame), max(1, int(thickness * _scale(frame))))
    return w, h


def _panel(frame, x1, y1, x2, y2, alpha=0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), BLACK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def clock_text(ms: float) -> str:
    seconds = int(ms // 1000)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60:02d}:{seconds % 60:02d}"


def draw_boxes(frame: np.ndarray, dets: list[dict], user_head: list[float] | None) -> None:
    """Debug view: every head and phone that passed its confidence threshold, with class and confidence."""
    s = _scale(frame)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        thick = max(2, int(3 * s)) if d["box"] == user_head else max(1, int(2 * s))
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLORS[d["cls"]], thick)
        _text(frame, f"{d['cls']} {d['conf']:.2f}", (x1, max(int(18 * s), y1 - 6)), 0.5, BOX_COLORS[d["cls"]], 1)


def draw_banner(frame: np.ndarray, state: str, note: str = "") -> None:
    """Large status bar across the top."""
    h, w = frame.shape[:2]
    bar = int(0.11 * h)
    cv2.rectangle(frame, (0, 0), (w, bar), STATE_COLORS[state], -1)
    label = STATE_LABELS[state] + (f"   {note}" if note else "")
    tw, th = _measure(frame, label, 1.4, 3)
    _text(frame, label, ((w - tw) // 2, (bar + th) // 2), 1.4, WHITE, 3)


def draw_idle_banner(frame: np.ndarray, text: str) -> None:
    h, w = frame.shape[:2]
    bar = int(0.11 * h)
    cv2.rectangle(frame, (0, 0), (w, bar), (70, 70, 70), -1)
    tw, th = _measure(frame, text, 1.0, 2)
    _text(frame, text, ((w - tw) // 2, (bar + th) // 2), 1.0, WHITE, 2)


def draw_counters(frame: np.ndarray, elapsed_ms: float, pickups: int, phone_ms: float, streak_ms: float) -> None:
    h, w = frame.shape[:2]
    s = _scale(frame)
    top = int(0.11 * h) + int(10 * s)
    rows = [("Time", clock_text(elapsed_ms)), ("Pickups", str(pickups)), ("Phone time", clock_text(phone_ms)),
            ("Focus streak", clock_text(streak_ms))]
    _panel(frame, int(10 * s), top, int(310 * s), top + int(len(rows) * 36 * s + 14 * s))
    for i, (label, value) in enumerate(rows):
        y = top + int((i + 1) * 36 * s)
        _text(frame, label, (int(22 * s), y), 0.7, (210, 210, 210), 1)
        _text(frame, value, (int(190 * s), y), 0.8, WHITE, 2)


KEYS_HINT = "S start/pause  E end  D boxes  C enhance  Q quit"
START_HINT = "Press S to start a session"


def draw_footer(frame: np.ndarray, fps: float, infer_ms: float, enhance_text: str, keys: str = KEYS_HINT) -> None:
    h, w = frame.shape[:2]
    s = _scale(frame)
    _panel(frame, 0, h - int(34 * s), w, h, 0.6)
    _text(frame, f"{fps:4.1f} FPS   inference {infer_ms:3.0f} ms   {enhance_text}", (int(12 * s), h - int(10 * s)), 0.6, WHITE, 1)
    kw, _ = _measure(frame, keys, 0.55, 1)
    _text(frame, keys, (w - kw - int(12 * s), h - int(10 * s)), 0.55, (200, 200, 200), 1)


def draw_tip(frame: np.ndarray, start_text: str = START_HINT) -> None:
    """Shown before a session starts."""
    h, w = frame.shape[:2]
    s = _scale(frame)
    _panel(frame, int(0.08 * w), int(0.60 * h), int(0.92 * w), int(0.84 * h), 0.65)
    for i, (text, size, color) in enumerate(((TIP, 0.8, WHITE), (PRIVACY, 0.65, (200, 200, 200)))):
        tw, _ = _measure(frame, text, size, 1)
        _text(frame, text, ((w - tw) // 2, int(0.67 * h) + i * int(46 * s)), size, color, 2 if i == 0 else 1)
    tw, _ = _measure(frame, start_text, 0.9, 2)
    _text(frame, start_text, ((w - tw) // 2, int(0.80 * h)), 0.9, (120, 220, 255), 2)


def draw_message(frame: np.ndarray, lines: list[str]) -> None:
    """A centred message box, used for the quit prompt."""
    h, w = frame.shape[:2]
    s = _scale(frame)
    _panel(frame, int(0.15 * w), int(0.38 * h), int(0.85 * w), int(0.38 * h) + int((len(lines) * 50 + 30) * s), 0.8)
    for i, line in enumerate(lines):
        tw, _ = _measure(frame, line, 0.85, 2)
        _text(frame, line, ((w - tw) // 2, int(0.38 * h) + int((i + 1) * 50 * s)), 0.85, WHITE, 2)
