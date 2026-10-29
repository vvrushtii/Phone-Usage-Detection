"""Frame source for the Focus Coach app: a webcam or a video file, read on its own thread.

Only the LATEST frame is kept. If detection is slower than the camera, old frames are dropped instead of piling up, so the
picture on screen never lags behind real life. Each frame carries a timestamp in milliseconds: for the webcam it is
the capture time; for a video file it is the position in the video, so a recorded clip behaves like a live session.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REQUEST_SIZE = (1280, 720)  # asked from the webcam; it may give something else
READ_FAILURES_BEFORE_GIVING_UP = 30
DEFAULT_VIDEO_FPS = 30.0


class CaptureError(RuntimeError):
    """The camera or video could not be opened or stopped working. The message says what to try."""


@dataclass
class Frame:
    image: np.ndarray
    t_ms: float
    index: int


def parse_source(text: str) -> int | Path:
    """'0' -> camera 0; anything else is a video file path."""
    return int(text) if text.isdigit() else Path(text)


class FrameSource:
    def __init__(self, source: int | Path) -> None:
        self.source = source
        self.is_camera = isinstance(source, int)
        self.fps = DEFAULT_VIDEO_FPS
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._returned = -1
        self._stop = False
        self.ended = False
        self.error: str | None = None
        self.frames_read = 0
        self.handed_out = 0

    def start(self) -> "FrameSource":
        if self.is_camera:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # DirectShow opens faster on Windows
            self._cap = cv2.VideoCapture(self.source, backend)
            if not self._cap.isOpened():
                raise CaptureError(
                    f"Cannot open camera {self.source}. Close other programs that use the camera (Teams, Zoom, the "
                    "Camera app), check Windows Settings > Privacy > Camera, or try another number with --source 1.")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_SIZE[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_SIZE[1])
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            if not self.source.is_file():
                raise CaptureError(f"Video file not found: {self.source}")
            self._cap = cv2.VideoCapture(str(self.source))
            if not self._cap.isOpened():
                raise CaptureError(f"Cannot open video file {self.source}. Try an .mp4 encoded with H.264.")
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        start, failures, index = time.monotonic(), 0, 0
        while not self._stop:
            if not self.is_camera:  # play a file in real time, like a webcam would deliver it
                wait = start + index / self.fps - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            ok, image = self._cap.read()
            if not ok:
                failures += 1
                if not self.is_camera:
                    self._finish()
                    return
                if failures >= READ_FAILURES_BEFORE_GIVING_UP:
                    self._finish("The camera stopped delivering frames. Is it still connected?")
                    return
                time.sleep(0.02)
                continue
            failures = 0
            t_ms = index / self.fps * 1000 if not self.is_camera else (time.monotonic() - start) * 1000
            with self._cond:
                self._latest = Frame(image, t_ms, index)
                self.frames_read += 1
                self._cond.notify_all()
            index += 1

    def _finish(self, error: str | None = None) -> None:
        with self._cond:
            self.ended, self.error = True, error
            self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Frame | None:
        """The newest frame not returned yet, or None after `timeout` seconds / when the source has ended."""
        with self._cond:
            self._cond.wait_for(lambda: self._fresh() or self.ended, timeout)
            if self.error and not self._fresh():
                raise CaptureError(self.error)
            if not self._fresh():
                return None
            self._returned = self._latest.index
            self.handed_out += 1
            return self._latest

    def _fresh(self) -> bool:
        return self._latest is not None and self._latest.index > self._returned

    @property
    def dropped(self) -> int:
        """Frames read from the camera that were never handed out because a newer one arrived first."""
        return self.frames_read - self.handed_out

    def release(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
