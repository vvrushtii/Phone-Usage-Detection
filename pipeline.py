"""One frame in, one usage state out: enhancement -> YOLO26n -> rules (the Focus Coach pipeline, minus the window).

Used by the app (app/focus_coach.py) and by the speed benchmark (app/benchmark.py), so both measure the same thing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from ml import enhance, rules
from ml.smoothing import NO_HEAD

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/best.pt"
ENHANCE_MODES = ("auto", "on", "off")  # the C key cycles through these


@dataclass
class FrameResult:
    state: str                       # FOCUSED | ON_PHONE | ON_CALL | NO_HEAD (raw, before smoothing)
    image: np.ndarray                # the frame to show: enhanced when enhancement was applied
    dets: list[dict]                 # detections that passed the confidence thresholds
    user_head: list[float] | None    # box of the head the state was decided for
    brightness: float
    enhanced: bool
    infer_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)


def frame_state(dets: list[dict], cfg: dict) -> tuple[str, list[float] | None]:
    """Raw state of the user (the largest head) in one frame; NO_HEAD when there is no head."""
    persons = rules.classify_image(dets, cfg, mode="largest")["persons"]
    return (persons[0]["state"], persons[0]["head"]) if persons else (NO_HEAD, None)


def should_enhance(brightness: float, app_cfg: dict, mode: str) -> bool:
    """'on' and 'off' are the user's override; 'auto' enhances dark frames when the experiment found a method that helps."""
    if mode != "auto":
        return mode == "on"
    return app_cfg["enhance_method"] != "none" and brightness < app_cfg["low_light_threshold"]


class FramePipeline:
    def __init__(self, cfg: dict, imgsz: int | None = None, weights: Path = WEIGHTS) -> None:
        if not weights.exists():
            raise FileNotFoundError(f"{weights} not found. Train the model (Phase 2) or copy best.pt there.")
        phone_conf = cfg["app"].get("phone_conf")  # the app asks for a bit more phone recall than the evaluated setting (see rules.json)
        self.cfg = cfg if phone_conf is None else {**cfg, "conf": {**cfg["conf"], "phone": phone_conf}}
        self.imgsz = imgsz or cfg["app"]["imgsz"]
        self.model = YOLO(weights)
        self.names = {i: n for i, n in self.model.names.items() if n in ("head", "phone")}
        self.conf_floor = min(self.cfg["conf"].values())  # rules.filter_dets applies the per-class thresholds afterwards
        self.process(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), "off")  # first call is slow; do it before the camera runs

    def process(self, frame: np.ndarray, enhance_mode: str = "auto") -> FrameResult:
        t0 = time.perf_counter()
        brightness = enhance.to_gray_mean(frame)
        use = should_enhance(brightness, self.cfg["app"], enhance_mode)
        method = self.cfg["app"]["enhance_method"] if self.cfg["app"]["enhance_method"] != "none" else "clahe_smooth"
        image = enhance.enhance(frame, method, self.cfg["app"]["enhance_params"]) if use else frame
        t1 = time.perf_counter()
        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf_floor, classes=list(self.names),
                                    device="cpu", verbose=False)[0]
        t2 = time.perf_counter()
        dets = [{"cls": self.names[c], "box": b, "conf": p} for b, p, c in
                zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.int().tolist())]
        dets = rules.filter_dets(dets, self.cfg)
        state, head = frame_state(dets, self.cfg)
        t3 = time.perf_counter()
        return FrameResult(state, image, dets, head, brightness, use, (t2 - t1) * 1000,
                           {"enhance": (t1 - t0) * 1000, "detect": (t2 - t1) * 1000, "rules": (t3 - t2) * 1000})
