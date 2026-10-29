"""Session bookkeeping for the Focus Coach app: what happened, when, and the counters shown on screen and in the report.

Times are milliseconds on the SESSION clock: it starts at 0 when the session starts and does not run while paused.
Pausing closes any open distraction; after resuming, the state starts again as FOCUSED.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ml.rules import FOCUSED
from ml.smoothing import AWAY, DISTRACTIONS, StateSmoother


@dataclass
class Segment:
    state: str
    start_ms: float
    end_ms: float | None = None  # None while it is the current one


@dataclass
class Distraction:
    type: str
    start_ms: float
    end_ms: float | None
    new_pickup: bool

    @property
    def duration_ms(self) -> float:
        return (self.end_ms or self.start_ms) - self.start_ms


class Session:
    def __init__(self, smoothing_cfg: dict) -> None:
        self._smoother = StateSmoother(smoothing_cfg["window_ms"], smoothing_cfg["enter_ratio"], smoothing_cfg["away_ms"])
        self.running = False
        self.paused = False
        self.ended = False
        self.wall_start: datetime | None = None
        self.wall_end: datetime | None = None
        self.segments: list[Segment] = []
        self.distractions: list[Distraction] = []
        self._start_t = self._paused_total = self._pause_t = 0.0
        self._clock = 0.0

    # --- control -------------------------------------------------------------------------------------------

    def start(self, t_ms: float) -> None:
        self.running, self.paused, self.ended = True, False, False
        self.wall_start = datetime.now()
        self._start_t, self._paused_total, self._clock = t_ms, 0.0, 0.0
        self._smoother.reset(0.0)
        self.segments = [Segment(FOCUSED, 0.0)]
        self.distractions = []

    def pause(self, t_ms: float) -> None:
        if not self.running or self.paused:
            return
        self._close_open(self._clock)
        self.paused, self._pause_t = True, t_ms

    def resume(self, t_ms: float) -> None:
        if not self.paused:
            return
        self._paused_total += t_ms - self._pause_t
        self.paused = False
        self._clock = t_ms - self._start_t - self._paused_total
        self._smoother.reset(self._clock)
        self.segments.append(Segment(FOCUSED, self._clock))

    def end(self, t_ms: float) -> None:
        if not self.running:
            return
        if not self.paused:
            self._clock = t_ms - self._start_t - self._paused_total
            self._close_open(self._clock)
        self.running, self.paused, self.ended = False, False, True
        self.wall_end = datetime.now()

    # --- per frame -------------------------------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._smoother.state if self.running and not self.paused else FOCUSED

    def update(self, t_ms: float, raw_state: str) -> list[dict]:
        """Feed one frame's raw state; returns the smoother's events (already applied to the session)."""
        if not self.running or self.paused:
            return []
        self._clock = t_ms - self._start_t - self._paused_total
        events = self._smoother.update(self._clock, raw_state)
        for e in events:
            self._apply(e)
        return events

    def _apply(self, e: dict) -> None:
        if e["kind"] == "state_change":
            current = self.segments[-1]
            current.end_ms = max(e["t"], current.start_ms)  # AWAY is dated back to the last frame with a head
            self.segments.append(Segment(e["to"], current.end_ms))
        elif e["kind"] == "distraction_start":
            self.distractions.append(Distraction(e["type"], e["t"], None, e["new_pickup"]))
        elif e["kind"] == "distraction_end":
            self.distractions[-1].end_ms = max(e["t"], self.distractions[-1].start_ms)

    def _close_open(self, t: float) -> None:
        for e in self._smoother.finish(t):
            self._apply(e)
        if self.segments and self.segments[-1].end_ms is None:
            self.segments[-1].end_ms = max(t, self.segments[-1].start_ms)

    # --- numbers -----------------------------------------------------------------------------------------

    def _duration(self, seg: Segment) -> float:
        return (seg.end_ms if seg.end_ms is not None else self._clock) - seg.start_ms

    def time_in(self, *states: str) -> float:
        return sum(self._duration(s) for s in self.segments if s.state in states)

    @property
    def elapsed_ms(self) -> float:
        return self._clock

    @property
    def pickups(self) -> int:
        return sum(d.new_pickup for d in self.distractions)

    @property
    def phone_ms(self) -> float:
        return self.time_in(*DISTRACTIONS)

    @property
    def longest_streak_ms(self) -> float:
        return max((self._duration(s) for s in self.segments if s.state == FOCUSED), default=0.0)

    @property
    def current_streak_ms(self) -> float:
        last = self.segments[-1] if self.segments else None
        return self._duration(last) if last is not None and last.state == FOCUSED and self.running and not self.paused else 0.0

    def summary(self) -> dict:
        total = self._clock
        focused, phone, away = self.time_in(FOCUSED), self.phone_ms, self.time_in(AWAY)
        events = [d for d in self.distractions if d.end_ms is not None]
        return {
            "total_s": round(total / 1000, 1), "focused_s": round(focused / 1000, 1), "phone_s": round(phone / 1000, 1),
            "away_s": round(away / 1000, 1), "percent_focused": round(100 * focused / total, 1) if total else 0.0,
            "pickups": self.pickups, "longest_focus_streak_s": round(self.longest_streak_ms / 1000, 1),
            "average_distraction_s": round(sum(d.duration_ms for d in events) / len(events) / 1000, 1) if events else 0.0,
        }
