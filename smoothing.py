"""Time smoothing: turn noisy per-frame states into steady states and distraction events (syllabus Unit V).

Pure Python, no OpenCV, so it can be tested with fake timestamps. Feed it one raw state per frame:
    FOCUSED | ON_PHONE | ON_CALL   from ml/rules.py, or NO_HEAD when no head was found in the frame.

Rules (numbers come from shared/rules.json, "smoothing"):
- A state is shown only when it filled at least `enter_ratio` of the last `window_ms` (counted in time, not frames, so it
  works at any frame rate). Otherwise the shown state stays as it was, so a short glance or a flickering detection changes
  nothing. A short loss of the head also changes nothing: NO_HEAD time only lowers the other states' share.
- No head for longer than `away_ms` shows AWAY. AWAY is not phone time. It is dated from the last frame that had a head.
- Entering ON_PHONE or ON_CALL starts a distraction, leaving it ends one. Switching between the two ends one event and
  starts the next, but only the first is a new pickup.
Each update returns the events it caused, as dicts with a "kind": state_change, distraction_start, distraction_end.
"""
from __future__ import annotations

from collections import deque

from ml.rules import FOCUSED, ON_CALL, ON_PHONE

NO_HEAD = "NO_HEAD"
AWAY = "AWAY"
DISTRACTIONS = (ON_PHONE, ON_CALL)
SHOWN_STATES = (FOCUSED, ON_PHONE, ON_CALL)
MAX_HOLD_FRACTION = 1 / 3


class StateSmoother:
    """Sliding-window smoother with AWAY handling. Times are milliseconds on any steady clock."""

    def __init__(self, window_ms: float, enter_ratio: float, away_ms: float, start_ms: float = 0.0) -> None:
        self.window_ms = window_ms
        self.enter_ratio = enter_ratio
        self.away_ms = away_ms
        self.reset(start_ms)

    def reset(self, t_ms: float) -> None:
        """Forget the window (used at the start of a session and after a pause). The shown state restarts as FOCUSED."""
        self.samples: deque[tuple[float, str]] = deque()
        self.state = FOCUSED
        self.last_head_ms = t_ms
        self._open_distraction: tuple[str, float] | None = None

    def update(self, t_ms: float, raw: str) -> list[dict]:
        """Add one frame's raw state, return the events it caused (usually none)."""
        self.samples.append((t_ms, raw))
        # keep one sample from before the window, so the first state inside it has a known start
        while len(self.samples) > 1 and self.samples[1][0] <= t_ms - self.window_ms:
            self.samples.popleft()
        if raw != NO_HEAD:
            self.last_head_ms = t_ms
        if t_ms - self.last_head_ms > self.away_ms:
            return [] if self.state == AWAY else self._change(AWAY, self.last_head_ms)
        shares = self._shares(t_ms)
        for state in SHOWN_STATES:
            if state != self.state and shares[state] >= self.enter_ratio * self.window_ms:
                return self._change(state, t_ms)
        return []

    def finish(self, t_ms: float) -> list[dict]:
        """Close an open distraction at the end of the session."""
        if self._open_distraction is None:
            return []
        kind, start = self._open_distraction
        self._open_distraction = None
        return [{"kind": "distraction_end", "type": kind, "t": t_ms, "duration_ms": t_ms - start}]

    def _shares(self, now: float) -> dict[str, float]:
        """Time each raw state held inside the window. A sample holds until the next one arrives, but at most
        MAX_HOLD_FRACTION of the window, so a stalled camera cannot turn one frame into a long state."""
        shares = {s: 0.0 for s in (*SHOWN_STATES, NO_HEAD)}
        lo = now - self.window_ms
        cap = MAX_HOLD_FRACTION * self.window_ms
        items = list(self.samples)
        for (t, state), (t_next, _) in zip(items, items[1:]):
            shares[state] += max(0.0, min(t_next, t + cap) - max(t, lo))
        return shares

    def _change(self, new: str, t_ms: float) -> list[dict]:
        events = [{"kind": "state_change", "from": self.state, "to": new, "t": t_ms}]
        if self._open_distraction is not None:
            kind, start = self._open_distraction
            events.append({"kind": "distraction_end", "type": kind, "t": t_ms, "duration_ms": t_ms - start})
            self._open_distraction = None
        if new in DISTRACTIONS:
            events.append({"kind": "distraction_start", "type": new, "t": t_ms,
                           "new_pickup": self.state not in DISTRACTIONS})
            self._open_distraction = (new, t_ms)
        self.state = new
        return events
