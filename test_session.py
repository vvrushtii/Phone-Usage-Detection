"""Session counters, pause/resume and the report files, driven by fake frames (no camera, no model)."""
import csv
import json

import pytest

from app.report import write_report
from app.session import Session
from ml.rules import FOCUSED, ON_CALL, ON_PHONE
from ml.smoothing import AWAY, NO_HEAD

CFG = {"window_ms": 1500, "enter_ratio": 0.6, "away_ms": 3000}
STEP = 100


def feed(session: Session, t0: float, segments: list[tuple[str, float]]) -> float:
    t = t0
    for state, seconds in segments:
        for _ in range(int(seconds * 1000 / STEP)):
            session.update(t, state)
            t += STEP
    return t


def started() -> Session:
    s = Session(CFG)
    s.start(0.0)
    return s


def test_one_pickup_with_phone_time_and_streak():
    s = started()
    t = feed(s, 0, [(FOCUSED, 10), (ON_PHONE, 5), (FOCUSED, 10)])
    s.end(t)
    assert s.pickups == 1
    assert s.phone_ms == pytest.approx(5000, abs=STEP)
    # the shown state changes about 0.9 s after the real change, so the streak includes that delay
    assert s.longest_streak_ms == pytest.approx(10_900, abs=2 * STEP)
    assert len(s.distractions) == 1 and s.distractions[0].end_ms is not None


def test_glance_changes_nothing():
    s = started()
    t = feed(s, 0, [(FOCUSED, 5), (ON_PHONE, 0.5), (FOCUSED, 5)])
    s.end(t)
    assert s.pickups == 0 and s.phone_ms == 0


def test_call_after_phone_is_one_pickup_two_events():
    s = started()
    t = feed(s, 0, [(FOCUSED, 3), (ON_PHONE, 3), (ON_CALL, 3), (FOCUSED, 4)])
    s.end(t)
    assert s.pickups == 1 and len(s.distractions) == 2


def test_away_is_not_phone_time_and_is_counted_separately():
    s = started()
    t = feed(s, 0, [(FOCUSED, 5), (NO_HEAD, 6)])
    s.end(t)
    assert s.phone_ms == 0
    assert s.time_in(AWAY) == pytest.approx(6000, abs=2 * STEP)  # dated from the last frame with a head


def test_session_ended_while_on_phone_closes_the_event():
    s = started()
    t = feed(s, 0, [(FOCUSED, 3), (ON_PHONE, 4)])
    s.end(t)
    assert s.distractions[-1].end_ms is not None
    assert s.summary()["pickups"] == 1


def test_pause_stops_the_clock_and_closes_the_distraction():
    s = started()
    t = feed(s, 0, [(FOCUSED, 3), (ON_PHONE, 3)])
    s.pause(t)
    assert s.update(t + 50, ON_PHONE) == []  # ignored while paused
    s.resume(t + 60_000)  # a minute of pause
    t2 = feed(s, t + 60_000, [(FOCUSED, 3)])
    s.end(t2)
    assert s.elapsed_ms == pytest.approx(9000, abs=2 * STEP)  # the minute is not counted
    assert s.pickups == 1


def test_summary_numbers_add_up():
    s = started()
    t = feed(s, 0, [(FOCUSED, 10), (ON_PHONE, 5), (FOCUSED, 5)])
    s.end(t)
    d = s.summary()
    assert d["focused_s"] + d["phone_s"] + d["away_s"] == pytest.approx(d["total_s"], abs=0.3)
    assert 0 < d["percent_focused"] < 100


def test_report_writes_three_files(tmp_path):
    s = started()
    t = feed(s, 0, [(FOCUSED, 5), (ON_PHONE, 4), (FOCUSED, 5), (NO_HEAD, 5)])
    s.end(t)
    folder = write_report(s, {"source": "test", "imgsz": 640}, root=tmp_path)
    assert {p.name for p in folder.iterdir()} == {"events.csv", "summary.json", "report.png"}
    rows = list(csv.DictReader((folder / "events.csv").open()))
    assert [r["type"] for r in rows] == [ON_PHONE, AWAY]  # the AWAY period is listed too, after the phone use
    assert float(rows[0]["duration_s"]) == pytest.approx(4, abs=0.3) and rows[1]["new_pickup"] == "False"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["pickups"] == 1 and summary["source"] == "test"
    assert (folder / "report.png").stat().st_size > 5000


def test_report_for_an_empty_session_does_not_crash(tmp_path):
    s = started()
    s.end(0.0)
    assert (write_report(s, {}, root=tmp_path) / "report.png").exists()
