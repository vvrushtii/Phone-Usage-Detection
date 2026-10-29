"""Smoothing with fake timestamps: glances, sustained use, flicker, AWAY. Window 1500 ms, ratio 0.6, away 3000 ms."""
import pytest

from ml.rules import FOCUSED, ON_CALL, ON_PHONE
from ml.smoothing import AWAY, NO_HEAD, StateSmoother

STEP = 100  # ms between frames (10 fps)


def run(states_by_second: list[tuple[str, float]], smoother: StateSmoother | None = None):
    """Feed (raw_state, seconds) segments at 10 fps; return every event and the smoother."""
    s = smoother or StateSmoother(1500, 0.6, 3000)
    events, t = [], 0
    for state, seconds in states_by_second:
        for _ in range(int(seconds * 1000 / STEP)):
            events += s.update(t, state)
            t += STEP
    return events, s


def kinds(events):
    return [e["kind"] for e in events]


def test_half_second_glance_logs_nothing():
    events, s = run([(FOCUSED, 3), (ON_PHONE, 0.5), (FOCUSED, 3)])
    assert events == [] and s.state == FOCUSED


def test_just_under_the_ratio_logs_nothing():
    # 0.8 s of phone is below 0.6 x 1.5 s = 0.9 s
    events, _ = run([(FOCUSED, 3), (ON_PHONE, 0.8), (FOCUSED, 3)])
    assert events == []


def test_sustained_use_is_exactly_one_event_with_the_right_duration():
    events, s = run([(FOCUSED, 3), (ON_PHONE, 5), (FOCUSED, 4)])
    starts = [e for e in events if e["kind"] == "distraction_start"]
    ends = [e for e in events if e["kind"] == "distraction_end"]
    assert len(starts) == len(ends) == 1
    assert starts[0]["type"] == ON_PHONE and starts[0]["new_pickup"] is True
    assert ends[0]["duration_ms"] == pytest.approx(5000, abs=STEP)
    assert s.state == FOCUSED


def test_flicker_on_the_boundary_does_not_double_count():
    # phone detected on and off every other frame: never 60% of the window, so nothing is committed
    s = StateSmoother(1500, 0.6, 3000)
    events = []
    for i in range(100):
        events += s.update(i * STEP, ON_PHONE if i % 2 else FOCUSED)
    assert events == []


def test_dropout_during_use_does_not_split_the_event():
    # one lost phone frame in the middle of sustained use
    events, _ = run([(FOCUSED, 3), (ON_PHONE, 3), (FOCUSED, 0.1), (ON_PHONE, 3), (FOCUSED, 4)])
    assert kinds(events).count("distraction_start") == 1


def test_switching_from_phone_to_call_is_one_pickup():
    events, _ = run([(FOCUSED, 3), (ON_PHONE, 3), (ON_CALL, 3), (FOCUSED, 4)])
    starts = [e for e in events if e["kind"] == "distraction_start"]
    assert [e["type"] for e in starts] == [ON_PHONE, ON_CALL]
    assert [e["new_pickup"] for e in starts] == [True, False]
    assert kinds(events).count("distraction_end") == 2


def test_no_head_for_longer_than_away_shows_away_dated_from_last_head():
    events, s = run([(FOCUSED, 2), (NO_HEAD, 4)])
    change = [e for e in events if e["kind"] == "state_change"]
    assert [e["to"] for e in change] == [AWAY]
    assert change[0]["t"] == pytest.approx(1900, abs=STEP)  # when the head was last seen
    assert s.state == AWAY


def test_short_head_loss_changes_nothing():
    events, s = run([(FOCUSED, 2), (NO_HEAD, 2.5), (FOCUSED, 2)])
    assert events == [] and s.state == FOCUSED


def test_away_is_not_phone_time_and_ends_a_distraction_at_the_last_head_frame():
    events, _ = run([(FOCUSED, 2), (ON_PHONE, 3), (NO_HEAD, 4)])
    end = [e for e in events if e["kind"] == "distraction_end"]
    assert len(end) == 1
    assert end[0]["t"] == pytest.approx(4900, abs=STEP)  # the last frame with a head, not 3 s later


def test_coming_back_from_away_returns_to_focused_and_away_is_reported_once():
    events, s = run([(FOCUSED, 1), (NO_HEAD, 5), (FOCUSED, 3)])
    assert [e["to"] for e in events if e["kind"] == "state_change"] == [AWAY, FOCUSED]
    assert s.state == FOCUSED


def test_stalled_camera_cannot_turn_one_frame_into_a_state():
    s = StateSmoother(1500, 0.6, 3000)
    s.update(0, FOCUSED)
    s.update(100, ON_PHONE)
    assert s.update(2500, FOCUSED) == []  # a 2.4 s stall after one phone frame


def test_finish_closes_an_open_distraction():
    events, s = run([(FOCUSED, 2), (ON_PHONE, 3)])
    closing = s.finish(5000)
    assert [e["kind"] for e in closing] == ["distraction_end"]
    assert s.finish(5000) == []


def test_reset_starts_clean():
    _, s = run([(FOCUSED, 2), (ON_PHONE, 3)])
    s.reset(10_000)
    assert s.state == FOCUSED and s.finish(10_000) == []
