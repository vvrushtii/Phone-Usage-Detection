"""The app's per-frame decisions, without a camera or a model."""
from app.pipeline import frame_state, should_enhance
from ml import rules
from ml.smoothing import NO_HEAD

CFG = rules.load_config()
HEAD = {"cls": "head", "box": [100, 100, 200, 200], "conf": 0.9}
APP = {"enhance_method": "clahe_smooth", "low_light_threshold": 29}


def phone(box, conf=0.9):
    return {"cls": "phone", "box": box, "conf": conf}


def test_no_head_is_reported_as_no_head():
    assert frame_state([phone([100, 220, 140, 260])], CFG) == (NO_HEAD, None)
    assert frame_state([], CFG) == (NO_HEAD, None)


def test_head_alone_is_focused():
    assert frame_state([HEAD], CFG) == (rules.FOCUSED, HEAD["box"])


def test_phone_in_front_is_on_phone_and_at_the_ear_is_on_call():
    assert frame_state([HEAD, phone([130, 230, 170, 280])], CFG)[0] == rules.ON_PHONE
    assert frame_state([HEAD, phone([190, 120, 220, 180])], CFG)[0] == rules.ON_CALL


def test_only_the_largest_head_counts():
    small = {"cls": "head", "box": [400, 100, 430, 130], "conf": 0.9}
    state, head = frame_state([small, HEAD, phone([130, 230, 170, 280])], CFG)
    assert head == HEAD["box"] and state == rules.ON_PHONE


def test_auto_enhancement_only_below_the_threshold():
    assert should_enhance(10, APP, "auto") and not should_enhance(60, APP, "auto")
    assert not should_enhance(10, {**APP, "enhance_method": "none"}, "auto")


def test_manual_override_beats_brightness():
    assert should_enhance(200, APP, "on") and not should_enhance(5, APP, "off")
