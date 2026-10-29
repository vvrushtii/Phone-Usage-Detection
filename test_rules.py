"""Tests for ml/rules.py with hand-built boxes.

Reference head H = [100, 100, 200, 200]: 100 x 100, centre (150, 150).
With head_expand 0.25 the call zone is [75, 75, 225, 225]; phone reach is 150 px sideways, 300 px down.
"""
from ml import rules

CFG = {
    "conf": {"phone": 0.3, "head": 0.3},
    "call": {"head_expand": 0.25},
    "using": {"max_dx_head_widths": 1.5, "max_dy_head_heights": 3.0},
}
H = {"cls": "head", "box": [100, 100, 200, 200], "conf": 0.9}


def det(cls: str, box: list[float], conf: float = 0.9) -> dict:
    return {"cls": cls, "box": box, "conf": conf}


def phone(box: list[float], conf: float = 0.9) -> dict:
    return det("phone", box, conf)


def state(*phones: dict) -> str:
    return rules.classify_person(H, list(phones), CFG)


# --- filter_dets ---------------------------------------------------------------------------------

def test_filter_dets_uses_the_threshold_of_each_class():
    dets = [det("head", [0, 0, 1, 1], 0.3), det("head", [0, 0, 1, 1], 0.29),
            det("phone", [0, 0, 1, 1], 0.3), det("phone", [0, 0, 1, 1], 0.29)]
    kept = rules.filter_dets(dets, CFG)
    assert [(d["cls"], d["conf"]) for d in kept] == [("head", 0.3), ("phone", 0.3)]  # threshold is inclusive


def test_filter_dets_thresholds_can_differ_per_class():
    cfg = {**CFG, "conf": {"phone": 0.6, "head": 0.1}}
    kept = rules.filter_dets([det("head", [0, 0, 1, 1], 0.2), det("phone", [0, 0, 1, 1], 0.5)], cfg)
    assert [d["cls"] for d in kept] == ["head"]


# --- pick_user_head ------------------------------------------------------------------------------

def test_pick_largest_head():
    small, big = det("head", [0, 0, 10, 10]), det("head", [0, 0, 50, 50])
    assert rules.pick_user_head([small, big], "largest") == [big]


def test_pick_largest_breaks_ties_by_confidence_then_left_edge():
    a = det("head", [0, 0, 10, 10], 0.5)
    b = det("head", [100, 0, 110, 10], 0.8)
    c = det("head", [50, 0, 60, 10], 0.8)
    assert rules.pick_user_head([a, b, c], "largest") == [c]


def test_pick_all_and_empty():
    heads = [det("head", [0, 0, 10, 10]), det("head", [20, 0, 30, 10])]
    assert rules.pick_user_head(heads, "all") == heads
    assert rules.pick_user_head([], "largest") == []


def test_pick_unknown_mode_raises():
    try:
        rules.pick_user_head([], "nearest")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --- classify_person: ON_CALL ----------------------------------------------------------------------

def test_no_phone_is_focused():
    assert state() == rules.FOCUSED


def test_phone_overlapping_the_head_is_on_call():
    assert state(phone([190, 120, 230, 160])) == rules.ON_CALL


def test_phone_inside_the_margin_but_not_touching_the_head_is_on_call():
    assert state(phone([215, 100, 260, 140])) == rules.ON_CALL  # reaches x = 215 < 225


def test_phone_just_outside_the_margin_is_not_on_call():
    assert state(phone([226, 100, 260, 140])) == rules.FOCUSED  # centre is above the head centre, so not ON_PHONE


def test_phone_only_touching_the_zone_edge_does_not_count():
    assert state(phone([225, 100, 260, 140])) == rules.FOCUSED


def test_margin_follows_head_expand():
    wide = {**CFG, "call": {"head_expand": 0.5}}  # zone grows to x = 250
    assert rules.classify_person(H, [phone([226, 100, 260, 140])], wide) == rules.ON_CALL


# --- classify_person: ON_PHONE ---------------------------------------------------------------------

def test_phone_below_the_face_is_on_phone():
    assert state(phone([130, 300, 170, 340])) == rules.ON_PHONE


def test_vertical_reach_boundary():
    assert state(phone([130, 430, 170, 470])) == rules.ON_PHONE  # centre y 450: exactly 300 below, allowed
    assert state(phone([130, 431, 170, 471])) == rules.FOCUSED   # one pixel further


def test_horizontal_reach_boundary():
    assert state(phone([280, 300, 320, 340])) == rules.ON_PHONE  # centre x 300: exactly 150 sideways, allowed
    assert state(phone([281, 300, 321, 340])) == rules.FOCUSED   # one pixel further


def test_phone_above_the_head_centre_is_not_on_phone():
    assert state(phone([130, 10, 170, 50])) == rules.FOCUSED


def test_phone_far_away_on_a_desk_is_focused():
    assert state(phone([600, 400, 640, 440])) == rules.FOCUSED


def test_reach_scales_with_head_size():
    big = det("head", [100, 100, 300, 300])  # twice as large: reach is 300 px sideways, 600 px down
    assert rules.classify_person(big, [phone([380, 600, 420, 640])], CFG) == rules.ON_PHONE
    assert rules.classify_person(H, [phone([380, 600, 420, 640])], CFG) == rules.FOCUSED


def test_on_call_wins_when_both_match():
    assert state(phone([130, 300, 170, 340]), phone([190, 120, 230, 160])) == rules.ON_CALL


# --- assign_phones -----------------------------------------------------------------------------

def test_each_phone_goes_to_its_nearest_head_only():
    a, b = det("head", [0, 0, 100, 100]), det("head", [300, 0, 400, 100])
    p = phone([190, 40, 230, 80])  # centre (210, 60): nearer to b
    assert rules.assign_phones([a, b], [p]) == [[], [p]]


def test_assignment_divides_distance_by_head_height():
    tall, short = det("head", [0, 0, 100, 200]), det("head", [300, 50, 400, 150])
    p = phone([180, 80, 220, 120])  # centre (200, 100): 150 px from both centres, tall head scores lower
    assert rules.assign_phones([tall, short], [p]) == [[p], []]


def test_phone_overlapping_two_heads_belongs_to_one():
    a, b = det("head", [0, 0, 100, 100]), det("head", [80, 0, 180, 100])
    p = phone([50, 20, 90, 60])  # overlaps both boxes, but its centre is nearer to a
    persons = rules.classify_image([a, b, p], CFG)["persons"]
    assert [x["state"] for x in persons] == [rules.ON_CALL, rules.FOCUSED]
    assert persons[1]["phones"] == []


# --- classify_image ------------------------------------------------------------------------------

def test_image_is_using_if_any_person_is():
    a = det("head", [50, 50, 150, 150])
    b = det("head", [600, 50, 700, 150])
    result = rules.classify_image([a, b, phone([80, 300, 120, 340])], CFG, mode="all")
    assert result["using"] is True
    assert [x["state"] for x in result["persons"]] == [rules.ON_PHONE, rules.FOCUSED]


def test_largest_mode_looks_only_at_the_main_person():
    small = det("head", [50, 50, 150, 150])
    big = det("head", [900, 100, 1100, 300])
    dets = [small, big, phone([80, 300, 120, 340])]  # phone is below the small head, far from the big one
    assert rules.classify_image(dets, CFG, mode="all")["using"] is True
    assert rules.classify_image(dets, CFG, mode="largest")["using"] is False


def test_no_head_means_not_using_even_with_a_phone():
    result = rules.classify_image([phone([130, 300, 170, 340])], CFG)
    assert result == {"using": False, "persons": []}


def test_low_confidence_phone_is_ignored_inside_classify_image():
    assert rules.classify_image([H, phone([190, 120, 230, 160], conf=0.2)], CFG)["using"] is False
    assert rules.classify_image([H, phone([190, 120, 230, 160], conf=0.5)], CFG)["using"] is True


def test_empty_image():
    assert rules.classify_image([], CFG) == {"using": False, "persons": []}


def test_shipped_config_has_every_key_the_rules_read():
    cfg = rules.load_config()
    assert set(cfg["conf"]) == {"phone", "head"}
    assert "head_expand" in cfg["call"]
    assert {"max_dx_head_widths", "max_dy_head_heights"} <= set(cfg["using"])
    assert {"window_ms", "enter_ratio", "away_ms"} <= set(cfg["smoothing"])
