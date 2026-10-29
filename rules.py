"""Usage rules: decide what a person is doing from head and phone boxes.

Pure functions. The only I/O is load_config(), which reads shared/rules.json, the single place where
thresholds live. Boxes are [x1, y1, x2, y2] in pixels; a detection is
{"cls": "head" | "phone", "box": [x1, y1, x2, y2], "conf": float}.

States: ON_CALL (phone at the ear or face), ON_PHONE (phone held in front, below the face), FOCUSED.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).resolve().parents[1] / "shared" / "rules.json"

ON_CALL = "ON_CALL"
ON_PHONE = "ON_PHONE"
FOCUSED = "FOCUSED"

Det = dict[str, Any]
Box = list[float]


def load_config(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _intersects(a: Box, b: Box) -> bool:
    """True when the boxes share some area; boxes that only touch at an edge do not count."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def filter_dets(dets: list[Det], cfg: dict) -> list[Det]:
    """Keep detections whose confidence reaches the threshold for their class."""
    return [d for d in dets if d["conf"] >= cfg["conf"][d["cls"]]]


def pick_user_head(heads: list[Det], mode: str) -> list[Det]:
    """'largest': the one person nearest the webcam (app). 'all': every head (evaluation of still images)."""
    if mode == "all":
        return list(heads)
    if mode == "largest":
        if not heads:
            return []
        return [max(heads, key=lambda h: (_area(h["box"]), h["conf"], -h["box"][0]))]
    raise ValueError(f"unknown mode {mode!r}")


def classify_person(head: Det, phones: list[Det], cfg: dict) -> str:
    """State of one person from their head box and the phones that belong to them."""
    hx1, hy1, hx2, hy2 = head["box"]
    hw, hh = hx2 - hx1, hy2 - hy1
    hcx, hcy = _center(head["box"])

    grow = cfg["call"]["head_expand"]
    call_zone = [hx1 - grow * hw, hy1 - grow * hh, hx2 + grow * hw, hy2 + grow * hh]
    if any(_intersects(p["box"], call_zone) for p in phones):
        return ON_CALL  # wins over ON_PHONE when both match

    reach_x = cfg["using"]["max_dx_head_widths"] * hw
    reach_y = cfg["using"]["max_dy_head_heights"] * hh
    for p in phones:
        pcx, pcy = _center(p["box"])
        if pcy > hcy and abs(pcx - hcx) <= reach_x and pcy - hcy <= reach_y:
            return ON_PHONE
    return FOCUSED


def assign_phones(heads: list[Det], phones: list[Det]) -> list[list[Det]]:
    """Give each phone to at most one head: the nearest, by centre distance divided by head height."""
    owned: list[list[Det]] = [[] for _ in heads]
    for p in phones:
        pcx, pcy = _center(p["box"])
        scores = []
        for h in heads:
            hcx, hcy = _center(h["box"])
            scores.append(((pcx - hcx) ** 2 + (pcy - hcy) ** 2) ** 0.5 / max(h["box"][3] - h["box"][1], 1e-6))
        if scores:
            owned[scores.index(min(scores))].append(p)
    return owned


def classify_image(dets: list[Det], cfg: dict, mode: str = "all") -> dict:
    """Image-level decision: 'using' is true if any person is ON_CALL or ON_PHONE.

    A phone with no head in the picture is not counted: the rules decide from the head-phone relation.
    """
    dets = filter_dets(dets, cfg)
    heads = pick_user_head([d for d in dets if d["cls"] == "head"], mode)
    phones = [d for d in dets if d["cls"] == "phone"]
    persons = []
    for head, owned in zip(heads, assign_phones(heads, phones)):
        persons.append({"head": head["box"], "state": classify_person(head, owned, cfg),
                        "phones": [p["box"] for p in owned]})
    return {"using": any(p["state"] != FOCUSED for p in persons), "persons": persons}
