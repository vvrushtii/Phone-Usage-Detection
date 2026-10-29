"""Build report/report.md, report/viva.md and the results block of README.md from the files in results/.

Run from the repo root:
    python report/build_report.py
Every number in the text and tables is read from results/*.json (and shared/rules.json) by ml/report_values.py; the
templates only contain {{placeholders}}. A placeholder with no value stops the build with its name, so a missing number is
never filled in by hand. After any re-evaluation, run this again and the report follows. "Narrative checks" at the end warn
when a result has changed enough that a sentence in the templates may no longer be true.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.report_values import METHODS, collect, render  # noqa: E402

REPORT = ROOT / "report"
README_START, README_END = "<!-- RESULTS:START (written by report/build_report.py) -->", "<!-- RESULTS:END -->"


# --- checks and rendering --------------------------------------------------------------------------------------

def narrative_checks(R: dict) -> list[str]:
    """Warnings when a sentence written in the templates may no longer match the numbers."""
    m = {n: R["cls"]["methods"][n]["metrics"] for n in METHODS.values()}
    ll = R["ll"]["test_map50"]
    checks = [
        (abs(m["B-rules"]["accuracy"] - m["A-ft"]["accuracy"]) < 0.01, "B-rules and A-ft are no longer within 0.01 accuracy ('level with')"),
        (m["C-mobilenet"]["accuracy"] < m["B-rules"]["accuracy"] - 0.05, "MobileNet is no longer clearly below B-rules"),
        (ll["dim"]["clahe_smooth"]["phone"] > ll["dim"]["none"]["phone"], "CLAHE + blur no longer helps on dim images"),
        (ll["dim"]["global_he"]["phone"] < ll["dim"]["none"]["phone"], "global HE no longer hurts on dim images"),
        (max(ll["dark"][k]["phone"] for k in ll["dark"]) < 0.05, "some method now works on dark images ('nothing helped')"),
        (ll["real_dark"]["none"]["phone"] >= max(ll["real_dark"][k]["phone"] for k in ll["real_dark"]) - 0.005, "enhancement now helps on real dark photos"),
        (R["det"]["phone"]["fine_tuned"]["map50"] > R["det"]["phone"]["yolo26n_coco"]["map50"] + 0.3, "fine-tuning gain is smaller than the text says"),
    ]
    return [msg for ok, msg in checks if not ok]


def update_readme(block: str) -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    if README_START not in text or README_END not in text:
        raise SystemExit(f"README.md has no {README_START} ... {README_END} markers")
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1]
    path.write_text(f"{head}{README_START}\n\n{block}\n\n{README_END}{tail}")


def abstract_words(text: str) -> int:
    m = re.search(r"## 1\. Abstract\n\n(.*?)\n\n## ", text, re.S)
    return len(m.group(1).split()) if m else 0


def main() -> None:
    values, R = collect()
    report = render((REPORT / "template.md").read_text(), values, "template.md")
    (REPORT / "report.md").write_text(report)
    (REPORT / "viva.md").write_text(render((REPORT / "viva_template.md").read_text(), values, "viva_template.md"))
    update_readme(values["readme_results"])
    words = abstract_words(report)
    print(f"wrote report/report.md ({len(report.split()):,} words), report/viva.md and the README results block; abstract {words} words")
    if words > 150:
        print(f"WARNING: the abstract is {words} words; the limit is 150")
    for warning in narrative_checks(R):
        print("WARNING (narrative may be stale):", warning)
    leftovers = sorted(set(re.findall(r"\[(?:CITE|SCREENSHOT|AUTHOR|GUIDE|CO)[^\]]*\]", report + (REPORT / "viva.md").read_text())))
    print(f"{len(leftovers)} placeholders left for the user:", *leftovers, sep="\n  ")


if __name__ == "__main__":
    sys.exit(main())
