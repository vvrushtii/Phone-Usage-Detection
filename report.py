"""The session report: three files in sessions/<date_time>/, then the picture on screen.

    events.csv    one row per distraction or AWAY period, in time order: start, end, type, duration_s, new_pickup
    summary.json  totals and the settings the session ran with
    report.png    a stats panel and a coloured timeline of FOCUSED / ON_PHONE / ON_CALL / AWAY

Only event times and totals are written. No video or frames.
"""
from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the report is drawn to a file; OpenCV shows it
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402

from app.hud import STATE_COLORS, STATE_LABELS  # noqa: E402
from app.session import Session  # noqa: E402
from ml.smoothing import AWAY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "sessions"
STATE_ORDER = ("FOCUSED", "ON_PHONE", "ON_CALL", "AWAY")
SHORT_SESSION_S = 180  # shorter sessions get a seconds axis
SCREEN_WIDTH = 1280  # the report window is shrunk to this width if the picture is wider


def _rgb(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    return bgr[2] / 255, bgr[1] / 255, bgr[0] / 255


def _fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def write_report(session: Session, meta: dict, root: Path = SESSIONS, folder: Path | None = None) -> Path:
    """Write the three files for an ended session and return the folder (root/<date_time> unless a folder is given)."""
    folder = folder or root / session.wall_start.strftime("%Y-%m-%d_%H-%M-%S")
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"started": session.wall_start.isoformat(timespec="seconds"),
               "ended": session.wall_end.isoformat(timespec="seconds") if session.wall_end else None,
               **session.summary(), **meta}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    with (folder / "events.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "type", "duration_s", "new_pickup"])
        rows = [(d.start_ms, d.end_ms if d.end_ms is not None else d.start_ms, d.type, d.new_pickup) for d in session.distractions]
        rows += [(s.start_ms, s.end_ms if s.end_ms is not None else s.start_ms, s.state, False)
                 for s in session.segments if s.state == AWAY]
        for start_ms, end_ms, kind, new_pickup in sorted(rows):
            start = session.wall_start + timedelta(milliseconds=start_ms)
            end = session.wall_start + timedelta(milliseconds=end_ms)
            writer.writerow([start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"), kind,
                             round((end_ms - start_ms) / 1000, 1), new_pickup])
    _draw(session, summary, folder / "report.png")
    return folder


def _draw(session: Session, summary: dict, path: Path) -> None:
    fig = plt.figure(figsize=(12, 6.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.35)
    stats = fig.add_subplot(grid[0])
    stats.axis("off")
    cards = [("Focused", f"{summary['percent_focused']:.0f}%"), ("Phone time", _fmt(summary["phone_s"])),
             ("Pickups", str(summary["pickups"])), ("Longest focus", _fmt(summary["longest_focus_streak_s"])),
             ("Avg distraction", _fmt(summary["average_distraction_s"])), ("Away", _fmt(summary["away_s"]))]
    for i, (label, value) in enumerate(cards):
        x = 0.02 + i * 0.165
        stats.text(x, 0.52, value, fontsize=26, fontweight="bold", transform=stats.transAxes)
        stats.text(x, 0.30, label, fontsize=12, color="#555", transform=stats.transAxes)
    stats.set_title(f"Focus Coach session, {session.wall_start:%A %d %B %Y, %H:%M}, total {_fmt(summary['total_s'])}",
                    loc="left", fontsize=14)

    ax = fig.add_subplot(grid[1])
    unit, ms_per_unit = ("seconds", 1000) if summary["total_s"] < SHORT_SESSION_S else ("minutes", 60000)
    total = max(summary["total_s"], 1) * 1000 / ms_per_unit
    for seg in session.segments:
        end = seg.end_ms if seg.end_ms is not None else session.elapsed_ms
        if end > seg.start_ms:
            ax.barh(0, (end - seg.start_ms) / ms_per_unit, left=seg.start_ms / ms_per_unit, height=0.6,
                    color=_rgb(STATE_COLORS[seg.state]), edgecolor="none")
    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"{unit} into the session")
    ax.set_title("Timeline", loc="left", fontsize=12)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=_rgb(STATE_COLORS[s])) for s in STATE_ORDER],
              labels=[STATE_LABELS[s].capitalize() for s in STATE_ORDER], ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), frameon=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def show_report(png: Path) -> None:
    """Open the report picture in an OpenCV window; any key (or closing the window) continues."""
    img = cv2.imread(str(png))
    if img is None:
        return
    if img.shape[1] > SCREEN_WIDTH:
        scale = SCREEN_WIDTH / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    title = "Focus Coach report (press any key)"
    cv2.imshow(title, img)
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) != -1:
            break
    cv2.destroyWindow(title)
