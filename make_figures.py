"""Collect and draw the report figures into report/figures/ (200 dpi).

Run from the repo root, after the earlier phases have produced their results:
    python report/make_figures.py
New figures (pipeline, rule geometry, dataset samples + box sizes, side-by-side confusion matrices, MobileNet curves) are
drawn here from files in results/ and data/; the others are copied from where the earlier phases saved them.
fig10_session_report.png comes from `python report/make_sample_session.py`.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import detections as dc  # noqa: E402
from ml import rules, splits  # noqa: E402
from ml.curves import plot_curves  # noqa: E402

FIG = ROOT / "report/figures"
RESULTS = ROOT / "results"
DPI = 200
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# figure file -> where an earlier phase saved it
COPIES = {
    "fig04a_training_curves.png": "results/train/results.png",
    "fig04b_before_after.png": "results/train/before_after.png",
    "fig05_map_by_size.png": "results/eval/detection_by_size.png",
    "fig07a_failures_fp.png": "results/eval/failures_fp.png",
    "fig07b_failures_fn.png": "results/eval/failures_fn.png",
    "fig08a_enhancement_examples.png": "results/lowlight/enhancement_examples.png",
    "fig08b_lowlight_map.png": "results/lowlight/lowlight_map.png",
    "fig09b_mobilenet_vs_rules.png": "results/mobilenet/examples.png",
}


def pipeline() -> None:
    """Figure 1: the Focus Coach pipeline, also saved as Mermaid text."""
    steps = ["Webcam\n(OpenCV)", "Brightness\ncheck", "CLAHE + blur\n(only if dark)", "YOLO26n\nhead + phone", "Rules\nON_CALL,\nON_PHONE,\nFOCUSED",
             "Time\nsmoothing", "Events", "Session\nreport"]
    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.set_xlim(0, len(steps) * 1.6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, text in enumerate(steps):
        x = 0.1 + i * 1.6
        optional = "only if dark" in text
        ax.add_patch(FancyBboxPatch((x, 0.2), 1.25, 0.6, boxstyle="round,pad=0.02", fc="#fff3bf" if optional else "#d0ebff",
                                    ec="#495057", lw=1.2))
        ax.text(x + 0.625, 0.5, text, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.27, 0.5), (x + 1.6, 0.5), arrowstyle="-|>", mutation_scale=14, color="#495057"))
    fig.savefig(FIG / "fig01_pipeline.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    (FIG / "fig01_pipeline.mmd").write_text(
        "flowchart LR\n  A[Webcam - OpenCV] --> B[Brightness check]\n  B --> C[CLAHE + blur, only if dark]\n"
        "  C --> D[YOLO26n: head + phone]\n  D --> E[Rules: ON_CALL / ON_PHONE / FOCUSED]\n  E --> F[Time smoothing]\n"
        "  F --> G[Events]\n  G --> H[Session report]\n")


def rule_geometry() -> None:
    """Figure 2: one photo with the head box, the ON_CALL zone and the ON_PHONE reach zone, from the tuned values."""
    cfg = rules.load_config()
    held = splits.load_split("held_out")
    cache = dc.get_detections("fine_tuned", [dc.IMAGES / r.split / r.image_name for r in held.itertuples()])
    for r in held.itertuples():
        path = dc.IMAGES / r.split / r.image_name
        dets = [d for d in dc.as_rule_dets(cache[path.stem]) if d["conf"] >= 0.8]
        heads = [d for d in dets if d["cls"] == "head"]
        phones = [d for d in dets if d["cls"] == "phone"]
        if len(heads) == 1 and len(phones) == 1 and rules.classify_image(dets, cfg, "largest")["persons"][0]["state"] == rules.ON_PHONE:
            break
    else:
        raise RuntimeError("no suitable photo for the rule geometry figure")
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    hx1, hy1, hx2, hy2 = heads[0]["box"]
    hw, hh = hx2 - hx1, hy2 - hy1
    hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    grow = cfg["call"]["head_expand"]
    reach_x, reach_y = cfg["using"]["max_dx_head_widths"] * hw, cfg["using"]["max_dy_head_heights"] * hh
    pad = int(max(reach_x - hcx, hcx + reach_x - img.shape[1], 0)) + 10
    pad_y = int(max(hcy + reach_y - img.shape[0], 0)) + 10
    canvas = cv2.copyMakeBorder(img, 0, pad_y, pad, pad, cv2.BORDER_CONSTANT, value=(235, 235, 235))
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * canvas.shape[0] / canvas.shape[1]))
    ax.imshow(canvas, extent=(-pad, img.shape[1] + pad, img.shape[0] + pad_y, 0))
    ax.add_patch(Rectangle((hx1 - grow * hw, hy1 - grow * hh), hw * (1 + 2 * grow), hh * (1 + 2 * grow), fill=False, ec="#c92a2a",
                           lw=2, ls="--", label=f"ON_CALL zone (head grown by {grow:g})"))
    ax.add_patch(Rectangle((hcx - reach_x, hcy), 2 * reach_x, reach_y, fill=False, ec="#e67700", lw=2, ls="--",
                           label=f"ON_PHONE reach (±{cfg['using']['max_dx_head_widths']:g} head widths sideways, "
                                 f"{cfg['using']['max_dy_head_heights']:g} head heights down from the head centre)"))
    for d, colour in ((heads[0], "#1c7ed6"), (phones[0], "#2b8a3e")):
        x1, y1, x2, y2 = d["box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=colour, lw=2.5))
        ax.text(x1, y1 - 4, d["cls"], color=colour, fontsize=9, fontweight="bold")
    ax.plot([hcx], [hcy], "o", color="#1c7ed6")
    ax.set_xlim(-pad, img.shape[1] + pad)
    ax.set_ylim(img.shape[0] + pad_y, 0)
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False)
    ax.set_title("Usage rules drawn with the tuned values from shared/rules.json", fontsize=10)
    fig.savefig(FIG / "fig02_rule_geometry.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def dataset() -> None:
    """Figure 3: sample images with their boxes (Phase 1 contact sheet) and the box-size histogram."""
    sheet = cv2.cvtColor(cv2.imread(str(RESULTS / "label_check/train.jpg")), cv2.COLOR_BGR2RGB)
    areas = {"phone": [], "head": []}
    for file in (ROOT / "data/fpi_yolo/labels/train").glob("*.txt"):
        for line in file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                areas["phone" if int(parts[0]) == 0 else "head"].append(float(parts[3]) * float(parts[4]))
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={"height_ratios": [1, 2]})
    a.imshow(sheet)
    a.axis("off")
    a.set_title("Training images with their boxes")
    bins = np.logspace(-4, 0, 40)
    for cls, colour in (("head", "#1c7ed6"), ("phone", "#e8590c")):
        b.hist(areas[cls], bins=bins, alpha=0.65, color=colour, label=f"{cls} (n={len(areas[cls])})")
    b.set_xscale("log")
    b.set_xlabel("box area as a fraction of the image")
    b.set_ylabel("boxes")
    b.set_title("Box sizes, training split")
    b.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig03_dataset.png", dpi=DPI)
    plt.close(fig)


def confusion_panels() -> None:
    """Figure 6: confusion matrices of A-ft, B-rules and C-mobilenet side by side."""
    data = json.loads((RESULTS / "eval/classification.json").read_text())["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, name in zip(axes, ("A-ft", "B-rules", "C-mobilenet")):
        c = data[name]["confusion"]
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        pct = m / m.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center", color="white" if pct[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["not using", "using"])
        ax.set_yticks([0, 1], ["not using", "using"])
        ax.set_xlabel("predicted")
        ax.set_title(f"{name}  (accuracy {data[name]['metrics']['accuracy']:.3f})")
    axes[0].set_ylabel("true label")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_confusion.png", dpi=DPI)
    plt.close(fig)


def mobilenet_curves() -> None:
    history = json.loads((RESULTS / "mobilenet/history.json").read_text())
    plot_curves({k: v for k, v in history.items() if isinstance(v, list)}, history["stage1_epochs"], FIG / "fig09a_mobilenet_curves.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pipeline()
    rule_geometry()
    dataset()
    confusion_panels()
    mobilenet_curves()
    missing = []
    for name, source in COPIES.items():
        if (ROOT / source).exists():
            shutil.copy(ROOT / source, FIG / name)
        else:
            missing.append(source)
    if not (FIG / "fig10_session_report.png").exists():
        missing.append("report/figures/fig10_session_report.png (run python report/make_sample_session.py)")
    print("figures in", FIG, "| missing:", missing or "none")


if __name__ == "__main__":
    main()
