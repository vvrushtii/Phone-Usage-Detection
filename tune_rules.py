"""Tune the rule thresholds on the TUNE split of label.csv (never on held-out).

Run from the repo root:
    python ml/tune_rules.py

Runs the detector once (cached by ml/detections.py), then sweeps thresholds in pure Python. The objective is the F1 of
"using phone". Writes the best values back into shared/rules.json and saves
    results/eval/tuning.csv                 every combination tried, with its metrics
    results/eval/tuning_summary.json        best combination and the baselines' thresholds
    results/eval/baseline_thresholds.json   best phone confidence for the "any phone" baselines (A-coco, A-ft)
    results/eval/tuning_sensitivity.png     how the two most influential parameters move F1
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import detections as det_cache  # noqa: E402
from ml import rules, splits  # noqa: E402
from ml.usage_metrics import confusion, metrics  # noqa: E402

OUT = ROOT / "results/eval"
# (section, key) -> values to try. 5 x 4 x 3 x 3 x 3 = 540 combinations. A first pass over a coarser grid put the best
# value of conf.head, max_dx and max_dy on the grid's edge, so the grid was widened that way (still tune split only).
GRID: dict[tuple[str, str], list[float]] = {
    ("conf", "phone"): [0.1, 0.2, 0.3, 0.4, 0.5],
    ("conf", "head"): [0.1, 0.175, 0.25, 0.4],
    ("call", "head_expand"): [0.1, 0.25, 0.4],
    ("using", "max_dx_head_widths"): [1.5, 2.0, 3.0],
    ("using", "max_dy_head_heights"): [1.0, 2.0, 3.0],
}
BASELINE_CONFS = [round(c, 2) for c in np.arange(0.05, 0.951, 0.05)]


def load_tune_set(model_key: str) -> tuple[list[dict], list[int]]:
    """Cached detections (as rule dets, low confidences already dropped) and 0/1 labels of the tune images."""
    tune = splits.load_split("tune")
    paths = [det_cache.IMAGES / row.split / row.image_name for row in tune.itertuples()]
    cache = det_cache.get_detections(model_key, paths)
    # the lowest confidence anything below will ask for: the baselines sweep phone confidence from 0.05
    floor = {"phone": min(BASELINE_CONFS + GRID[("conf", "phone")]), "head": min(GRID[("conf", "head")])}
    dets = [det_cache.as_rule_dets(cache[p.stem], floor) for p in paths]
    return dets, tune["using_phone"].tolist()


def with_params(base: dict, combo: dict[tuple[str, str], float]) -> dict:
    cfg = copy.deepcopy(base)
    for (section, key), value in combo.items():
        cfg[section][key] = value
    return cfg


def sweep_rules(base: dict, dets: list[list[dict]], y: list[int]) -> pd.DataFrame:
    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        combo = dict(zip(keys, values))
        cfg = with_params(base, combo)
        pred = [int(rules.classify_image(d, cfg)["using"]) for d in dets]
        c = confusion(y, pred)
        rows.append({**{f"{s}.{k}": v for (s, k), v in combo.items()}, **c, **metrics(c)})
    return pd.DataFrame(rows)


def best_row(df: pd.DataFrame) -> pd.Series:
    # F1 first, accuracy second; stable sort keeps grid order on exact ties, so the result is deterministic
    return df.sort_values(["f1", "accuracy"], ascending=False, kind="stable").iloc[0]


def baseline_threshold(dets: list[list[dict]], y: list[int]) -> dict:
    """Best phone confidence for the 'any phone detected -> using' rule."""
    best = None
    for conf in BASELINE_CONFS:
        pred = [int(any(d["cls"] == "phone" and d["conf"] >= conf for d in ds)) for ds in dets]
        m = metrics(confusion(y, pred))
        if best is None or (m["f1"], m["accuracy"]) > (best["tune_f1"], best["tune_accuracy"]):
            best = {"phone_conf": conf, "tune_f1": m["f1"], "tune_accuracy": m["accuracy"]}
    return best


def influence(df: pd.DataFrame) -> dict[str, float]:
    """How much F1 moves when one parameter changes: spread of its per-value mean F1."""
    cols = [f"{s}.{k}" for s, k in GRID]
    return {c: float(df.groupby(c)["f1"].mean().max() - df.groupby(c)["f1"].mean().min()) for c in cols}


def plot_sensitivity(df: pd.DataFrame, infl: dict[str, float], path: Path) -> list[str]:
    top = sorted(infl, key=infl.get, reverse=True)[:2]
    grid = df.groupby(top)["f1"].max().unstack()  # best F1 over the other three parameters
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid.values, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), [str(v) for v in grid.columns])
    ax.set_yticks(range(len(grid.index)), [str(v) for v in grid.index])
    ax.set_xlabel(top[1])
    ax.set_ylabel(top[0])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="best F1 on tune split (over the other parameters)")
    ax.set_title("Sensitivity of the two most influential parameters")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return top


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = rules.load_config()
    dets, y = load_tune_set("fine_tuned")
    print(f"tune split: {len(y)} images, {sum(y)} using phone; sweeping {np.prod([len(v) for v in GRID.values()])} combinations")

    df = sweep_rules(base, dets, y)
    df.to_csv(OUT / "tuning.csv", index=False)
    best = best_row(df)
    infl = influence(df)
    top = plot_sensitivity(df, infl, OUT / "tuning_sensitivity.png")

    tuned = with_params(base, {(s, k): float(best[f"{s}.{k}"]) for s, k in GRID})
    (ROOT / "shared/rules.json").write_text(json.dumps(tuned, indent=2) + "\n")

    coco_dets, _ = load_tune_set("coco")
    baselines = {"A-ft": baseline_threshold(dets, y), "A-coco": baseline_threshold(coco_dets, y)}
    (OUT / "baseline_thresholds.json").write_text(json.dumps(baselines, indent=2))
    summary = {
        "split": "tune", "images": len(y), "objective": "F1 of 'using phone'",
        "best": {k: (float(v) if isinstance(v, float) else v) for k, v in best.to_dict().items()},
        "influence_f1_spread": infl, "plotted_parameters": top, "baselines": baselines,
    }
    (OUT / "tuning_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\nbest rule thresholds (tune split):")
    for s, k in GRID:
        print(f"  {s}.{k} = {best[f'{s}.{k}']}")
    print(f"  F1 {best['f1']:.4f}  accuracy {best['accuracy']:.4f}  precision {best['precision']:.4f}  recall {best['recall']:.4f}")
    print("most influential:", ", ".join(f"{k} ({v:.3f})" for k, v in sorted(infl.items(), key=lambda kv: -kv[1])[:3]))
    print("baselines (tune split):", {k: f"conf {v['phone_conf']} F1 {v['tune_f1']:.4f} acc {v['tune_accuracy']:.4f}" for k, v in baselines.items()})
    print("wrote shared/rules.json and results/eval/tuning*.{csv,json,png}")


if __name__ == "__main__":
    main()
