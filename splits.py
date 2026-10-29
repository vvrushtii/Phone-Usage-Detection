"""The tune / held-out split of label.csv, the only usage ground truth.

label.csv is the val and test images of the detector split. They are split here 30% tune / 70% held-out with a
fixed seed. Tune is used to choose thresholds (Phase 3) and to train the MobileNet baseline (Phase 4); every
reported number comes from held-out. Stratified by (source split, using phone) so both halves have the same
mix: val and test differ a lot in how many "using" images have no visible head.

Run from the repo root to (re)write the files:
    python ml/splits.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
USAGE_LABELS = ROOT / "data/fpi_yolo/usage_labels.csv"
SPLIT_DIR = ROOT / "results/eval/splits"
SEED = 42
TUNE_FRACTION = 0.30


def read_usage_labels() -> pd.DataFrame:
    """image_name, split (val | test, the detector split it came from), using_phone (0 | 1)."""
    return pd.read_csv(USAGE_LABELS)


def make_splits() -> dict[str, pd.DataFrame]:
    labels = read_usage_labels().sort_values("image_name").reset_index(drop=True)
    tune_parts = []
    for _, group in labels.groupby(["split", "using_phone"]):
        n_tune = round(TUNE_FRACTION * len(group))
        tune_parts.append(group.sample(n=n_tune, random_state=SEED))
    tune = pd.concat(tune_parts).sort_values("image_name")
    held_out = labels[~labels["image_name"].isin(tune["image_name"])]
    return {"tune": tune.reset_index(drop=True), "held_out": held_out.reset_index(drop=True)}


def write_splits() -> dict[str, pd.DataFrame]:
    parts = make_splits()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "tune_fraction": TUNE_FRACTION, "stratified_by": ["split", "using_phone"]}
    for name, df in parts.items():
        df.to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        summary[name] = {
            "images": len(df),
            "using_phone": int(df["using_phone"].sum()),
            "using_rate": round(float(df["using_phone"].mean()), 4),
            "from_val": int((df["split"] == "val").sum()),
            "from_test": int((df["split"] == "test").sum()),
        }
    (SPLIT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return parts


def load_split(name: str) -> pd.DataFrame:
    """Read a saved split ('tune' or 'held_out'), creating the files first if they do not exist."""
    path = SPLIT_DIR / f"{name}.csv"
    if not path.exists():
        write_splits()
    return pd.read_csv(path)


if __name__ == "__main__":
    for name, df in write_splits().items():
        print(f"{name}: {len(df)} images, {df['using_phone'].sum()} using phone "
              f"({df['using_phone'].mean():.1%}), {(df['split'] == 'val').sum()} from val")
