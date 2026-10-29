"""Build the clean YOLO dataset from the raw FPI-Det download.

Run from the repo root:
    python ml/prepare_dataset.py

Reads  data/raw/reorganized_dataset/{train,val,test}/{images,labels}
       third_party/FPI-Det/classification_task/label.csv
Writes data/fpi_yolo/{images,labels}/{train,val,test}, fpi.yaml, usage_labels.csv
       data/fpi_yolo.zip                (upload this to Colab)
       results/dataset_stats.json, results/splits.json, results/label_check/*.jpg

The raw splits are kept as they are. label.csv covers exactly the val and test images, so
the detector never trains on an image used for usage evaluation: val is for tuning, test is
for the final numbers. Safe to re-run.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import statistics
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/reorganized_dataset"
OUT = ROOT / "data/fpi_yolo"
LABEL_CSV = ROOT / "third_party/FPI-Det/classification_task/label.csv"
RESULTS = ROOT / "results"

SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "phone", 1: "head"}  # ids as shipped in the dataset
# label.csv: 1 = using phone. Checked against the boxes in check_polarity(), because the
# repo's own script docstring claims the opposite.
USING_PHONE_LABEL = 1
# Boxes thinner than this (as a fraction of the image) are annotation noise, not objects.
MIN_BOX_SIDE = 0.002
# Edges that overshoot the image by less than this are float noise from the annotation
# tool's conversions; they are still clipped but not counted as defects.
CLIP_EPS = 1e-4
CHECK_IMAGES_PER_SPLIT = 10
CHECK_SEED = 42
CHECK_TILE_H = 256
PHONE_COLOR = (0, 0, 255)  # BGR
HEAD_COLOR = (255, 128, 0)


@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def area(self) -> float:
        return self.w * self.h

    def line(self) -> str:
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class ParseLog:
    joined_lines: int = 0
    clipped: int = 0
    dropped: int = 0
    files_with_fix: int = 0
    notes: list[str] = field(default_factory=list)


def parse_label(path: Path, log: ParseLog) -> list[Box]:
    """Read one label file. Some files put several boxes on one line, so split on tokens."""
    text = path.read_text()
    tokens = text.split()
    if len(tokens) % 5:
        raise ValueError(f"{path}: {len(tokens)} tokens is not a multiple of 5")
    if any(len(line.split()) != 5 for line in text.splitlines() if line.strip()):
        log.joined_lines += 1
    boxes: list[Box] = []
    changed = False
    for i in range(0, len(tokens), 5):
        cls = int(tokens[i])
        if cls not in CLASS_NAMES:
            raise ValueError(f"{path}: unknown class {cls}")
        cx, cy, w, h = map(float, tokens[i + 1 : i + 5])
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        cx1, cy1, cx2, cy2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
        if max(abs(a - b) for a, b in zip((cx1, cy1, cx2, cy2), (x1, y1, x2, y2))) > CLIP_EPS:
            log.clipped += 1
            changed = True
        nw, nh = cx2 - cx1, cy2 - cy1
        if nw < MIN_BOX_SIDE or nh < MIN_BOX_SIDE:
            log.dropped += 1
            changed = True
            log.notes.append(f"dropped {path.parent.parent.name}/{path.name}: {tokens[i:i + 5]}")
            continue
        boxes.append(Box(cls, (cx1 + cx2) / 2, (cy1 + cy2) / 2, nw, nh))
    log.files_with_fix += changed
    return boxes


def copy_image(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def build_split(split: str, log: ParseLog) -> dict[str, list[Box]]:
    img_dir, lbl_dir = RAW / split / "images", RAW / split / "labels"
    out_img, out_lbl = OUT / "images" / split, OUT / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    boxes_by_stem: dict[str, list[Box]] = {}
    for img in sorted(img_dir.iterdir()):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"no label for {img}")
        boxes = parse_label(lbl, log)
        boxes_by_stem[img.stem] = boxes
        copy_image(img, out_img / img.name)
        (out_lbl / lbl.name).write_text("".join(b.line() + "\n" for b in boxes))
    return boxes_by_stem


def load_usage_labels() -> dict[str, int]:
    with LABEL_CSV.open() as f:
        return {Path(r["image_name"]).stem: int(float(r["class_id"])) for r in csv.DictReader(f)}


def check_polarity(boxes: dict[str, dict[str, list[Box]]], labels: dict[str, int]) -> dict[str, float]:
    """Fraction of images with a phone box, per label.csv class. Fails if 1 is not 'using phone'."""
    rate: dict[str, float] = {}
    for lab in (0, 1):
        images = [bs for sp in ("val", "test") for s, bs in boxes[sp].items() if labels.get(s) == lab]
        rate[str(lab)] = sum(any(b.cls == 0 for b in bs) for bs in images) / len(images)
    if rate[str(USING_PHONE_LABEL)] <= rate[str(1 - USING_PHONE_LABEL)]:
        raise RuntimeError(f"label.csv polarity looks wrong, phone-box rates per label: {rate}")
    return rate


def write_usage_labels(boxes: dict[str, dict[str, list[Box]]], labels: dict[str, int]) -> dict[str, dict]:
    """usage_labels.csv: one row per image in label.csv, with a clear using_phone column."""
    missing = [s for s in labels if not any(s in boxes[sp] for sp in ("val", "test"))]
    if missing:
        raise RuntimeError(f"{len(missing)} label.csv images are not in val/test, e.g. {missing[:3]}")
    summary: dict[str, dict] = {}
    with (OUT / "usage_labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "split", "using_phone"])
        for sp in ("val", "test"):
            table: Counter = Counter()
            for stem in sorted(boxes[sp]):
                if stem not in labels:
                    raise RuntimeError(f"{sp}/{stem} has no row in label.csv")
                using = int(labels[stem] == USING_PHONE_LABEL)
                w.writerow([f"{stem}.jpg", sp, using])
                cls = {b.cls for b in boxes[sp][stem]}
                table[f"using={using} phone_box={0 in cls} head_box={1 in cls}"] += 1
            summary[sp] = dict(sorted(table.items()))
    return summary


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    q = statistics.quantiles(values, n=20)
    return {"min": min(values), "p05": q[0], "median": statistics.median(values), "p95": q[-1], "max": max(values)}


def build_stats(boxes: dict[str, dict[str, list[Box]]]) -> dict:
    stats: dict = {}
    for sp in SPLITS:
        per_cls: Counter = Counter()
        areas: dict[str, list[float]] = {n: [] for n in CLASS_NAMES.values()}
        widths, heights = [], []
        for stem, bs in boxes[sp].items():
            for b in bs:
                per_cls[CLASS_NAMES[b.cls]] += 1
                areas[CLASS_NAMES[b.cls]].append(b.area())
            with Image.open(OUT / "images" / sp / f"{stem}.jpg") as im:
                widths.append(im.width)
                heights.append(im.height)
        bs_all = boxes[sp].values()
        stats[sp] = {
            "images": len(boxes[sp]),
            "boxes": dict(per_cls),
            "images_without_boxes": sum(not b for b in bs_all),
            "images_with_phone": sum(any(x.cls == 0 for x in b) for b in bs_all),
            "images_with_head": sum(any(x.cls == 1 for x in b) for b in bs_all),
            "images_with_both": sum({0, 1} <= {x.cls for x in b} for b in bs_all),
            "boxes_per_image_max": max(len(b) for b in bs_all),
            "relative_box_area": {n: quantiles(a) for n, a in areas.items()},
            "image_width_px": quantiles([float(v) for v in widths]),
            "image_height_px": quantiles([float(v) for v in heights]),
        }
    return stats


def make_label_check(boxes: dict[str, dict[str, list[Box]]]) -> None:
    """One contact sheet per split: random images with their boxes drawn, for checking by eye."""
    out = RESULTS / "label_check"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(CHECK_SEED)
    for sp in SPLITS:
        picks = rng.sample(sorted(boxes[sp]), CHECK_IMAGES_PER_SPLIT)
        tiles = []
        for stem in picks:
            img = cv2.imread(str(OUT / "images" / sp / f"{stem}.jpg"))
            h, w = img.shape[:2]
            for b in boxes[sp][stem]:
                p1 = (int((b.cx - b.w / 2) * w), int((b.cy - b.h / 2) * h))
                p2 = (int((b.cx + b.w / 2) * w), int((b.cy + b.h / 2) * h))
                cv2.rectangle(img, p1, p2, PHONE_COLOR if b.cls == 0 else HEAD_COLOR, 2)
            scale = CHECK_TILE_H / h
            tiles.append(cv2.resize(img, (max(1, int(w * scale)), CHECK_TILE_H)))
        cv2.imwrite(str(out / f"{sp}.jpg"), cv2.hconcat(tiles))


def write_yaml() -> None:
    # No `path:` key on purpose: Ultralytics then uses this file's folder, so the same
    # dataset works unchanged on Colab.
    (OUT / "fpi.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in CLASS_NAMES.items())
    )


def make_zip() -> Path:
    zip_path = OUT.parent / "fpi_yolo.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # jpgs are already compressed
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.suffix != ".cache":  # Ultralytics label caches are machine-specific
                z.write(p, Path("fpi_yolo") / p.relative_to(OUT))
    return zip_path


def main() -> None:
    log = ParseLog()
    boxes = {sp: build_split(sp, log) for sp in SPLITS}
    labels = load_usage_labels()
    polarity = check_polarity(boxes, labels)
    usage = write_usage_labels(boxes, labels)
    write_yaml()

    RESULTS.mkdir(exist_ok=True)
    stats = build_stats(boxes)
    stats["label_cleaning"] = {
        "files_with_joined_boxes_on_one_line": log.joined_lines,
        "boxes_clipped_to_image": log.clipped,
        "boxes_dropped_degenerate": log.dropped,
        "files_changed": log.files_with_fix,
        "dropped_details": log.notes,
    }
    stats["usage_labels"] = {
        "using_phone_label_value": USING_PHONE_LABEL,
        "phone_box_rate_per_label_csv_class": polarity,
        "by_split": usage,
    }
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (RESULTS / "splits.json").write_text(json.dumps({
        "source": "raw splits kept as shipped",
        "det_train": "train", "det_val": "val", "det_test": "test",
        "usage_val": "val (label.csv rows)", "usage_test": "test (label.csv rows)",
        "images": {sp: len(boxes[sp]) for sp in SPLITS},
    }, indent=2))
    make_label_check(boxes)
    zip_path = make_zip()
    print(json.dumps({sp: stats[sp]["images"] for sp in SPLITS}), stats["label_cleaning"] | {"dropped_details": len(log.notes)})
    print(f"wrote {OUT}, {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB), results/dataset_stats.json")


if __name__ == "__main__":
    main()
