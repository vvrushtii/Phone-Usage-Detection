"""Build the low-light test set: darkened copies of the FPI-Det test images, plus a real-dark subset.

Run from the repo root:
    python ml/make_lowlight.py

Writes data/lowlight/dim/ and data/lowlight/dark/ (one JPEG per test image) and results/lowlight/real_dark_subset.csv
(the 15% of original test images with the lowest mean brightness, i.e. photos that really were dark).
Darkening is deterministic: the noise for an image depends only on the seed, the condition and the image name, so
the same image always gets the same dark copy, whether it is read from disk or generated in memory (as for val images).
Boxes do not move when an image is darkened, so the original labels still apply.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import enhance  # noqa: E402

IMAGES = ROOT / "data/fpi_yolo/images"
OUT_DIR = ROOT / "data/lowlight"
RESULTS = ROOT / "results/lowlight"
SEED = 42
JPEG_QUALITY = 95
REAL_DARK_FRACTION = 0.15
# brightness factor, gamma, noise sigma (in 0-255 grey levels)
CONDITIONS: dict[str, dict[str, float]] = {
    "dim": {"brightness": 0.45, "gamma": 1.8, "sigma": 4.0},
    "dark": {"brightness": 0.25, "gamma": 2.4, "sigma": 8.0},
}


def darken(img: np.ndarray, condition: str, name: str) -> np.ndarray:
    """A darkened, noisy copy: scale brightness, apply gamma, add Gaussian noise, then save as JPEG would."""
    p = CONDITIONS[condition]
    rng = np.random.default_rng([SEED, zlib.crc32(f"{condition}/{name}".encode())])
    x = (img.astype(np.float32) / 255.0 * p["brightness"]) ** p["gamma"] * 255.0
    x = np.clip(x + rng.normal(0.0, p["sigma"], img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # what a file on disk would give back, so disk and memory agree


def load_condition(path: Path, condition: str) -> np.ndarray:
    """The image at `path` (a data/fpi_yolo/images/<split>/ file) as it looks under `condition` (original, dim, dark)."""
    if condition == "original":
        return cv2.imread(str(path))
    on_disk = OUT_DIR / condition / path.name
    if path.parent.name == "test" and on_disk.exists():
        return cv2.imread(str(on_disk))
    return darken(cv2.imread(str(path)), condition, path.stem)


def test_images() -> list[Path]:
    return sorted((IMAGES / "test").glob("*.jpg"))


def real_dark_subset(paths: list[Path]) -> pd.DataFrame:
    """The darkest REAL_DARK_FRACTION of the given images by mean grey level, darkest first."""
    table = pd.DataFrame({"image_name": [p.name for p in paths],
                          "brightness": [enhance.to_gray_mean(cv2.imread(str(p))) for p in paths]})
    table = table.sort_values(["brightness", "image_name"]).reset_index(drop=True)
    return table.head(round(REAL_DARK_FRACTION * len(table)))


def load_real_dark() -> pd.DataFrame:
    """The subset written by main() (reading every image again to rank brightness is slow)."""
    return pd.read_csv(RESULTS / "real_dark_subset.csv")


def main() -> None:
    paths = test_images()
    for condition in CONDITIONS:
        folder = OUT_DIR / condition
        folder.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(paths, 1):
            cv2.imwrite(str(folder / path.name), darken(cv2.imread(str(path)), condition, path.stem),
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if i % 500 == 0:
                print(f"{condition}: {i}/{len(paths)}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    subset = real_dark_subset(paths)
    subset.to_csv(RESULTS / "real_dark_subset.csv", index=False)
    print(f"wrote {len(paths)} images per condition to {OUT_DIR}; real-dark subset: {len(subset)} images, "
          f"mean brightness up to {subset['brightness'].max():.1f}")


if __name__ == "__main__":
    main()
