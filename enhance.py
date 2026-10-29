"""Image enhancement for dark webcam frames (syllabus Unit II: histogram equalization and filtering).

Pure functions on 8-bit BGR images (OpenCV's order), reused by the experiment (ml/lowlight_experiment.py) and the app.
Methods: "none" | "global_he" | "clahe" | "clahe_smooth".
"""
from __future__ import annotations

import cv2
import numpy as np

METHODS = ("none", "global_he", "clahe", "clahe_smooth")
DEFAULT_PARAMS = {"clip_limit": 3.0, "tile_grid": (8, 8), "smooth_ksize": 3}


def to_gray_mean(img: np.ndarray) -> float:
    """Mean brightness of the grayscale version of the image, 0 (black) to 255 (white)."""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def global_he(img: np.ndarray) -> np.ndarray:
    """Histogram equalization on the luminance channel only, so colours are not shifted."""
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    return cv2.cvtColor(cv2.merge([cv2.equalizeHist(y), cr, cb]), cv2.COLOR_YCrCb2BGR)


def clahe(img: np.ndarray, clip_limit: float = 3.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on the L channel of LAB."""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    equalizer = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid[0]), int(tile_grid[1])))
    return cv2.cvtColor(cv2.merge([equalizer.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def smooth(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Gaussian blur against noise (sigma is derived from the kernel size). ksize must be odd."""
    return cv2.GaussianBlur(img, (int(ksize), int(ksize)), 0)


def enhance(img: np.ndarray, method: str, params: dict | None = None) -> np.ndarray:
    """Apply one of METHODS. Missing params fall back to DEFAULT_PARAMS. "none" returns the input itself."""
    if method not in METHODS:
        raise ValueError(f"unknown enhancement method {method!r}, expected one of {METHODS}")
    p = {**DEFAULT_PARAMS, **(params or {})}
    if method == "none":
        return img
    if method == "global_he":
        return global_he(img)
    out = clahe(img, p["clip_limit"], p["tile_grid"])
    return smooth(out, p["smooth_ksize"]) if method == "clahe_smooth" else out
