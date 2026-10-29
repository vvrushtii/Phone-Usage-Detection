"""Enhancement functions keep the image format, survive degenerate input and leave "none" alone."""
import numpy as np
import pytest

from ml import enhance

RNG = np.random.default_rng(0)
IMG = RNG.integers(0, 80, size=(60, 90, 3), dtype=np.uint8)  # a dark, noisy picture; not square on purpose


@pytest.mark.parametrize("method", enhance.METHODS)
def test_shape_and_dtype_are_preserved(method):
    out = enhance.enhance(IMG, method)
    assert out.shape == IMG.shape and out.dtype == np.uint8


@pytest.mark.parametrize("method", enhance.METHODS)
def test_all_black_image_does_not_crash(method):
    black = np.zeros((40, 40, 3), dtype=np.uint8)
    assert enhance.enhance(black, method).shape == black.shape


def test_none_returns_the_input_unchanged():
    assert enhance.enhance(IMG, "none") is IMG


def test_enhancement_does_not_modify_the_input():
    before = IMG.copy()
    for method in enhance.METHODS:
        enhance.enhance(IMG, method)
    assert np.array_equal(IMG, before)


@pytest.mark.parametrize("method", ["global_he", "clahe", "clahe_smooth"])
def test_dark_image_gets_brighter(method):
    assert enhance.to_gray_mean(enhance.enhance(IMG, method)) > enhance.to_gray_mean(IMG)


def test_params_are_used():
    weak = enhance.enhance(IMG, "clahe", {"clip_limit": 1.0})
    strong = enhance.enhance(IMG, "clahe", {"clip_limit": 8.0})
    assert not np.array_equal(weak, strong)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        enhance.enhance(IMG, "sharpen")


def test_to_gray_mean_range():
    assert enhance.to_gray_mean(np.zeros((4, 4, 3), np.uint8)) == 0
    assert enhance.to_gray_mean(np.full((4, 4, 3), 255, np.uint8)) == 255
