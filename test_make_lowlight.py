"""Darkening is deterministic, darker, and keeps the image size."""
import numpy as np

from ml import enhance, make_lowlight as ml

IMG = np.random.default_rng(1).integers(60, 220, size=(50, 70, 3), dtype=np.uint8)


def test_same_image_same_condition_same_result():
    assert np.array_equal(ml.darken(IMG, "dark", "a.jpg"), ml.darken(IMG, "dark", "a.jpg"))


def test_noise_depends_on_the_image_name():
    assert not np.array_equal(ml.darken(IMG, "dark", "a.jpg"), ml.darken(IMG, "dark", "b.jpg"))


def test_dark_is_darker_than_dim_is_darker_than_original():
    dim, dark = ml.darken(IMG, "dim", "a.jpg"), ml.darken(IMG, "dark", "a.jpg")
    assert enhance.to_gray_mean(dark) < enhance.to_gray_mean(dim) < enhance.to_gray_mean(IMG)


def test_shape_and_dtype_kept():
    out = ml.darken(IMG, "dim", "a.jpg")
    assert out.shape == IMG.shape and out.dtype == np.uint8
