"""Unit tests for Stage 1: Color Balance, Temperature, Tint & Exposure."""
import numpy as np
import pytest
from app.pipeline.color import apply_color_correction, apply_white_balance


def test_neutral_params_preserve_identity(synthetic_flat_bgr):
    """Default neutral settings (temp=0, tint=0, exp=0) must leave image unchanged."""
    out = apply_color_correction(synthetic_flat_bgr, temperature=0.0, tint=0.0, exposure=0.0)
    np.testing.assert_array_equal(out, synthetic_flat_bgr)


def test_warm_temperature_shifts_red_and_decreases_blue(synthetic_flat_bgr):
    """Positive temperature must increase Red and decrease Blue channel means."""
    out = apply_color_correction(synthetic_flat_bgr, temperature=0.5, tint=0.0, exposure=0.0)
    # OpenCV BGR: index 0 is Blue, index 2 is Red
    assert out[:, :, 2].mean() > synthetic_flat_bgr[:, :, 2].mean()
    assert out[:, :, 0].mean() < synthetic_flat_bgr[:, :, 0].mean()


def test_cool_temperature_shifts_blue_and_decreases_red(synthetic_flat_bgr):
    """Negative temperature must increase Blue and decrease Red channel means."""
    out = apply_color_correction(synthetic_flat_bgr, temperature=-0.5, tint=0.0, exposure=0.0)
    assert out[:, :, 0].mean() > synthetic_flat_bgr[:, :, 0].mean()
    assert out[:, :, 2].mean() < synthetic_flat_bgr[:, :, 2].mean()


def test_magenta_tint_shifts_red_blue_over_green(synthetic_flat_bgr):
    """Positive tint (magenta) must suppress Green relative to Red and Blue."""
    out = apply_color_correction(synthetic_flat_bgr, temperature=0.0, tint=0.5, exposure=0.0)
    assert out[:, :, 1].mean() < synthetic_flat_bgr[:, :, 1].mean()


def test_exposure_positive_brightens(synthetic_flat_bgr):
    """Positive exposure compensation must brighten the entire image."""
    out = apply_color_correction(synthetic_flat_bgr, temperature=0.0, tint=0.0, exposure=1.0)
    assert out.mean() > synthetic_flat_bgr.mean()


def test_exposure_negative_darkens(synthetic_flat_bgr):
    """Negative exposure compensation must darken the image."""
    out = apply_color_correction(synthetic_flat_bgr, temperature=0.0, tint=0.0, exposure=-1.0)
    assert out.mean() < synthetic_flat_bgr.mean()


def test_output_clipping_bounds(synthetic_gradient_bgr):
    """Extreme parameter values must not overflow uint8 bounds [0, 255]."""
    out = apply_color_correction(synthetic_gradient_bgr, temperature=1.0, tint=1.0, exposure=2.0)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_alias_equivalence(sample_bgr_u8):
    """apply_white_balance and apply_color_correction must yield identical results."""
    res1 = apply_white_balance(sample_bgr_u8, temperature=0.3, tint=-0.2, exposure=0.5)
    res2 = apply_color_correction(sample_bgr_u8, temperature=0.3, tint=-0.2, exposure=0.5)
    np.testing.assert_array_equal(res1, res2)


def test_float32_support(sample_bgr_f32):
    """Color correction must support float32 [0.0, 1.0] arrays."""
    out = apply_white_balance(sample_bgr_f32, temperature=0.4, tint=0.1, exposure=0.2)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert not np.isnan(out).any()


def test_alpha_channel_preservation(sample_bgra_u8):
    """4-channel BGRA images must retain their exact alpha channel."""
    out = apply_color_correction(sample_bgra_u8, temperature=0.5, tint=-0.3, exposure=0.4)
    assert out.shape == sample_bgra_u8.shape
    np.testing.assert_array_equal(out[:, :, 3], sample_bgra_u8[:, :, 3])


def test_invalid_input_raises_value_error():
    """Invalid image inputs must raise ValueError."""
    with pytest.raises(ValueError):
        apply_color_correction(None)
    with pytest.raises(ValueError):
        apply_color_correction(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        apply_color_correction(np.zeros((100, 100), dtype=np.uint8))  # 2D grayscale
