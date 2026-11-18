"""Unit tests for Stage 2: Dynamic Range Tone Curve & CIE L*a*b* CLAHE."""
import cv2
import numpy as np
import pytest
from app.pipeline.tone import (
    apply_tone_curve,
    apply_clahe,
    adjust_tone_and_contrast,
    apply_tone_curve_and_clahe,
    build_tone_lut,
)


def test_shadow_lift_increases_underexposed_regions(synthetic_gradient_bgr):
    """Shadow lift must selectively elevate pixels with luminance < 0.20."""
    out = apply_tone_curve_and_clahe(
        synthetic_gradient_bgr,
        shadow_lift=0.30,
        highlight_recovery=0.0,
        clahe_clip_limit=0.0,
    )
    # Check pixels where input was dark (e.g. value 25 / 255 ~= 0.10)
    dark_mask = synthetic_gradient_bgr[:, :, 0] < 50
    assert out[dark_mask].mean() > synthetic_gradient_bgr[dark_mask].mean()


def test_highlight_recovery_compresses_glare(synthetic_gradient_bgr):
    """Highlight recovery must roll off pixels with luminance > 0.70 without clipping."""
    out = apply_tone_curve_and_clahe(
        synthetic_gradient_bgr,
        shadow_lift=0.0,
        highlight_recovery=0.30,
        clahe_clip_limit=0.0,
    )
    bright_mask = synthetic_gradient_bgr[:, :, 0] > 200
    assert out[bright_mask].mean() < synthetic_gradient_bgr[bright_mask].mean()


def test_clahe_expands_contrast():
    """CLAHE on low-contrast synthetic image must expand its histogram variance."""
    low_contrast = np.full((128, 128, 3), 120, dtype=np.uint8)
    low_contrast[32:96, 32:96, :] = 135
    std_before = low_contrast.std()

    out = apply_tone_curve_and_clahe(
        low_contrast,
        shadow_lift=0.0,
        highlight_recovery=0.0,
        clahe_clip_limit=3.0,
    )
    std_after = out.std()
    assert std_after > std_before


def test_clahe_strictly_preserves_chroma():
    """
    CLAHE operating strictly on CIE L* must not introduce color cast
    or significantly alter chromatic channels a* and b*.
    """
    wood_bgr = np.zeros((128, 128, 3), dtype=np.uint8)
    wood_bgr[:, :, 0] = 50   # Blue
    wood_bgr[:, :, 1] = 90   # Green
    wood_bgr[:, :, 2] = 140  # Red

    lab_orig = cv2.cvtColor(wood_bgr, cv2.COLOR_BGR2Lab)
    out = apply_tone_curve_and_clahe(wood_bgr, clahe_clip_limit=2.0)
    lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2Lab)

    np.testing.assert_allclose(lab_orig[:, :, 1], lab_out[:, :, 1], atol=1)
    np.testing.assert_allclose(lab_orig[:, :, 2], lab_out[:, :, 2], atol=1)


def test_neutral_params_preserve_identity(synthetic_flat_bgr):
    """Zero adjustments must leave image identical."""
    out1 = apply_tone_curve(synthetic_flat_bgr, shadow_lift=0.0, highlight_recovery=0.0)
    np.testing.assert_array_equal(out1, synthetic_flat_bgr)

    out2 = apply_clahe(synthetic_flat_bgr, clip_limit=0.0)
    np.testing.assert_array_equal(out2, synthetic_flat_bgr)

    out3 = adjust_tone_and_contrast(synthetic_flat_bgr, shadow_lift=0.0, highlight_recovery=0.0, clahe_clip_limit=0.0)
    np.testing.assert_array_equal(out3, synthetic_flat_bgr)


def test_alias_equivalence(sample_bgr_u8):
    """adjust_tone_and_contrast and apply_tone_curve_and_clahe must produce identical results."""
    res1 = adjust_tone_and_contrast(sample_bgr_u8, shadow_lift=0.2, highlight_recovery=0.1, clahe_clip_limit=1.5)
    res2 = apply_tone_curve_and_clahe(sample_bgr_u8, shadow_lift=0.2, highlight_recovery=0.1, clahe_clip_limit=1.5)
    np.testing.assert_array_equal(res1, res2)


def test_float32_support(sample_bgr_f32):
    """Tone module must support float32 images."""
    out = apply_tone_curve_and_clahe(sample_bgr_f32, shadow_lift=0.2, highlight_recovery=0.1, clahe_clip_limit=2.0)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert not np.isnan(out).any()


def test_alpha_channel_preservation(sample_bgra_u8):
    """Alpha channel must be preserved in 4-channel BGRA images."""
    out = apply_tone_curve_and_clahe(sample_bgra_u8, shadow_lift=0.3, clahe_clip_limit=2.0)
    assert out.shape == sample_bgra_u8.shape
    np.testing.assert_array_equal(out[:, :, 3], sample_bgra_u8[:, :, 3])


def test_invalid_input_raises_value_error():
    """Invalid input images must raise ValueError."""
    with pytest.raises(ValueError):
        apply_tone_curve(None)
    with pytest.raises(ValueError):
        apply_clahe(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        adjust_tone_and_contrast(np.zeros((64, 64), dtype=np.uint8))


def test_tone_lut_strict_monotonicity_under_extreme_parameters():
    """Tone LUT must be strictly non-decreasing even under extreme parameters (e.g. shadow_lift > 0.40)."""
    for sl in [0.0, 0.20, 0.40, 0.45, 0.60, 0.80, 1.0]:
        for hr in [0.0, 0.12, 0.30, 0.50, 0.80, 1.0]:
            lut = build_tone_lut(shadow_lift=sl, highlight_recovery=hr)
            diffs = np.diff(lut.astype(np.int32))
            assert np.all(diffs >= 0), f"Monotonicity violated for shadow_lift={sl}, highlight_recovery={hr}"
