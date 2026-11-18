"""Unit tests for Stage 4: Micro-Contrast (Unsharp Mask) & Bilateral Denoise."""
import cv2
import numpy as np
import pytest
from app.pipeline.detail import apply_detail_enhancement


def test_unsharp_mask_boosts_edge_sharpness(synthetic_checkerboard_bgr):
    """Unsharp masking must boost Laplacian gradient variance across sharp edges."""
    lap_orig = cv2.Laplacian(synthetic_checkerboard_bgr, cv2.CV_64F).var()
    out = apply_detail_enhancement(synthetic_checkerboard_bgr, sharpness=0.60, denoise_strength=0.0)
    lap_out = cv2.Laplacian(out, cv2.CV_64F).var()
    assert lap_out > lap_orig


def test_sharpness_zero_identity(synthetic_checkerboard_bgr):
    """Sharpness = 0.0 with denoise = 0.0 must preserve input pixels."""
    out = apply_detail_enhancement(synthetic_checkerboard_bgr, sharpness=0.0, denoise_strength=0.0)
    np.testing.assert_array_equal(out, synthetic_checkerboard_bgr)


def test_bilateral_denoise_smoothes_uniform_noise():
    """Bilateral denoise must reduce variance on noisy uniform patches."""
    np.random.seed(42)
    noise = np.random.normal(0, 8, (128, 128, 3)).astype(np.float32)
    patch = np.clip(128.0 + noise, 0, 255).astype(np.uint8)
    std_noisy = patch.std()

    out = apply_detail_enhancement(patch, sharpness=0.0, denoise_strength=0.30)
    std_clean = out.std()
    assert std_clean < std_noisy


def test_output_bounds_preserved(synthetic_gradient_bgr):
    """Detail enhancement must never overflow uint8 bounds [0, 255]."""
    out = apply_detail_enhancement(synthetic_gradient_bgr, sharpness=1.5, denoise_strength=0.20)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_detail_float32_support(sample_bgr_f32):
    """Detail enhancement must support float32 images."""
    out = apply_detail_enhancement(sample_bgr_f32, sharpness=0.4, denoise_strength=0.1)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert not np.isnan(out).any()


def test_detail_alpha_preservation(sample_bgra_u8):
    """Alpha channel must be preserved in 4-channel BGRA images."""
    out = apply_detail_enhancement(sample_bgra_u8, sharpness=0.4, denoise_strength=0.1)
    assert out.shape == sample_bgra_u8.shape
    np.testing.assert_array_equal(out[:, :, 3], sample_bgra_u8[:, :, 3])


def test_detail_invalid_inputs():
    """Invalid input images must raise ValueError."""
    with pytest.raises(ValueError):
        apply_detail_enhancement(None)
    with pytest.raises(ValueError):
        apply_detail_enhancement(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        apply_detail_enhancement(np.zeros((64, 64), dtype=np.uint8))
