"""Unit tests for Stage 3: Multi-Scale Screen-Space Ambient Occlusion."""
import statistics
import time
import warnings
import numpy as np
import pytest
from app.pipeline.ao import apply_ambient_occlusion


def test_ao_darkens_crevices(synthetic_crevice_bgr):
    """Narrow joints and crevices must be darkened by the Black Top-Hat filter."""
    out = apply_ambient_occlusion(synthetic_crevice_bgr, strength=0.25)
    crevice_orig = synthetic_crevice_bgr[:, 63:65, :].mean()
    crevice_out = out[:, 63:65, :].mean()
    assert crevice_out < crevice_orig


def test_ao_uniform_surface_invariance(synthetic_flat_bgr):
    """Uniform surfaces have zero Black Top-Hat response and must remain unaltered."""
    out = apply_ambient_occlusion(synthetic_flat_bgr, strength=0.25)
    np.testing.assert_array_equal(out, synthetic_flat_bgr)


def test_ao_strength_scaling(synthetic_crevice_bgr):
    """Higher strength parameter must darken crevices more aggressively."""
    out_weak = apply_ambient_occlusion(synthetic_crevice_bgr, strength=0.10)
    out_strong = apply_ambient_occlusion(synthetic_crevice_bgr, strength=0.30)
    assert out_strong[:, 63:65, :].mean() < out_weak[:, 63:65, :].mean()


def test_ao_zero_strength_identity(synthetic_crevice_bgr):
    """Strength = 0.0 must leave crevice image identical to input."""
    out = apply_ambient_occlusion(synthetic_crevice_bgr, strength=0.0)
    np.testing.assert_array_equal(out, synthetic_crevice_bgr)


def test_ao_float32_support(sample_bgr_f32):
    """AO module must support float32 images."""
    out = apply_ambient_occlusion(sample_bgr_f32, strength=0.2)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert not np.isnan(out).any()


def test_ao_alpha_channel_preservation(sample_bgra_u8):
    """Alpha channel must be preserved in 4-channel BGRA images."""
    out = apply_ambient_occlusion(sample_bgra_u8, strength=0.2)
    assert out.shape == sample_bgra_u8.shape
    np.testing.assert_array_equal(out[:, :, 3], sample_bgra_u8[:, :, 3])


def test_ao_invalid_inputs():
    """Invalid input images must raise ValueError."""
    with pytest.raises(ValueError):
        apply_ambient_occlusion(None)
    with pytest.raises(ValueError):
        apply_ambient_occlusion(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        apply_ambient_occlusion(np.zeros((64, 64), dtype=np.uint8))


def test_ao_multi_scale_darkens_medium_joints_and_wide_corners():
    """
    Multi-scale SSAO must darken the interior of medium joints (radius=11)
    and wide architectural corners/recesses (radius=35 Gaussian pyramid),
    where a fine radius=3 (7x7) kernel alone cannot reach.
    """
    # Medium joint: 16px-wide recessed band (cols 120..136) in a 256x256 panel.
    # Center cols 126..130 are >= 6px from either edge, beyond radius=3 (3px reach).
    joint_img = np.full((256, 256, 3), 200, dtype=np.uint8)
    joint_img[:, 120:136, :] = 100
    out_joint = apply_ambient_occlusion(joint_img, strength=0.35, radius=3)
    assert out_joint[:, 126:130, :].mean() < joint_img[:, 126:130, :].mean()

    # Wide corner/niche recess: 36px-wide recessed wall niche (cols 110..146).
    # Center cols 125..131 are >= 15px from edges, reached by radius=35 depth shading.
    corner_img = np.full((256, 256, 3), 210, dtype=np.uint8)
    corner_img[:, 110:146, :] = 110
    out_corner = apply_ambient_occlusion(corner_img, strength=0.40, radius=3)
    assert out_corner[:, 125:131, :].mean() < corner_img[:, 125:131, :].mean()


def test_ao_odd_and_tiny_dimensions(non_power_of_two_bgr):
    """Odd, sliver, and tiny dimensions must process cleanly without shape errors or NaNs."""
    shapes = [
        non_power_of_two_bgr,
        np.full((16, 16, 3), 150, dtype=np.uint8),
        np.random.default_rng(7).integers(30, 220, (5, 5, 3), dtype=np.uint8),
        np.random.default_rng(8).integers(30, 220, (2, 2, 3), dtype=np.uint8),
        np.random.default_rng(9).integers(30, 220, (64, 1920, 3), dtype=np.uint8),
        np.random.default_rng(10).integers(30, 220, (1920, 64, 3), dtype=np.uint8),
    ]
    for arr in shapes:
        out = apply_ambient_occlusion(arr, strength=0.25)
        assert out.shape == arr.shape
        assert out.dtype == arr.dtype
        assert not np.isnan(out).any()


def test_ao_latency_960px_proxy():
    """Multi-scale SSAO on a 540x960 preview proxy must execute well within 30ms min / 50ms median."""
    rng = np.random.default_rng(42)
    proxy = rng.integers(30, 225, (540, 960, 3), dtype=np.uint8)
    for _ in range(2):
        _ = apply_ambient_occlusion(proxy, strength=0.15)
    times = []
    out = None
    for _ in range(7):
        t0 = time.perf_counter()
        out = apply_ambient_occlusion(proxy, strength=0.15)
        times.append(time.perf_counter() - t0)
    assert out.shape == proxy.shape
    assert min(times) < 0.030, f"AO 960px proxy min={min(times) * 1000:.2f}ms (expected < 30ms)"
    assert statistics.median(times) < 0.050, (
        f"AO 960px proxy median={statistics.median(times) * 1000:.2f}ms (expected < 50ms)"
    )


@pytest.mark.parametrize("bad_dtype", [np.float64, np.int32, np.uint16, np.int16, np.bool_])
@pytest.mark.parametrize("strength", [0.0, 0.15])
def test_ao_rejects_unsupported_dtypes(bad_dtype, strength):
    """Unsupported image dtypes must raise ValueError consistently regardless of strength."""
    arr = np.ones((16, 16, 3), dtype=bad_dtype)
    with pytest.raises(ValueError, match="Unsupported image dtype"):
        apply_ambient_occlusion(arr, strength=strength)


@pytest.mark.parametrize("bad_strength", [float("nan"), float("inf"), float("-inf")])
def test_ao_rejects_non_finite_strength(sample_bgr_u8, bad_strength):
    """Non-finite strength values must raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        apply_ambient_occlusion(sample_bgr_u8, strength=bad_strength)


def test_ao_float32_nan_and_oor_sanitized_without_warnings():
    """float32 NaN, Inf, and out-of-range inputs must clamp cleanly to [0.0, 1.0] without warnings."""
    uniform_neg = np.full((32, 32, 3), -0.5, dtype=np.float32)
    uniform_high = np.full((32, 32, 3), 1.5, dtype=np.float32)
    mixed = np.full((32, 32, 4), 0.6, dtype=np.float32)
    mixed[10:20, 10:20, :3] = 0.2
    mixed[0, 0, 0] = np.nan
    mixed[0, 1, 1] = np.inf
    mixed[0, 2, 2] = -np.inf
    mixed[1, 0, 0] = -0.75
    mixed[1, 1, 1] = 1.75
    mixed[:, :, 3] = 0.75

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        out_neg = apply_ambient_occlusion(uniform_neg, strength=0.15)
        out_high = apply_ambient_occlusion(uniform_high, strength=0.15)
        out_mixed = apply_ambient_occlusion(mixed, strength=0.25)

    assert len(caught) == 0, f"Unexpected RuntimeWarnings: {[str(w.message) for w in caught]}"
    assert np.all(out_neg == 0.0)
    assert np.all(out_high == 1.0)
    assert np.all(np.isfinite(out_mixed[:, :, :3]))
    assert out_mixed[:, :, :3].min() >= 0.0
    assert out_mixed[:, :, :3].max() <= 1.0
    np.testing.assert_array_equal(out_mixed[:, :, 3], mixed[:, :, 3])


def test_ao_default_strength_darkens_wide_36_40px_recess():
    """Scale 3 Gaussian pyramid shading must measurably darken 36-40px wide recesses at default strength=0.15."""
    for width in (36, 40):
        panel = np.full((256, 256, 3), 200, dtype=np.uint8)
        start = 128 - width // 2
        panel[:, start : start + width, :] = 100
        out = apply_ambient_occlusion(panel, strength=0.15)
        center_delta = 100.0 - float(out[:, 126:130, :].mean())
        assert center_delta >= 1.5, f"{width}px recess center delta={center_delta:.2f} too weak at strength=0.15"
