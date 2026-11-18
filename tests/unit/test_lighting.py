"""Unit tests for Stage 5: Photometric PBR Lighting (Soft GI Bloom & Specular)."""
import statistics
import time
import warnings
import numpy as np
import pytest
from app.pipeline.lighting import apply_pbr_lighting


def test_lighting_zero_strength_and_negative_identity(sample_bgr_u8):
    """Zero or negative strength must return an exact copy of the input image."""
    out_zero = apply_pbr_lighting(sample_bgr_u8, strength=0.0)
    out_neg = apply_pbr_lighting(sample_bgr_u8, strength=-0.5)
    np.testing.assert_array_equal(out_zero, sample_bgr_u8)
    np.testing.assert_array_equal(out_neg, sample_bgr_u8)
    assert out_zero is not sample_bgr_u8


def test_lighting_soft_gi_bloom_spreads_bright_light():
    """
    A bright emitter patch must cast soft indirect GI bloom onto adjacent mid/dark
    surfaces while leaving distant dark corners unaffected.
    """
    img = np.full((256, 256, 3), 40, dtype=np.uint8)
    img[104:152, 104:152, :] = 245

    out = apply_pbr_lighting(img, strength=0.45)

    # Ring immediately adjacent to the bright emitter gains positive GI bounce
    orig_ring = img[92:102, 104:152, :].mean()
    out_ring = out[92:102, 104:152, :].mean()
    assert out_ring > orig_ring

    # Far dark corner stays unchanged at 40
    np.testing.assert_array_equal(out[0:12, 0:12, :], img[0:12, 0:12, :])


def test_lighting_selective_specular_enhancement():
    """
    Selective specular must boost localized bright micro-facets without lifting pure
    dark regions or causing uint8 wraparound.
    """
    img = np.full((128, 128, 3), 130, dtype=np.uint8)
    img[0:32, 0:32, :] = 20  # Shadow corner (below spec gate threshold)
    img[64:96, 64, :] = 195  # Localized specular highlight streak
    img[64:96, 80, :] = 255  # Peak white highlight (must not wrap around)

    out = apply_pbr_lighting(img, strength=0.50)

    assert out[64:96, 64, :].mean() > img[64:96, 64, :].mean()
    assert (out[64:96, 80, :] == 255).all()
    np.testing.assert_array_equal(out[4:28, 4:28, :], img[4:28, 4:28, :])


def test_lighting_strength_scaling(sample_bgr_u8):
    """Higher lighting strength must yield a strictly stronger positive boost."""
    out_weak = apply_pbr_lighting(sample_bgr_u8, strength=0.15)
    out_strong = apply_pbr_lighting(sample_bgr_u8, strength=0.60)
    delta_weak = (out_weak.astype(np.float32) - sample_bgr_u8.astype(np.float32)).mean()
    delta_strong = (out_strong.astype(np.float32) - sample_bgr_u8.astype(np.float32)).mean()
    assert delta_strong > delta_weak > 0.0


def test_lighting_float32_support(sample_bgr_f32):
    """Lighting module must preserve float32 dtype and [0.0, 1.0] range without NaNs."""
    out = apply_pbr_lighting(sample_bgr_f32, strength=0.25)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert not np.isnan(out).any()


def test_lighting_alpha_channel_preservation(sample_bgra_u8):
    """Alpha channel must remain byte-for-byte identical on 4-channel BGRA images."""
    out = apply_pbr_lighting(sample_bgra_u8, strength=0.35)
    assert out.shape == sample_bgra_u8.shape
    np.testing.assert_array_equal(out[:, :, 3], sample_bgra_u8[:, :, 3])


def test_lighting_odd_and_tiny_dimensions(non_power_of_two_bgr):
    """Odd, sliver, and tiny dimensions must execute safely without shape errors."""
    shapes = [
        non_power_of_two_bgr,
        np.random.default_rng(1).integers(30, 220, (16, 16, 3), dtype=np.uint8),
        np.random.default_rng(2).integers(30, 220, (5, 5, 3), dtype=np.uint8),
        np.random.default_rng(3).integers(30, 220, (2, 2, 3), dtype=np.uint8),
        np.random.default_rng(4).integers(30, 220, (64, 1920, 3), dtype=np.uint8),
        np.random.default_rng(5).integers(30, 220, (1920, 64, 3), dtype=np.uint8),
    ]
    for arr in shapes:
        out = apply_pbr_lighting(arr, strength=0.25)
        assert out.shape == arr.shape
        assert out.dtype == arr.dtype
        assert not np.isnan(out).any()


def test_lighting_invalid_inputs():
    """Invalid inputs must raise ValueError."""
    with pytest.raises(ValueError):
        apply_pbr_lighting(None)
    with pytest.raises(ValueError):
        apply_pbr_lighting(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        apply_pbr_lighting(np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError):
        apply_pbr_lighting(np.zeros((32, 32, 1), dtype=np.uint8))


def test_lighting_latency_960px_proxy():
    """Soft GI Bloom + Specular on a 540x960 preview proxy must execute within 30ms min / 50ms median."""
    rng = np.random.default_rng(42)
    proxy = rng.integers(30, 225, (540, 960, 3), dtype=np.uint8)
    for _ in range(2):
        _ = apply_pbr_lighting(proxy, strength=0.25)
    times = []
    out = None
    for _ in range(7):
        t0 = time.perf_counter()
        out = apply_pbr_lighting(proxy, strength=0.25)
        times.append(time.perf_counter() - t0)
    assert out.shape == proxy.shape
    assert min(times) < 0.030, f"Lighting 960px proxy min={min(times) * 1000:.2f}ms (expected < 30ms)"
    assert statistics.median(times) < 0.050, (
        f"Lighting 960px proxy median={statistics.median(times) * 1000:.2f}ms (expected < 50ms)"
    )


@pytest.mark.parametrize("bad_dtype", [np.float64, np.int32, np.uint16, np.int16, np.bool_])
@pytest.mark.parametrize("strength", [0.0, 0.25])
def test_lighting_rejects_unsupported_dtypes(bad_dtype, strength):
    """Unsupported image dtypes must raise ValueError consistently regardless of strength."""
    arr = np.ones((16, 16, 3), dtype=bad_dtype)
    with pytest.raises(ValueError, match="Unsupported image dtype"):
        apply_pbr_lighting(arr, strength=strength)


@pytest.mark.parametrize("bad_strength", [float("nan"), float("inf"), float("-inf")])
def test_lighting_rejects_non_finite_strength(sample_bgr_u8, bad_strength):
    """Non-finite strength values must raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        apply_pbr_lighting(sample_bgr_u8, strength=bad_strength)


def test_lighting_float32_nan_and_oor_sanitized_without_warnings():
    """float32 NaN, Inf, and out-of-range inputs must produce finite [0.0, 1.0] outputs without warnings."""
    uniform_neg = np.full((32, 32, 3), -0.5, dtype=np.float32)
    uniform_high = np.full((32, 32, 3), 1.5, dtype=np.float32)
    mixed = np.full((32, 32, 4), 0.6, dtype=np.float32)
    mixed[10:20, 10:20, :3] = 0.95
    mixed[0, 0, 0] = np.nan
    mixed[0, 1, 1] = np.inf
    mixed[0, 2, 2] = -np.inf
    mixed[1, 0, 0] = -0.75
    mixed[1, 1, 1] = 1.75
    mixed[:, :, 3] = 0.75

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        out_neg = apply_pbr_lighting(uniform_neg, strength=0.25)
        out_high = apply_pbr_lighting(uniform_high, strength=0.25)
        out_mixed = apply_pbr_lighting(mixed, strength=0.25)

    assert len(caught) == 0, f"Unexpected RuntimeWarnings: {[str(w.message) for w in caught]}"
    assert np.all(out_neg == 0.0)
    assert np.all(out_high == 1.0)
    assert np.all(np.isfinite(out_mixed[:, :, :3]))
    assert out_mixed[:, :, :3].min() >= 0.0
    assert out_mixed[:, :, :3].max() <= 1.0
    np.testing.assert_array_equal(out_mixed[:, :, 3], mixed[:, :, 3])
