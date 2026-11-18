"""Tier 2: Boundary & Corner Cases E2E Tests for R2-R4.

Tests schema limits, out-of-bounds rejection (HTTP 422), backward-compatible omission
of R4 parameters, extreme image dimensions under Multi-Scale SSAO (r=35) + Soft GI Bloom,
uniform flat gray surface invariance, and highlight protection under max AO + Bloom.
"""

import io

import numpy as np
import pytest
from PIL import Image


def _run_enhance_png(client, job_poller, image_bytes: bytes, filename: str, data: dict) -> bytes:
    files = {"file": (filename, image_bytes, "image/png")}
    resp = client.post("/api/enhance", files=files, data=data)
    assert resp.status_code in (200, 202), f"/api/enhance failed ({resp.status_code}): {resp.text}"
    job_id = resp.json()["job_id"]
    job_data = job_poller(client, job_id, timeout_seconds=45.0)
    assert job_data.get("status") in ("completed", "finished", "success")
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    return dl.content


# ============================================================================
# Group 1: R4 Schema Enum & Numeric Bounds Validation (HTTP 200 vs 422)
# ============================================================================

def test_t2_r4_render_mode_valid_and_invalid(client, synthetic_jpeg, job_poller):
    """render_mode accepts 'classic'/'generative' (200) and rejects invalid enum values (422)."""
    for valid_mode in ("classic", "generative"):
        prev = client.post(
            "/api/preview",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"render_mode": valid_mode},
        )
        assert prev.status_code == 200, f"/api/preview rejected valid render_mode={valid_mode!r}: {prev.text}"

        enh = client.post(
            "/api/enhance",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"render_mode": valid_mode, "generative_strength": "0.0", "upscale_factor": "1"},
        )
        assert enh.status_code in (200, 202), f"/api/enhance rejected valid render_mode={valid_mode!r}"
        job_poller(client, enh.json()["job_id"], timeout_seconds=30.0)

    for invalid_mode in ("hybrid", "sd15", "", "CLASSIC"):
        prev = client.post(
            "/api/preview",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"render_mode": invalid_mode},
        )
        assert prev.status_code == 422, (
            f"/api/preview expected 422 for render_mode={invalid_mode!r}, got {prev.status_code}"
        )
        enh = client.post(
            "/api/enhance",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"render_mode": invalid_mode, "upscale_factor": "1"},
        )
        assert enh.status_code == 422, (
            f"/api/enhance expected 422 for render_mode={invalid_mode!r}, got {enh.status_code}"
        )


def test_t2_r4_generative_strength_bounds_and_invalid(client, synthetic_jpeg, job_poller):
    """generative_strength accepts [0.0, 0.48, 1.0] (200) and rejects OOB/NaN/non-numeric (422)."""
    for valid_val in ("0.0", "0.48", "1.0"):
        prev = client.post(
            "/api/preview",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"render_mode": "generative", "generative_strength": valid_val},
        )
        assert prev.status_code == 200, f"Expected 200 for generative_strength={valid_val}, got {prev.text}"

    # Verify boundary endpoint 0.0 on /api/enhance without invoking diffusion
    enh_zero = client.post(
        "/api/enhance",
        files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
        data={"render_mode": "generative", "generative_strength": "0.0", "upscale_factor": "1"},
    )
    assert enh_zero.status_code in (200, 202)
    job_poller(client, enh_zero.json()["job_id"], timeout_seconds=30.0)

    for invalid_val in ("-0.01", "1.01", "nan", "high"):
        prev = client.post(
            "/api/preview",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"generative_strength": invalid_val},
        )
        assert prev.status_code == 422, (
            f"/api/preview expected 422 for generative_strength={invalid_val!r}, got {prev.status_code}"
        )
        enh = client.post(
            "/api/enhance",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"generative_strength": invalid_val, "upscale_factor": "1"},
        )
        assert enh.status_code == 422, (
            f"/api/enhance expected 422 for generative_strength={invalid_val!r}, got {enh.status_code}"
        )


def test_t2_r4_scene_profile_valid_and_invalid(client, synthetic_jpeg, job_poller):
    """scene_profile accepts valid enums (200) and rejects invalid options (422)."""
    for valid_profile in ("auto", "banheiro", "cozinha", "quarto_sala"):
        prev = client.post(
            "/api/preview",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"scene_profile": valid_profile},
        )
        assert prev.status_code == 200, f"/api/preview rejected scene_profile={valid_profile!r}"

        enh = client.post(
            "/api/enhance",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={
                "render_mode": "generative",
                "generative_strength": "0.0",
                "scene_profile": valid_profile,
                "upscale_factor": "1",
            },
        )
        assert enh.status_code in (200, 202)
        job_poller(client, enh.json()["job_id"], timeout_seconds=30.0)

    for invalid_profile in ("garagem", "sala", "", "escritorio"):
        prev = client.post(
            "/api/preview",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"scene_profile": invalid_profile},
        )
        assert prev.status_code == 422, (
            f"/api/preview expected 422 for scene_profile={invalid_profile!r}, got {prev.status_code}"
        )
        enh = client.post(
            "/api/enhance",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"scene_profile": invalid_profile, "upscale_factor": "1"},
        )
        assert enh.status_code == 422, (
            f"/api/enhance expected 422 for scene_profile={invalid_profile!r}, got {enh.status_code}"
        )


def test_t2_r4_specular_bloom_bounds_and_invalid(client, synthetic_jpeg, job_poller):
    """specular_bloom accepts [0.0, 0.25, 1.0] (200) and rejects OOB/non-numeric (422)."""
    for valid_bloom in ("0.0", "0.25", "1.0"):
        prev = client.post(
            "/api/preview",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"specular_bloom": valid_bloom},
        )
        assert prev.status_code == 200, f"/api/preview rejected specular_bloom={valid_bloom!r}"

        enh = client.post(
            "/api/enhance",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"specular_bloom": valid_bloom, "upscale_factor": "1"},
        )
        assert enh.status_code in (200, 202)
        job_poller(client, enh.json()["job_id"], timeout_seconds=30.0)

    for invalid_bloom in ("-0.1", "1.5", "abc"):
        prev = client.post(
            "/api/preview",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"specular_bloom": invalid_bloom},
        )
        assert prev.status_code == 422, (
            f"/api/preview expected 422 for specular_bloom={invalid_bloom!r}, got {prev.status_code}"
        )
        enh = client.post(
            "/api/enhance",
            files={"file": ("img.jpg", synthetic_jpeg, "image/jpeg")},
            data={"specular_bloom": invalid_bloom, "upscale_factor": "1"},
        )
        assert enh.status_code == 422, (
            f"/api/enhance expected 422 for specular_bloom={invalid_bloom!r}, got {enh.status_code}"
        )


def test_t2_r4_backward_compatible_omission_of_new_params(client, synthetic_jpeg, job_poller):
    """Omitting all 4 new R4 parameters preserves backward-compatible classic defaults."""
    prev = client.post("/api/preview", files={"file": ("legacy.jpg", synthetic_jpeg, "image/jpeg")})
    assert prev.status_code == 200
    assert Image.open(io.BytesIO(prev.content)).size[0] > 0

    enh = client.post(
        "/api/enhance",
        files={"file": ("legacy.jpg", synthetic_jpeg, "image/jpeg")},
        data={"temperature": "0.10", "upscale_factor": "1"},
    )
    assert enh.status_code in (200, 202)
    job_id = enh.json()["job_id"]
    job_data = job_poller(client, job_id, timeout_seconds=30.0)
    assert job_data.get("status") in ("completed", "finished", "success")
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    assert Image.open(io.BytesIO(dl.content)).format == "PNG"


# ============================================================================
# Group 2: R2 Geometric Dimensions, Uniform Invariance & Highlight Protection
# ============================================================================

def test_t2_r2_extreme_dimensions_max_ssao_and_bloom(
    client,
    synthetic_tiny_image,
    synthetic_banner_image,
    synthetic_tall_image,
    synthetic_odd_image,
):
    """Extreme dimensions (16x16, 1920x64, 64x1920, 1001x733) handle r=35 SSAO + Bloom=1.0 cleanly."""
    cases = [
        ("tiny_16x16.jpg", synthetic_tiny_image),
        ("banner_1920x64.jpg", synthetic_banner_image),
        ("tall_64x1920.jpg", synthetic_tall_image),
        ("odd_1001x733.jpg", synthetic_odd_image),
    ]
    for name, payload in cases:
        resp = client.post(
            "/api/preview",
            files={"file": (name, payload, "image/jpeg")},
            data={
                "ambient_occlusion": "1.0",
                "specular_bloom": "1.0",
                "render_mode": "classic",
            },
        )
        assert resp.status_code == 200, f"Extreme dim {name} failed: {resp.text}"
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        assert arr.shape[0] > 0 and arr.shape[1] > 0
        assert not np.isnan(arr).any(), f"NaN detected on {name}"


def test_t2_r2_extreme_odd_dimensions_enhance_bypass(client, synthetic_odd_image, job_poller):
    """Odd dimensions (1001x733) pass /api/enhance cleanly with render_mode='generative', strength=0.0."""
    resp = client.post(
        "/api/enhance",
        files={"file": ("odd.jpg", synthetic_odd_image, "image/jpeg")},
        data={
            "render_mode": "generative",
            "generative_strength": "0.0",
            "ambient_occlusion": "1.0",
            "specular_bloom": "1.0",
            "upscale_factor": "1",
        },
    )
    assert resp.status_code in (200, 202)
    job_id = resp.json()["job_id"]
    job_poller(client, job_id, timeout_seconds=30.0)
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    img = Image.open(io.BytesIO(dl.content))
    assert img.size == (1001, 733)


def test_t2_r2_uniform_flat_gray_ao_invariance(client, job_poller):
    """Uniform flat gray (128,128,128) PNG with ambient_occlusion=1.0 must remain invariant without edge halos."""
    flat = np.full((256, 256, 3), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(flat, mode="RGB").save(buf, format="PNG")

    out_png = _run_enhance_png(
        client,
        job_poller,
        buf.getvalue(),
        "flat_gray.png",
        {
            "temperature": "0.0",
            "tint": "0.0",
            "exposure": "0.0",
            "shadow_lift": "0.0",
            "highlight_recovery": "0.0",
            "clahe_clip_limit": "0.0",
            "ambient_occlusion": "1.0",
            "specular_bloom": "0.0",
            "sharpness": "0.0",
            "denoise_strength": "0.0",
            "render_mode": "classic",
            "upscale_factor": "1",
        },
    )
    arr = np.asarray(Image.open(io.BytesIO(out_png)).convert("RGB"), dtype=np.float32)
    assert float(np.std(arr)) < 0.5, f"Uniform flat gray produced non-zero std={np.std(arr):.4f}"
    assert np.allclose(arr, 128.0, atol=1.0), f"Uniform gray drifted from 128 (mean={arr.mean():.2f})"


def test_t2_r2_specular_bloom_zero_identity_on_enhance(client, synthetic_png, job_poller):
    """specular_bloom=0.0 must be identical to baseline with all enhancements neutral."""
    neutral = {
        "temperature": "0.0",
        "tint": "0.0",
        "exposure": "0.0",
        "shadow_lift": "0.0",
        "highlight_recovery": "0.0",
        "clahe_clip_limit": "0.0",
        "ambient_occlusion": "0.0",
        "specular_bloom": "0.0",
        "sharpness": "0.0",
        "denoise_strength": "0.0",
        "render_mode": "classic",
        "upscale_factor": "1",
    }
    out_png = _run_enhance_png(client, job_poller, synthetic_png, "identity.png", neutral)
    in_arr = np.asarray(Image.open(io.BytesIO(synthetic_png)).convert("RGB"))
    out_arr = np.asarray(Image.open(io.BytesIO(out_png)).convert("RGB"))
    assert np.array_equal(in_arr, out_arr), "Neutral specular_bloom=0.0 modified pixels unexpectedly"


def test_t2_r2_highlight_protection_under_max_ao_and_bloom(client, job_poller):
    """Pure-white (255,255,255) patch stays bright (>=245) under max AO=1.0 + Bloom=1.0 while dark crevice darkens."""
    img = np.full((256, 256, 3), 190, dtype=np.uint8)
    img[40:100, 40:100] = (255, 255, 255)  # Pure-white specular patch
    img[150:220, 178:182] = (100, 100, 100)  # 4px dark crevice

    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    raw_png = buf.getvalue()

    neutral = {
        "temperature": "0.0",
        "tint": "0.0",
        "exposure": "0.0",
        "shadow_lift": "0.0",
        "highlight_recovery": "0.0",
        "clahe_clip_limit": "0.0",
        "sharpness": "0.0",
        "denoise_strength": "0.0",
        "render_mode": "classic",
        "upscale_factor": "1",
    }

    base_png = _run_enhance_png(
        client,
        job_poller,
        raw_png,
        "hl_base.png",
        {**neutral, "ambient_occlusion": "0.0", "specular_bloom": "0.0"},
    )
    max_png = _run_enhance_png(
        client,
        job_poller,
        raw_png,
        "hl_max.png",
        {**neutral, "ambient_occlusion": "1.0", "specular_bloom": "1.0"},
    )

    base_arr = np.asarray(Image.open(io.BytesIO(base_png)).convert("RGB"), dtype=np.float32)
    max_arr = np.asarray(Image.open(io.BytesIO(max_png)).convert("RGB"), dtype=np.float32)

    white_mean = float(np.mean(max_arr[50:90, 50:90]))
    assert white_mean >= 245.0, f"Pure-white highlight darkened under max AO+Bloom: {white_mean:.2f}"

    base_crevice = float(np.mean(base_arr[160:210, 178:182]))
    max_crevice = float(np.mean(max_arr[160:210, 178:182]))
    assert max_crevice < base_crevice - 2.0, (
        f"Dark crevice failed to darken under max AO+Bloom: base={base_crevice:.2f}, max={max_crevice:.2f}"
    )
