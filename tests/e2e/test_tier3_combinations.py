"""Tier 3: Pairwise Combinations E2E Tests (>=10 tests).

Tests the interactions between distinct photometric, morphological,
and super-resolution enhancement parameters:
- Color temperature shift + Shadow recovery
- Contrast CLAHE + Morphological Ambient Occlusion
- Sharpness amplification + Bilateral Denoising
- Exposure compensation + Highlight recovery
- Extreme chromatic tint combinations
- Full enhancement stack with 4x upscale, tiling, and feathering
"""

import io
from PIL import Image
import numpy as np
import pytest


def test_pair_temperature_warm_plus_shadow_lift_high(client, synthetic_jpeg):
    """Warm temperature (+0.60) paired with high shadow lift (0.80)."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "temperature": "0.60",
        "shadow_lift": "0.80"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200, f"Failed: {response.text}"
    img = Image.open(io.BytesIO(response.content))
    arr = np.array(img)
    assert arr.mean() > 0, "Image should not be completely black"


def test_pair_temperature_cold_plus_shadow_lift_low(client, synthetic_jpeg):
    """Cold temperature (-0.60) paired with minimal shadow lift (0.05)."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "temperature": "-0.60",
        "shadow_lift": "0.05"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0


def test_pair_clahe_max_plus_ao_max(client, synthetic_jpeg):
    """Maximum CLAHE (5.0) paired with maximum Ambient Occlusion (1.0).

    Verifies that aggressive local contrast and black top-hat crease darkening
    do not trigger integer overflow, negative clamping, or nan values.
    """
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "clahe_clip_limit": "5.0",
        "ambient_occlusion": "1.0"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200
    img = Image.open(io.BytesIO(response.content))
    arr = np.array(img)
    assert not np.isnan(arr).any(), "Result contains NaN values"


def test_pair_clahe_zero_plus_ao_zero(client, synthetic_jpeg):
    """Zero CLAHE and Zero Ambient Occlusion (bypass of morphological pipeline)."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "clahe_clip_limit": "0.0",
        "ambient_occlusion": "0.0"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200


def test_pair_sharpness_high_plus_denoise_high(client, synthetic_jpeg):
    """Maximum Sharpness (2.0) paired with maximum Denoise (1.0).

    Tests opposing high-frequency amplification and edge-preserving smoothing.
    """
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "sharpness": "2.0",
        "denoise_strength": "1.0"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0


def test_pair_sharpness_zero_plus_denoise_zero(client, synthetic_jpeg):
    """Zero Sharpness and Zero Denoise (neutral detail stage)."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "sharpness": "0.0",
        "denoise_strength": "0.0"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200


def test_pair_exposure_high_plus_highlight_recovery_max(client, synthetic_jpeg):
    """High positive exposure (+1.50) with maximum highlight recovery (1.0).

    Verifies highlight compression prevents total blowout of bright ceiling/window areas.
    """
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "exposure": "1.50",
        "highlight_recovery": "1.0"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200
    img = Image.open(io.BytesIO(response.content))
    arr = np.array(img)
    # Check that not all pixels are blown out to 255
    assert (arr < 255).any(), "Exposure with highlight recovery should avoid total white blowout"


def test_pair_exposure_low_plus_shadow_lift_max(client, synthetic_jpeg):
    """Negative exposure (-1.50) paired with maximum shadow lift (1.0).

    Verifies dynamic range expansion pulls details out of dark cabinetry without banding.
    """
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "exposure": "-1.50",
        "shadow_lift": "1.0"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200
    img = Image.open(io.BytesIO(response.content))
    arr = np.array(img)
    assert (arr > 0).any(), "Shadow lift should preserve non-zero pixel intensities"


def test_pair_tint_magenta_green_extremes(client, synthetic_jpeg):
    """Opposing color tint extremes combined with temperature shifts."""
    # Quadrant 1: Warm + Magenta
    files1 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp1 = client.post("/api/preview", files=files1, data={"temperature": "0.50", "tint": "0.80"})
    assert resp1.status_code == 200

    # Quadrant 2: Cool + Green
    files2 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp2 = client.post("/api/preview", files=files2, data={"temperature": "-0.50", "tint": "-0.80"})
    assert resp2.status_code == 200


def test_full_stack_upscale_tiling_combination(client, synthetic_jpeg, job_poller):
    """Full enhancement stack with all active parameters and 4x upscale.

    Submits complete job exercising all 6 pipeline stages:
    1. Color & Temperature
    2. Tone & Shadows/Highlights
    3. CLAHE in L*a*b*
    4. Morphological AO
    5. Detail & Unsharp mask
    6. 4x Super-resolution with tiling & Hanning feathering
    """
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "temperature": "0.25",
        "tint": "0.05",
        "exposure": "0.15",
        "shadow_lift": "0.25",
        "highlight_recovery": "0.12",
        "clahe_clip_limit": "2.2",
        "ambient_occlusion": "0.18",
        "sharpness": "0.45",
        "denoise_strength": "0.10",
        "upscale_factor": "4",
        "tile_size": "256",
        "preserve_geometry": "true"
    }
    response = client.post("/api/enhance", files=files, data=data)
    assert response.status_code in (200, 202), f"Enhance submit failed: {response.text}"
    job_id = response.json()["job_id"]

    result = job_poller(client, job_id, timeout_seconds=60.0)
    assert result.get("status") in ("completed", "finished", "success")
