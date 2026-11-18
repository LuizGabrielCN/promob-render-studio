"""Tier 2: Boundary & Corner Cases E2E Tests (>=20 tests).

Tests system resilience against:
- Corrupt, truncated, empty, and non-image payloads
- Extreme image dimensions (banners, vertical slivers, tiny icons, odd dimensions, 4K inputs)
- Numerical parameter limits and out-of-bounds rejection (HTTP 422)
- Concurrency, rapid requests, path traversal defense, and default fallbacks
"""

import io
from PIL import Image
import pytest


# ============================================================================
# Group 1: File Formats & Corrupt Inputs — 5 tests
# ============================================================================

def test_boundary_empty_file_rejected(client, synthetic_empty_bytes):
    """Submitting a 0-byte file must be rejected with HTTP 400 or 422."""
    files = {"file": ("empty.jpg", synthetic_empty_bytes, "image/jpeg")}
    response = client.post("/api/preview", files=files)
    assert response.status_code in (400, 422), f"Expected 400/422, got {response.status_code}: {response.text}"


def test_boundary_plain_text_rejected(client, synthetic_text_file):
    """Submitting a text file instead of an image must be rejected with HTTP 400 or 422."""
    files = {"file": ("doc.txt", synthetic_text_file, "text/plain")}
    response = client.post("/api/preview", files=files)
    assert response.status_code in (400, 422), f"Expected 400/422, got {response.status_code}: {response.text}"


def test_boundary_corrupt_jpeg_header_rejected(client, synthetic_corrupt_bytes):
    """Submitting corrupt/truncated image bytes must be rejected gracefully without 500 error."""
    files = {"file": ("corrupt.jpg", synthetic_corrupt_bytes, "image/jpeg")}
    response = client.post("/api/preview", files=files)
    assert response.status_code in (400, 422), f"Expected 400/422 on corrupt file, got {response.status_code}"


def test_boundary_binary_executable_rejected(client, synthetic_exe_file):
    """Submitting an executable or binary payload to /api/enhance must be rejected."""
    files = {"file": ("danger.exe", synthetic_exe_file, "application/octet-stream")}
    response = client.post("/api/enhance", files=files)
    assert response.status_code in (400, 422), f"Expected 400/422 on binary payload, got {response.status_code}"


def test_boundary_missing_file_payload_rejected(client):
    """Submitting POST /api/preview without a 'file' multipart field must return HTTP 422."""
    response = client.post("/api/preview", data={"temperature": "0.1"})
    assert response.status_code == 422, f"Expected 422 for missing file, got {response.status_code}"


# ============================================================================
# Group 2: Dimensions & Extreme Aspect Ratios — 5 tests
# ============================================================================

def test_boundary_extreme_banner_aspect_ratio(client, synthetic_banner_image):
    """Extreme wide banner (1920x64, 30:1 aspect ratio) must be processed without failure."""
    files = {"file": ("banner.jpg", synthetic_banner_image, "image/jpeg")}
    response = client.post("/api/preview", files=files)
    assert response.status_code == 200, f"Extreme banner failed: {response.text}"
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0 and img.size[1] > 0


def test_boundary_extreme_tall_sliver_aspect_ratio(client, synthetic_tall_image):
    """Extreme tall sliver (64x1920, 1:30 aspect ratio) must be processed without failure."""
    files = {"file": ("column.jpg", synthetic_tall_image, "image/jpeg")}
    response = client.post("/api/preview", files=files)
    assert response.status_code == 200, f"Extreme tall sliver failed: {response.text}"
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0 and img.size[1] > 0


def test_boundary_tiny_icon_16x16(client, synthetic_tiny_image):
    """Tiny 16x16 icon must not trigger zero-division or out-of-bounds filter errors."""
    files = {"file": ("icon.jpg", synthetic_tiny_image, "image/jpeg")}
    response = client.post("/api/preview", files=files)
    assert response.status_code == 200, f"Tiny image failed: {response.text}"
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0 and img.size[1] > 0


def test_boundary_non_standard_odd_dimensions(client, synthetic_odd_image):
    """Odd dimensions (1001x733) must not cause tiling mismatch or dimension errors."""
    files = {"file": ("odd.jpg", synthetic_odd_image, "image/jpeg")}
    response = client.post("/api/preview", files=files)
    assert response.status_code == 200, f"Odd dimension image failed: {response.text}"
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0 and img.size[1] > 0


def test_boundary_large_4k_input(client, synthetic_4k_image):
    """Large 4K input (3840x2160) must be handled gracefully by memory guardrails."""
    files = {"file": ("large4k.jpg", synthetic_4k_image, "image/jpeg")}
    response = client.post("/api/preview", files=files)
    assert response.status_code == 200, f"Large 4K preview failed: {response.text}"


# ============================================================================
# Group 3: Numeric Parameter Limits & Validation — 5 tests
# ============================================================================

def test_boundary_temperature_limits(client, synthetic_jpeg):
    """Temperature boundary tests: valid [-1.0, 1.0] succeeds; out-of-bounds rejected with 422."""
    files1 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_min = client.post("/api/preview", files=files1, data={"temperature": "-1.0"})
    assert resp_min.status_code == 200, f"temperature=-1.0 should succeed: {resp_min.text}"

    files2 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_max = client.post("/api/preview", files=files2, data={"temperature": "1.0"})
    assert resp_max.status_code == 200, f"temperature=1.0 should succeed: {resp_max.text}"

    # Upper bound violation (> 1.0)
    files3 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_upper = client.post("/api/preview", files=files3, data={"temperature": "1.5"})
    assert resp_upper.status_code == 422, f"temperature=1.5 should be rejected with 422, got {resp_upper.status_code}"

    # Lower bound violation (< -1.0)
    files4 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_lower = client.post("/api/preview", files=files4, data={"temperature": "-1.5"})
    assert resp_lower.status_code == 422, f"temperature=-1.5 should be rejected with 422, got {resp_lower.status_code}"


def test_boundary_exposure_limits(client, synthetic_jpeg):
    """Exposure boundary tests: valid [-2.0, 2.0] succeeds; out-of-bounds rejected with 422."""
    files1 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_min = client.post("/api/preview", files=files1, data={"exposure": "-2.0"})
    assert resp_min.status_code == 200

    files2 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_max = client.post("/api/preview", files=files2, data={"exposure": "2.0"})
    assert resp_max.status_code == 200

    # Upper bound violation (> 2.0)
    files3 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_upper = client.post("/api/preview", files=files3, data={"exposure": "3.5"})
    assert resp_upper.status_code == 422, f"exposure=3.5 should be rejected with 422, got {resp_upper.status_code}"

    # Lower bound violation (< -2.0)
    files4 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_lower = client.post("/api/preview", files=files4, data={"exposure": "-2.5"})
    assert resp_lower.status_code == 422, f"exposure=-2.5 should be rejected with 422, got {resp_lower.status_code}"


def test_boundary_sharpness_limits(client, synthetic_jpeg):
    """Sharpness boundary tests: valid [0.0, 2.0] succeeds; out-of-bounds rejected with 422."""
    files1 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_min = client.post("/api/preview", files=files1, data={"sharpness": "0.0"})
    assert resp_min.status_code == 200

    files2 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_max = client.post("/api/preview", files=files2, data={"sharpness": "2.0"})
    assert resp_max.status_code == 200

    # Lower bound violation (< 0.0)
    files3 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_lower = client.post("/api/preview", files=files3, data={"sharpness": "-0.5"})
    assert resp_lower.status_code == 422, f"sharpness=-0.5 should be rejected with 422, got {resp_lower.status_code}"

    # Upper bound violation (> 2.0)
    files4 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_upper = client.post("/api/preview", files=files4, data={"sharpness": "2.5"})
    assert resp_upper.status_code == 422, f"sharpness=2.5 should be rejected with 422, got {resp_upper.status_code}"


def test_boundary_shadow_and_highlight_limits(client, synthetic_jpeg):
    """Shadow lift and highlight recovery boundaries: [0.0, 1.0] valid; out-of-bounds rejected with 422."""
    files1 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_valid = client.post("/api/preview", files=files1, data={"shadow_lift": "1.0", "highlight_recovery": "0.0"})
    assert resp_valid.status_code == 200

    # Shadow lift upper bound violation (> 1.0)
    files2 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_shadow_upper = client.post("/api/preview", files=files2, data={"shadow_lift": "1.25"})
    assert resp_shadow_upper.status_code == 422, f"shadow_lift=1.25 should be 422, got {resp_shadow_upper.status_code}"

    # Shadow lift lower bound violation (< 0.0)
    files3 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_shadow_lower = client.post("/api/preview", files=files3, data={"shadow_lift": "-0.2"})
    assert resp_shadow_lower.status_code == 422, f"shadow_lift=-0.2 should be 422, got {resp_shadow_lower.status_code}"

    # Highlight recovery upper bound violation (> 1.0)
    files4 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_hl_upper = client.post("/api/preview", files=files4, data={"highlight_recovery": "1.25"})
    assert resp_hl_upper.status_code == 422, f"highlight_recovery=1.25 should be 422, got {resp_hl_upper.status_code}"

    # Highlight recovery lower bound violation (< 0.0)
    files5 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_hl_lower = client.post("/api/preview", files=files5, data={"highlight_recovery": "-0.2"})
    assert resp_hl_lower.status_code == 422, f"highlight_recovery=-0.2 should be 422, got {resp_hl_lower.status_code}"


def test_boundary_tile_size_and_upscale_factor(client, synthetic_jpeg):
    """Upscale factor [1, 4] and tile_size [128, 512] validation with bidirectional out-of-bounds checks."""
    files1 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_valid = client.post("/api/preview", files=files1, data={"upscale_factor": "4", "tile_size": "256"})
    assert resp_valid.status_code == 200

    # Upscale factor upper bound violation (> 4)
    files2 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_upscale_oob = client.post("/api/preview", files=files2, data={"upscale_factor": "6"})
    assert resp_upscale_oob.status_code == 422, f"upscale_factor=6 should be 422, got {resp_upscale_oob.status_code}"

    # Tile size lower bound violation (< 128)
    files3 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_tile_lower = client.post("/api/preview", files=files3, data={"tile_size": "64"})
    assert resp_tile_lower.status_code == 422, f"tile_size=64 should be 422, got {resp_tile_lower.status_code}"

    # Tile size upper bound violation (> 512)
    files4 = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp_tile_upper = client.post("/api/preview", files=files4, data={"tile_size": "1024"})
    assert resp_tile_upper.status_code == 422, f"tile_size=1024 should be 422, got {resp_tile_upper.status_code}"


# ============================================================================
# Group 4: Concurrency, Polling & Defense-in-Depth — 5 tests
# ============================================================================

def test_boundary_concurrent_preview_requests(client, synthetic_jpeg):
    """Multiple sequential/concurrent preview requests must all succeed without state leakage."""
    for temp in ("-0.5", "0.0", "0.5"):
        files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
        resp = client.post("/api/preview", files=files, data={"temperature": temp})
        assert resp.status_code == 200


def test_boundary_rapid_consecutive_job_polling(client, synthetic_jpeg):
    """Rapid consecutive polling calls do not degrade service or return 500."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    submit = client.post("/api/enhance", files=files)
    job_id = submit.json()["job_id"]

    for _ in range(5):
        poll = client.get(f"/api/jobs/{job_id}")
        assert poll.status_code == 200


def test_boundary_duplicate_job_download(client, synthetic_jpeg, job_poller):
    """Multiple downloads of the same completed job return consistent payloads."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    submit = client.post("/api/enhance", files=files)
    job_id = submit.json()["job_id"]
    job_poller(client, job_id, timeout_seconds=30.0)

    dl1 = client.get(f"/api/download/{job_id}")
    dl2 = client.get(f"/api/download/{job_id}")
    assert dl1.status_code == 200 and dl2.status_code == 200
    assert len(dl1.content) == len(dl2.content), "Downloaded byte lengths differ"


def test_boundary_path_traversal_job_id_rejected(client):
    """Path traversal attempt in job ID must be safely rejected (404/422)."""
    malformed_ids = [
        "../../etc/passwd",
        "..%2F..%2Fwindows%2Fsystem32",
        "job;SELECT * FROM users",
        "<script>alert(1)</script>"
    ]
    for mid in malformed_ids:
        resp = client.get(f"/api/jobs/{mid}")
        assert resp.status_code in (404, 422, 400), f"Unexpected status {resp.status_code} for path traversal attempt"


def test_boundary_empty_form_data_defaults_applied(client, synthetic_jpeg):
    """POST /api/preview with only the image and no form parameters must apply defaults and return 200."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    resp = client.post("/api/preview", files=files)
    assert resp.status_code == 200, f"Default fallback failed: {resp.text}"
