"""Tier 1: Feature Coverage E2E Tests (Isolated Happy Path, >=25 tests).

Tests the fundamental REST API capabilities:
- Health check endpoint (GET /api/health)
- Interactive preview endpoint (POST /api/preview)
- Asynchronous enhance submission (POST /api/enhance)
- Job lifecycle and polling (GET /api/jobs/{id})
- Final download endpoint (GET /api/download/{id})
- Architectural presets and parameter configuration
"""

import io
import time
from PIL import Image
import pytest


# ============================================================================
# Group 1: Health Check Endpoint (GET /api/health) — 5 tests
# ============================================================================

def test_health_status_ok(client):
    """GET /api/health must return HTTP 200 with status 'ok'."""
    response = client.get("/api/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"


def test_health_gpu_info_present(client):
    """GET /api/health must report GPU or hardware execution engine."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "gpu" in data, "Response JSON missing 'gpu' field"
    assert isinstance(data["gpu"], str), f"GPU field must be str, got {type(data['gpu'])}"
    assert len(data["gpu"]) > 0, "GPU field should not be empty"


def test_health_vram_free_positive(client):
    """GET /api/health must report available VRAM as a non-negative number."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "vram_free_mb" in data, "Response JSON missing 'vram_free_mb' field"
    vram = data["vram_free_mb"]
    assert isinstance(vram, (int, float)), f"vram_free_mb must be numeric, got {type(vram)}"
    assert vram >= 0, f"vram_free_mb must be >= 0, got {vram}"


def test_health_response_time(client):
    """GET /api/health must respond with low latency (< 200ms)."""
    start = time.perf_counter()
    response = client.get("/api/health")
    duration = time.perf_counter() - start
    assert response.status_code == 200
    assert duration < 0.20, f"Health check took too long: {duration * 1000:.1f}ms (threshold 200ms)"


def test_health_content_type_header(client):
    """GET /api/health must return application/json Content-Type header."""
    response = client.get("/api/health")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "application/json" in content_type, f"Expected application/json, got '{content_type}'"


# ============================================================================
# Group 2: Preview Endpoint (POST /api/preview) — 5 tests
# ============================================================================

def test_preview_standard_jpeg(client, synthetic_jpeg):
    """POST /api/preview with standard JPEG must return a valid preview image."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    response = client.post("/api/preview", files=files)
    assert response.status_code == 200, f"Preview failed: {response.text}"
    assert len(response.content) > 100, "Preview returned empty content"

    # Verify content can be parsed as a valid image
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0 and img.size[1] > 0, f"Invalid image dimensions: {img.size}"


def test_preview_standard_png(client, synthetic_png):
    """POST /api/preview with standard PNG must process and return image."""
    files = {"file": ("scene.png", synthetic_png, "image/png")}
    response = client.post("/api/preview", files=files)
    assert response.status_code == 200, f"Preview failed: {response.text}"
    assert len(response.content) > 100


def test_preview_warm_temperature_param(client, synthetic_jpeg):
    """POST /api/preview with custom temperature/tint parameters."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {"temperature": "0.35", "tint": "0.05"}
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200, f"Preview with params failed: {response.text}"
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0


def test_preview_shadow_and_exposure_params(client, synthetic_jpeg):
    """POST /api/preview with shadow lift and exposure parameters."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {"shadow_lift": "0.30", "exposure": "0.40"}
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200, f"Preview failed: {response.text}"


def test_preview_latency_under_threshold(client, synthetic_jpeg):
    """POST /api/preview must complete within interactive latency budget (< 1000ms)."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    start = time.perf_counter()
    response = client.post("/api/preview", files=files)
    duration = time.perf_counter() - start
    assert response.status_code == 200
    assert duration < 1.0, f"Preview took {duration:.2f}s, exceeding 1.0s target"


# ============================================================================
# Group 3: Enhance Submission & Polling (POST /api/enhance, GET /api/jobs/{id}) — 5 tests
# ============================================================================

def test_enhance_submission_returns_job_id(client, synthetic_jpeg):
    """POST /api/enhance must return HTTP 200 or 202 and a non-empty job_id."""
    files = {"file": ("input.jpg", synthetic_jpeg, "image/jpeg")}
    response = client.post("/api/enhance", files=files)
    assert response.status_code in (200, 202), f"Expected 200/202, got {response.status_code}: {response.text}"
    data = response.json()
    assert "job_id" in data, f"Missing 'job_id' in response: {data}"
    assert isinstance(data["job_id"], str) and len(data["job_id"]) > 0


def test_enhance_submission_initial_status(client, synthetic_jpeg):
    """POST /api/enhance response must indicate 'processing', 'queued', or 'completed' status."""
    files = {"file": ("input.jpg", synthetic_jpeg, "image/jpeg")}
    response = client.post("/api/enhance", files=files)
    assert response.status_code in (200, 202)
    data = response.json()
    assert "status" in data
    assert data["status"] in ("queued", "processing", "completed")


def test_job_polling_lifecycle(client, synthetic_jpeg, job_poller):
    """GET /api/jobs/{id} tracks progress from start to completion."""
    files = {"file": ("input.jpg", synthetic_jpeg, "image/jpeg")}
    submit_resp = client.post("/api/enhance", files=files)
    assert submit_resp.status_code in (200, 202)
    job_id = submit_resp.json()["job_id"]

    final_data = job_poller(client, job_id, timeout_seconds=30.0)
    assert final_data.get("status") in ("completed", "finished", "success")
    progress = final_data.get("progress", 100)
    assert progress >= 0 and progress <= 100


def test_job_polling_completed_metadata(client, synthetic_jpeg, job_poller):
    """GET /api/jobs/{id} for completed job returns resolution and result link."""
    files = {"file": ("input.jpg", synthetic_jpeg, "image/jpeg")}
    submit_resp = client.post("/api/enhance", files=files)
    job_id = submit_resp.json()["job_id"]

    data = job_poller(client, job_id, timeout_seconds=30.0)
    assert "width" in data or "output_width" in data or "result_url" in data, f"Missing metadata: {data}"


def test_job_polling_nonexistent_id_returns_404(client):
    """GET /api/jobs/{id} with non-existent job ID must return HTTP 404."""
    response = client.get("/api/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"


# ============================================================================
# Group 4: Download Endpoint (GET /api/download/{id}) — 5 tests
# ============================================================================

def test_download_completed_job_image(client, synthetic_jpeg, job_poller):
    """GET /api/download/{id} returns binary stream with enhanced image."""
    files = {"file": ("input.jpg", synthetic_jpeg, "image/jpeg")}
    submit_resp = client.post("/api/enhance", files=files)
    job_id = submit_resp.json()["job_id"]
    job_poller(client, job_id, timeout_seconds=30.0)

    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200, f"Download failed: {dl_resp.status_code}"
    assert len(dl_resp.content) > 500, "Downloaded content too small"


def test_download_content_type_header(client, synthetic_jpeg, job_poller):
    """GET /api/download/{id} returns proper image mime type."""
    files = {"file": ("input.jpg", synthetic_jpeg, "image/jpeg")}
    submit_resp = client.post("/api/enhance", files=files)
    job_id = submit_resp.json()["job_id"]
    job_poller(client, job_id, timeout_seconds=30.0)

    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200
    ctype = dl_resp.headers.get("content-type", "")
    assert any(img_t in ctype for img_t in ("image/png", "image/jpeg", "image/webp", "octet-stream")), \
        f"Unexpected content-type: {ctype}"


def test_download_image_integrity(client, synthetic_jpeg, job_poller):
    """Downloaded file can be parsed into valid image object."""
    files = {"file": ("input.jpg", synthetic_jpeg, "image/jpeg")}
    submit_resp = client.post("/api/enhance", files=files)
    job_id = submit_resp.json()["job_id"]
    job_poller(client, job_id, timeout_seconds=30.0)

    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200
    img = Image.open(io.BytesIO(dl_resp.content))
    assert img.size[0] > 0 and img.size[1] > 0


def test_download_nonexistent_job_id_returns_404(client):
    """GET /api/download/{id} with unknown job ID must return HTTP 404."""
    response = client.get("/api/download/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"


def test_download_content_disposition_header(client, synthetic_jpeg, job_poller):
    """GET /api/download/{id} provides appropriate download disposition."""
    files = {"file": ("input.jpg", synthetic_jpeg, "image/jpeg")}
    submit_resp = client.post("/api/enhance", files=files)
    job_id = submit_resp.json()["job_id"]
    job_poller(client, job_id, timeout_seconds=30.0)

    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200
    cd = dl_resp.headers.get("content-disposition", "")
    assert len(cd) > 0, "Missing Content-Disposition header in download response"
    assert any(term in cd.lower() for term in ("attachment", "filename")), f"Invalid Content-Disposition: {cd}"


# ============================================================================
# Group 5: Presets & Explicit Parameter Configuration — 5 tests
# ============================================================================

def test_preset_natural_config(client, synthetic_jpeg):
    """Preset Natural config: balanced architectural calibration."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "temperature": "0.15",
        "shadow_lift": "0.20",
        "highlight_recovery": "0.10",
        "clahe_clip_limit": "1.5",
        "sharpness": "0.30"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200, f"Preset Natural failed: {response.text}"


def test_preset_marcenaria_config(client, synthetic_jpeg):
    """Preset Marcenaria: micro-contrast and edge boost for woodwork and joinery."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "ambient_occlusion": "0.25",
        "sharpness": "0.60",
        "clahe_clip_limit": "2.5",
        "shadow_lift": "0.25"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200, f"Preset Marcenaria failed: {response.text}"


def test_preset_luz_quente_config(client, synthetic_jpeg):
    """Preset Luz Quente: warm interior illumination (3000K simulation)."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "temperature": "0.50",
        "tint": "0.10",
        "exposure": "0.10",
        "shadow_lift": "0.30"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200, f"Preset Luz Quente failed: {response.text}"


def test_preset_ultra_detalhe_config(client, synthetic_jpeg):
    """Preset Ultra Detalhe: maximum texture resolution and clarity."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "sharpness": "0.80",
        "clahe_clip_limit": "3.0",
        "denoise_strength": "0.15",
        "ambient_occlusion": "0.20"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200, f"Preset Ultra Detalhe failed: {response.text}"


def test_explicit_valid_parameter_ranges(client, synthetic_jpeg):
    """Full parameter payload with all 12 EnhancementConfig fields within valid bounds."""
    files = {"file": ("render.jpg", synthetic_jpeg, "image/jpeg")}
    data = {
        "temperature": "0.20",
        "tint": "-0.05",
        "exposure": "0.25",
        "shadow_lift": "0.22",
        "highlight_recovery": "0.14",
        "clahe_clip_limit": "2.1",
        "ambient_occlusion": "0.16",
        "sharpness": "0.45",
        "denoise_strength": "0.08",
        "upscale_factor": "2",
        "tile_size": "256",
        "preserve_geometry": "true"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200, f"Explicit params failed: {response.text}"
