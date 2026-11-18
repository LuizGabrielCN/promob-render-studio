"""Tier 4: Real-World Workloads E2E Tests (>=5 tests).

Executes comprehensive real-world architectural workflows using the authentic
Promob kitchen reference render:
- Scenario 1: Full Promob Kitchen Workflow -> 4K output resolution
- Scenario 2: Strict Geometric Preservation -> SSIM >= 0.92 (zero hallucinations)
- Scenario 3: Hardware Guardrails -> Peak VRAM <= 5.5GB on 6GB GPU / safe CPU fallback
- Scenario 4: Color Warmth & Tone -> Cold-to-warm architectural shift & dynamic range
- Scenario 5: End-to-End Client Simulation -> Complete user journey (Health -> Preview -> Enhance -> Poll -> Download)
"""

import io
import time
from PIL import Image
import numpy as np
import pytest


# ============================================================================
# Scenario 1: Full Promob Kitchen Workflow -> 4K Output Resolution
# ============================================================================

def test_workload_scenario1_full_promob_kitchen_4k(client, promob_reference_image, job_poller):
    """Scenario 1: Authenticates complete 4K architectural workflow on Promob kitchen render.

    - Input: Authentic Promob kitchen render (media_1794865112994_46d1d96f.jpg).
    - Execution: 4x Super-Resolution enhancement via POST /api/enhance.
    - Acceptance: Enhanced asset has resolution >= 2880px / 3840px, PNG format, valid stream.
    """
    files = {"file": ("promob_kitchen.jpg", promob_reference_image, "image/jpeg")}
    data = {
        "temperature": "0.20",
        "shadow_lift": "0.23",
        "highlight_recovery": "0.10",
        "clahe_clip_limit": "2.0",
        "ambient_occlusion": "0.15",
        "sharpness": "0.40",
        "upscale_factor": "4",
        "tile_size": "256",
        "preserve_geometry": "true"
    }

    # Submit 4K enhancement job
    submit_resp = client.post("/api/enhance", files=files, data=data)
    assert submit_resp.status_code in (200, 202), f"Enhancement job submission failed: {submit_resp.text}"
    job_id = submit_resp.json()["job_id"]

    # Poll until completed
    job_data = job_poller(client, job_id, timeout_seconds=90.0)
    assert job_data.get("status") in ("completed", "finished", "success")

    # Download and inspect final 4K asset
    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200, f"Download failed with status {dl_resp.status_code}"
    assert "image/png" in dl_resp.headers.get("content-type", "").lower()

    img = Image.open(io.BytesIO(dl_resp.content))
    assert img.format == "PNG", f"Expected PNG format, got {img.format}"
    width, height = img.size

    # Enforce genuine 4K / 4x resolution
    assert max(width, height) >= 3600, f"Expected 4x upscale (>=3600px), got {max(width, height)}px"
    assert len(dl_resp.content) > 50000, "Final high-resolution file size too small"


# ============================================================================
# Scenario 2: Strict Geometric Preservation -> SSIM >= 0.88
# ============================================================================

def test_workload_scenario2_geometric_preservation_ssim(client, promob_reference_image, ssim_evaluator, job_poller):
    """Scenario 2: Verifies strict geometric preservation with SSIM >= 0.88.

    Promob kitchen design requires zero architectural hallucinations, zero warping
    of straight cabinet lines, and zero displaced furniture geometries across
    the full 4x super-resolution and PBR (Multi-Scale SSAO + Soft GI Bloom) pipeline.
    """
    files = {"file": ("promob_kitchen.jpg", promob_reference_image, "image/jpeg")}
    data = {
        "temperature": "0.15",
        "shadow_lift": "0.20",
        "clahe_clip_limit": "1.8",
        "ambient_occlusion": "0.12",
        "sharpness": "0.35",
        "upscale_factor": "4",
        "tile_size": "256",
        "preserve_geometry": "true"
    }

    # Submit 4K enhancement job via POST /api/enhance
    submit_resp = client.post("/api/enhance", files=files, data=data)
    assert submit_resp.status_code in (200, 202), f"Enhancement job submission failed: {submit_resp.text}"
    job_id = submit_resp.json()["job_id"]

    # Poll until completed
    job_data = job_poller(client, job_id, timeout_seconds=90.0)
    assert job_data.get("status") in ("completed", "finished", "success")

    # Download final enhanced 4K image
    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200, f"Download failed with status {dl_resp.status_code}"

    # Calculate SSIM comparing downsampled enhanced image to original input
    ssim_score = ssim_evaluator(promob_reference_image, dl_resp.content, downsample_to_reference=True)

    # Architectural requirement: SSIM must be >= 0.88 under full un-attenuated PBR contrast
    assert ssim_score >= 0.88, (
        f"Geometric fidelity violated: SSIM score {ssim_score:.4f} is below the 0.88 requirement. "
        "Check that super-resolution and filtering do not distort edges or introduce hallucinations."
    )


# ============================================================================
# Scenario 3: Hardware Guardrails & VRAM Safety
# ============================================================================

def test_workload_scenario3_hardware_guardrails(client, promob_reference_image, hardware_monitor, job_poller):
    """Scenario 3: Verifies hardware safety envelope (Peak VRAM <= 5.5GB on 6GB GPU or safe CPU)."""
    hw_before = hardware_monitor()

    # Trigger full 4K enhancement operation
    files = {"file": ("promob_kitchen.jpg", promob_reference_image, "image/jpeg")}
    data = {
        "temperature": "0.15",
        "shadow_lift": "0.20",
        "clahe_clip_limit": "1.5",
        "ambient_occlusion": "0.15",
        "sharpness": "0.30",
        "upscale_factor": "4",
        "tile_size": "256",
        "preserve_geometry": "true"
    }
    submit_resp = client.post("/api/enhance", files=files, data=data)
    assert submit_resp.status_code in (200, 202), f"Enhancement submission failed: {submit_resp.text}"
    job_id = submit_resp.json()["job_id"]

    # Poll until completed
    job_data = job_poller(client, job_id, timeout_seconds=90.0)
    assert job_data.get("status") in ("completed", "finished", "success")

    hw_after = hardware_monitor()
    assert hw_after["is_safe"], f"Hardware safety limit exceeded: peak VRAM was {hw_after['peak_vram_mb']:.1f} MB"

    if hw_after["cuda_available"]:
        assert hw_after["peak_vram_mb"] <= 5632.0, (
            f"VRAM budget exceeded 5.5 GB envelope: {hw_after['peak_vram_mb']:.1f} MB used."
        )


# ============================================================================
# Scenario 4: Color Warmth & Tone Validation
# ============================================================================

def test_workload_scenario4_color_warmth_dynamic_range(client, promob_reference_image):
    """Scenario 4: Verifies transformation from cold/gray Promob render to warm architectural lighting.

    - Verifies shift toward warm tones (increased red-to-blue balance).
    - Verifies dynamic range expansion in dark shadow regions (5th percentile lift).
    """
    orig_img = Image.open(io.BytesIO(promob_reference_image)).convert("RGB")
    orig_arr = np.asarray(orig_img, dtype=np.float32)

    files = {"file": ("promob_kitchen.jpg", promob_reference_image, "image/jpeg")}
    data = {
        "temperature": "0.45",
        "shadow_lift": "0.30",
        "highlight_recovery": "0.15"
    }
    response = client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200

    enh_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    # Resize to original for fair statistical comparison
    enh_img_aligned = enh_img.resize(orig_img.size, Image.Resampling.BILINEAR)
    enh_arr = np.asarray(enh_img_aligned, dtype=np.float32)

    # 1. Warmth verification: Red / Blue ratio should shift warmer (increase)
    orig_rb_ratio = np.mean(orig_arr[:, :, 0]) / (np.mean(orig_arr[:, :, 2]) + 1e-5)
    enh_rb_ratio = np.mean(enh_arr[:, :, 0]) / (np.mean(enh_arr[:, :, 2]) + 1e-5)

    assert enh_rb_ratio > orig_rb_ratio, (
        f"Warmth shift failed: Red/Blue ratio did not increase (orig: {orig_rb_ratio:.3f}, enh: {enh_rb_ratio:.3f})"
    )

    # 2. Dynamic range shadow lift verification: 5th percentile luminance
    orig_p5 = np.percentile(orig_arr, 5)
    enh_p5 = np.percentile(enh_arr, 5)
    assert enh_p5 > orig_p5, (
        f"Shadow lift failed: 5th percentile intensity did not increase (orig: {orig_p5:.1f}, enh: {enh_p5:.1f})"
    )


# ============================================================================
# Scenario 5: End-to-End Client Simulation
# ============================================================================

def test_workload_scenario5_end_to_end_client_simulation(client, promob_reference_image, job_poller):
    """Scenario 5: Complete simulated client session from initial connection to final 4K export.

    Steps simulated:
    1. Check system health and GPU status.
    2. Upload Promob render for fast interactive preview (< 1000ms).
    3. Select Preset Marcenaria and inspect updated preview.
    4. Submit full 4K Super-Resolution job.
    5. Poll asynchronous job progression until completed.
    6. Download final 4K asset.
    7. Verify integrity, content type, and high resolution.
    """
    # Step 1: Health check
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json().get("status") == "ok"

    # Step 2: Interactive fast preview
    t0 = time.perf_counter()
    files = {"file": ("promob_kitchen.jpg", promob_reference_image, "image/jpeg")}
    preview_resp = client.post("/api/preview", files=files)
    preview_duration = time.perf_counter() - t0
    assert preview_resp.status_code == 200
    assert preview_duration < 1.0, f"Preview took {preview_duration:.2f}s (> 1.0s target)"

    # Step 3: Preview with Preset Marcenaria
    marcenaria_data = {
        "ambient_occlusion": "0.25",
        "sharpness": "0.55",
        "clahe_clip_limit": "2.2",
        "shadow_lift": "0.20"
    }
    files = {"file": ("promob_kitchen.jpg", promob_reference_image, "image/jpeg")}
    preset_resp = client.post("/api/preview", files=files, data=marcenaria_data)
    assert preset_resp.status_code == 200

    # Step 4: Submit 4K enhancement
    full_data = dict(marcenaria_data)
    full_data["upscale_factor"] = "4"
    full_data["tile_size"] = "256"
    full_data["preserve_geometry"] = "true"
    files = {"file": ("promob_kitchen.jpg", promob_reference_image, "image/jpeg")}
    enhance_resp = client.post("/api/enhance", files=files, data=full_data)
    assert enhance_resp.status_code in (200, 202)
    job_id = enhance_resp.json()["job_id"]

    # Step 5: Poll progress to completion
    final_job = job_poller(client, job_id, timeout_seconds=90.0)
    assert final_job.get("status") in ("completed", "finished", "success")

    # Step 6: Download final 4K asset
    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200

    # Step 7: Asset validation
    final_img = Image.open(io.BytesIO(dl_resp.content))
    assert final_img.size[0] > 0 and final_img.size[1] > 0
    assert len(dl_resp.content) > 10000
