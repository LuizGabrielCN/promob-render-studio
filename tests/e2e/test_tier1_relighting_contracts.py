"""Tier 1: Opaque-Box Contracts for F1-F14 & R1-R4 (Relighting, Preservation, Offline & App Flow).

Includes:
- CPU-only self-validation of metrics_helpers.py against ChatGPT reference pair and synthetic ablations
- F1/F2 CLI & offline socket-audit contracts (CPU)
- F11/F12/F13/F14 API/UI/classic contracts (CPU)
- F2-F11 isolated GPU relighting, fireplace, spot, smooth LED, and OCR/TV preservation contracts (@pytest.mark.gpu)
"""

from __future__ import annotations

import io
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Dict

import cv2
import numpy as np
import pytest
from PIL import Image

from tests.e2e.conftest import (
    LED_CHANNEL_ROIS,
    REPO_ROOT,
    SPOT_FIREPLACE_ROIS,
    TEXT_TV_ROIS,
)
from tests.e2e.metrics_helpers import (
    R1_THRESHOLDS,
    R2_THRESHOLDS,
    align_to_reference,
    compute_fireplace_metrics,
    compute_geometry_preservation,
    compute_led_continuity_metrics,
    compute_relighting_metrics,
    compute_roi_preservation,
    compute_spot_cone_metrics,
    decode_bgr,
)


def _run_enhance_png(
    client,
    job_poller,
    image_bytes: bytes,
    filename: str,
    data: Dict[str, str],
    timeout: float = 120.0,
) -> bytes:
    mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    resp = client.post("/api/enhance", files={"file": (filename, image_bytes, mime)}, data=data)
    assert resp.status_code in (200, 202), f"/api/enhance failed ({resp.status_code}): {resp.text}"
    job_id = resp.json()["job_id"]
    job_data = job_poller(client, job_id, timeout_seconds=timeout)
    assert job_data.get("status") in ("completed", "finished", "success")
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200, f"/api/download/{job_id} failed ({dl.status_code}): {dl.text}"
    return dl.content


def _require_gpu_timeout(client) -> float:
    caps = client.get("/api/capabilities").json()
    ge = caps.get("generative_engine", {})
    assert ge.get("available") is True, f"Generative engine unavailable: {caps}"
    return 600.0 if not ge.get("cached_locally") else 300.0


# ============================================================================
# Group A: CPU-Only Self-Validation of Opaque-Box Metrics Helpers
# ============================================================================

def test_t1_r1_r2_metrics_pass_on_chatgpt_reference_pair(chatgpt_reference_pair):
    """Calibrated R1 and R2 metrics must pass all gates on the ChatGPT before/after reference pair."""
    before = chatgpt_reference_pair["before"]
    after = chatgpt_reference_pair["after"]
    rois = chatgpt_reference_pair["rois"]

    relit = compute_relighting_metrics(before, after, exclude_rois=[rois["fireplace_firebox"]])
    assert relit["d_r_minus_b"] >= R1_THRESHOLDS["d_r_minus_b_min"], relit
    assert relit["db_mean"] >= R1_THRESHOLDS["db_mean_min"], relit
    assert relit["dL_low_std"] >= R1_THRESHOLDS["dL_low_std_min"], relit
    assert relit["dL_low_spread"] >= R1_THRESHOLDS["dL_low_spread_min"], relit

    spots = compute_spot_cone_metrics(before, after, spot_roi=rois["walnut_upper_spots"])
    assert spots["spot_dL_spread"] >= R1_THRESHOLDS["spot_dL_spread_min"], spots
    assert spots["spot_d_r_minus_b"] >= R1_THRESHOLDS["spot_d_r_minus_b_min"], spots

    fp = compute_fireplace_metrics(
        before,
        after,
        firebox_roi=rois["fireplace_firebox"],
        spill_roi=rois["fireplace_spill_floor"],
    )
    assert fp["fp_L_rng_ratio"] >= R1_THRESHOLDS["fp_L_rng_ratio_min"], fp
    assert fp["fp_dchr"] >= R1_THRESHOLDS["fp_dchr_min"], fp
    assert fp["spill_dL"] >= R1_THRESHOLDS["fp_spill_dL_min"], fp
    assert fp["spill_d_r_minus_b"] >= R1_THRESHOLDS["fp_spill_d_r_minus_b_min"], fp

    geom = compute_geometry_preservation(before, after, exclude_rois=[rois["fireplace_firebox"]])
    assert geom["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"], geom
    assert geom["canny_f1_d2"] >= R2_THRESHOLDS["canny_f1_d2_min"], geom
    assert geom["global_ssim_L"] >= R2_THRESHOLDS["global_ssim_L_min"], geom
    assert geom["dE_ab_norm"] <= R2_THRESHOLDS["dE_ab_norm_max"], geom

    tv = compute_roi_preservation(before, after, roi=rois["tv_screen"], expected_text="Elleganza")
    assert tv["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], tv
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv
    assert tv["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv


def test_t1_r1_metrics_autodownsample_4x_reference_equivalence(chatgpt_reference_pair):
    """4x upscaled outputs are auto-downsampled to reference resolution with consistent R1/R2 scores."""
    before = chatgpt_reference_pair["before"]
    after_bgr = decode_bgr(chatgpt_reference_pair["after"])
    h, w = after_bgr.shape[:2]
    after_4x = cv2.resize(after_bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    rois = chatgpt_reference_pair["rois"]

    native_geom = compute_geometry_preservation(before, after_bgr, exclude_rois=[rois["fireplace_firebox"]])
    upscaled_geom = compute_geometry_preservation(before, after_4x, exclude_rois=[rois["fireplace_firebox"]])
    assert abs(native_geom["sobel_corr"] - upscaled_geom["sobel_corr"]) < 0.03
    assert abs(native_geom["global_ssim_L"] - upscaled_geom["global_ssim_L"]) < 0.03
    assert upscaled_geom["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"]


def test_t1_r1_metrics_reject_noop_self_comparison(chatgpt_reference_pair):
    """No-op (before vs before) must fail R1 global warmth and spatial lighting spread gates."""
    before = chatgpt_reference_pair["before"]
    rois = chatgpt_reference_pair["rois"]
    relit = compute_relighting_metrics(before, before, exclude_rois=[rois["fireplace_firebox"]])
    assert relit["d_r_minus_b"] == pytest.approx(0.0, abs=1e-4)
    assert relit["dL_low_spread"] == pytest.approx(0.0, abs=1e-4)
    assert relit["d_r_minus_b"] < R1_THRESHOLDS["d_r_minus_b_min"]
    assert relit["dL_low_spread"] < R1_THRESHOLDS["dL_low_spread_min"]


def test_t1_r1_metrics_reject_flat_warm_tint(chatgpt_reference_pair):
    """Flat global warm tint passes warmth shift but must fail spatial lighting spread & spot-cone gates."""
    before_bgr = decode_bgr(chatgpt_reference_pair["before"])
    rois = chatgpt_reference_pair["rois"]
    flat_warm = before_bgr.astype(np.int16)
    flat_warm[:, :, 2] = np.clip(flat_warm[:, :, 2] + 18, 0, 255)
    flat_warm[:, :, 1] = np.clip(flat_warm[:, :, 1] + 5, 0, 255)
    flat_warm[:, :, 0] = np.clip(flat_warm[:, :, 0] - 15, 0, 255)
    flat_warm_u8 = flat_warm.astype(np.uint8)

    relit = compute_relighting_metrics(before_bgr, flat_warm_u8, exclude_rois=[rois["fireplace_firebox"]])
    spots = compute_spot_cone_metrics(before_bgr, flat_warm_u8, spot_roi=rois["walnut_upper_spots"])
    assert relit["d_r_minus_b"] >= R1_THRESHOLDS["d_r_minus_b_min"]
    assert relit["dL_low_std"] < R1_THRESHOLDS["dL_low_std_min"], relit
    assert relit["dL_low_spread"] < R1_THRESHOLDS["dL_low_spread_min"], relit
    assert spots["spot_dL_spread"] < R1_THRESHOLDS["spot_dL_spread_min"], spots


def test_t1_r1_led_metrics_accept_smooth_and_reject_beaded(synthetic_led_channel_png):
    """LED continuity metrics pass smooth warm concealed strips and reject No-Op and beaded segments."""
    orig_bgr = decode_bgr(synthetic_led_channel_png)

    # No-op keeps saturated yellow marker
    noop_m = compute_led_continuity_metrics(orig_bgr, orig_bgr)
    assert noop_m["yellow_rem"] > 0.90

    # Smooth recessed warm-white LED + warm halo spill
    smooth = orig_bgr.copy().astype(np.float32)
    smooth[169:175, 80:432] = (225.0, 244.0, 255.0)
    smooth[175:188, 80:432] += np.array([-4.0, 6.0, 16.0], dtype=np.float32)
    smooth[156:169, 80:432] += np.array([-4.0, 6.0, 16.0], dtype=np.float32)
    smooth_u8 = np.clip(smooth, 0, 255).astype(np.uint8)
    smooth_m = compute_led_continuity_metrics(orig_bgr, smooth_u8)
    assert smooth_m["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], smooth_m
    assert smooth_m["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], smooth_m
    assert smooth_m["gap_frac"] <= R1_THRESHOLDS["led_gap_frac_max"], smooth_m
    assert smooth_m["spill_dL"] >= R1_THRESHOLDS["led_spill_dL_min"], smooth_m
    assert smooth_m["spill_db"] >= R1_THRESHOLDS["led_spill_db_min"], smooth_m

    # Beaded/segmented LED artifact (periodic dark gaps every 16px)
    beaded = smooth_u8.copy()
    for x in range(80, 432, 16):
        beaded[169:175, x:x + 7] = (65, 65, 70)
    beaded_m = compute_led_continuity_metrics(orig_bgr, beaded)
    assert beaded_m["center_cv"] > R1_THRESHOLDS["led_center_cv_max"], beaded_m
    assert beaded_m["gap_frac"] > R1_THRESHOLDS["led_gap_frac_max"], beaded_m


def test_t1_r2_geometry_metrics_reject_warped_input(chatgpt_reference_pair):
    """2% geometric scale + 6px shift must fail Sobel correlation and Canny F1 gates."""
    before_bgr = decode_bgr(chatgpt_reference_pair["before"])
    after_bgr = decode_bgr(chatgpt_reference_pair["after"])
    h, w = after_bgr.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, 1.02)
    mat[0, 2] += 6.0
    mat[1, 2] += 4.0
    warped = cv2.warpAffine(after_bgr, mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    geom = compute_geometry_preservation(
        before_bgr,
        warped,
        exclude_rois=[chatgpt_reference_pair["rois"]["fireplace_firebox"]],
    )
    assert geom["sobel_corr"] < R2_THRESHOLDS["sobel_corr_min"], geom
    assert geom["canny_f1_d2"] < R2_THRESHOLDS["canny_f1_d2_min"], geom


def test_t1_r2_roi_metrics_reject_blurred_shifted_noisy_tv_text(chatgpt_reference_pair):
    """ROI preservation metrics catch blurred (EdgeF1), shifted (ZNCC), and noisy (SSIM) TV text corruptions."""
    before_bgr = decode_bgr(chatgpt_reference_pair["before"])
    after_bgr = decode_bgr(chatgpt_reference_pair["after"])
    tv_roi = chatgpt_reference_pair["rois"]["tv_screen"]

    blurred = cv2.GaussianBlur(after_bgr, (7, 7), 2.5)
    m_blur = compute_roi_preservation(before_bgr, blurred, roi=tv_roi)
    assert m_blur["edge_f1_d1"] < R2_THRESHOLDS["roi_edge_f1_d1_min"], m_blur

    shifted = np.roll(after_bgr, shift=4, axis=1)
    m_shift = compute_roi_preservation(before_bgr, shifted, roi=tv_roi)
    assert m_shift["zncc"] < R2_THRESHOLDS["roi_zncc_min"], m_shift

    rng = np.random.default_rng(42)
    noisy = np.clip(after_bgr.astype(np.int16) + rng.normal(0, 35, after_bgr.shape).astype(np.int16), 0, 255).astype(np.uint8)
    m_noise = compute_roi_preservation(before_bgr, noisy, roi=tv_roi)
    assert m_noise["ssim_L"] < R2_THRESHOLDS["roi_ssim_L_min"], m_noise


def test_t1_r2_synthetic_roi_metrics_pass_on_tv_and_sign(synthetic_text_tv_scene_png):
    """Synthetic TV/sign ROIs pass all preservation metrics under mild warm luminance modulation."""
    orig_bgr = decode_bgr(synthetic_text_tv_scene_png)
    relit = np.clip(orig_bgr.astype(np.float32) * np.array([0.96, 1.0, 1.05]), 0, 255).astype(np.uint8)
    for roi_name, expected in (("tv_screen", "PROMOB"), ("framed_sign", "WOLF")):
        m = compute_roi_preservation(
            orig_bgr, relit, roi=TEXT_TV_ROIS[roi_name], expected_text=expected
        )
        assert m["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], (roi_name, m)
        assert m["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], (roi_name, m)
        assert m["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], (roi_name, m)
        if m["ocr_available"]:
            assert m["ocr_matched"] is True, (roi_name, m)


# ============================================================================
# Group B: F1 Model Provisioning & Offline Socket-Audit Contracts (CPU)
# ============================================================================

def test_t1_f1_download_models_cli_help():
    """backend/scripts/download_models.py --help exits 0 and documents --required/--optional."""
    proc = subprocess.run(
        [sys.executable, "backend/scripts/download_models.py", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--required" in proc.stdout and "--optional" in proc.stdout


def test_t1_f1_capabilities_reports_local_cache_status(client):
    """GET /api/capabilities reports generative_engine available & cached_locally booleans."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    ge = resp.json().get("generative_engine", {})
    assert isinstance(ge.get("available"), bool)
    assert isinstance(ge.get("cached_locally"), bool)


@pytest.mark.offline
def test_t1_f1_offline_audited_server_health_and_capabilities(offline_audited_server):
    """Offline audited server serves /api/health and /api/capabilities with zero outbound TCP calls."""
    import httpx

    with httpx.Client(base_url=offline_audited_server.base_url, timeout=30.0) as http:
        h = http.get("/api/health")
        c = http.get("/api/capabilities")
    assert h.status_code == 200 and h.json().get("status") == "ok"
    assert c.status_code == 200 and "generative_engine" in c.json()
    assert offline_audited_server.get_outbound_attempts() == []


@pytest.mark.offline
def test_t1_f1_offline_audited_server_preview_zero_network(offline_audited_server, synthetic_jpeg):
    """POST /api/preview under offline socket audit executes with zero outbound connections."""
    import httpx

    with httpx.Client(base_url=offline_audited_server.base_url, timeout=30.0) as http:
        resp = http.post(
            "/api/preview",
            files={"file": ("room.jpg", synthetic_jpeg, "image/jpeg")},
            data={"render_mode": "generative", "scene_profile": "auto"},
        )
    assert resp.status_code == 200
    assert Image.open(io.BytesIO(resp.content)).size[0] > 0
    assert offline_audited_server.get_outbound_attempts() == []


@pytest.mark.offline
def test_t1_f1_offline_audited_server_strength_zero_enhance_zero_network(
    offline_audited_server, synthetic_png, job_poller
):
    """POST /api/enhance (strength=0.0, upscale=1) completes cleanly offline with zero network traffic."""
    import httpx

    with httpx.Client(base_url=offline_audited_server.base_url, timeout=60.0) as http:
        resp = http.post(
            "/api/enhance",
            files={"file": ("room.png", synthetic_png, "image/png")},
            data={"render_mode": "generative", "generative_strength": "0.0", "upscale_factor": "1"},
        )
        assert resp.status_code in (200, 202)
        job_poller(http, resp.json()["job_id"], timeout_seconds=45.0)
    assert offline_audited_server.get_outbound_attempts() == []


# ============================================================================
# Group C: F2, F5, F11, F12, F13, F14 CPU API/CLI/Browser Contracts
# ============================================================================

def test_t1_f2_bench_realismo_cli_help():
    """backend/scripts/bench_realismo.py --help exits 0 and documents --tag, --subset, --strengths."""
    proc = subprocess.run(
        [sys.executable, "backend/scripts/bench_realismo.py", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--tag", "--subset", "--strengths"):
        assert flag in proc.stdout, f"Missing {flag} in bench_realismo --help"


def test_t1_f2_enhance_job_reports_processing_time_and_dimensions(client, synthetic_png, job_poller):
    """Completed job JSON includes processing_time_seconds >= 0 and exact output dimensions."""
    resp = client.post(
        "/api/enhance",
        files={"file": ("dim.png", synthetic_png, "image/png")},
        data={"render_mode": "classic", "upscale_factor": "1"},
    )
    assert resp.status_code in (200, 202)
    job = job_poller(client, resp.json()["job_id"], timeout_seconds=45.0)
    assert float(job.get("processing_time_seconds", -1.0)) >= 0.0
    assert job.get("output_width") == 256 and job.get("output_height") == 256


def test_t1_f11_photo_finish_zero_vs_positive_classic_preview(client, synthetic_png):
    """photo_finish=0.0 and 0.5 execute cleanly on /api/preview without clipping or NaN."""
    for pf in ("0.0", "0.5"):
        resp = client.post(
            "/api/preview",
            files={"file": ("pf.png", synthetic_png, "image/png")},
            data={"render_mode": "classic", "photo_finish": pf},
        )
        assert resp.status_code == 200
        arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
        assert not np.isnan(arr).any() and arr.size > 0


def test_t1_f12_enhance_accepts_new_form_flags(client, synthetic_png, job_poller):
    """POST /api/enhance accepts led_strips, photo_finish, latent_object_lock, freeu, and prompt_style."""
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_png,
        "flags.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.0",
            "upscale_factor": "1",
            "led_strips": "true",
            "photo_finish": "0.5",
            "latent_object_lock": "true",
            "freeu": "false",
            "prompt_style": "photo",
        },
        timeout=45.0,
    )
    assert Image.open(io.BytesIO(out)).size == (256, 256)


def test_t1_f12_presets_schema_includes_all_five_presets(client):
    """GET /api/presets returns all 5 named presets with valid name, description, and config dicts."""
    presets = client.get("/api/presets").json()
    for key in ("realismo_ia", "natural", "marcenaria", "luz_quente", "ultra_detalhe"):
        assert key in presets
        entry = presets[key]
        assert isinstance(entry.get("name"), str) and entry["name"].strip()
        assert isinstance(entry.get("description"), str) and entry["description"].strip()
        assert isinstance(entry.get("config"), dict)


def test_t1_f13_frontend_typecheck_passes():
    """TypeScript strict check (tsc --noEmit) on frontend/tsconfig.json exits 0."""
    tsc_bin = REPO_ROOT / "frontend" / "node_modules" / ".bin" / ("tsc.cmd" if os.name == "nt" else "tsc")
    if not tsc_bin.is_file():
        pytest.skip(f"TypeScript binary not installed at {tsc_bin}")
    proc = subprocess.run(
        [str(tsc_bin), "-p", "frontend/tsconfig.json", "--noEmit"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=90.0,
        shell=(os.name == "nt"),
        check=False,
    )
    assert proc.returncode == 0, f"tsc failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.browser
def test_t1_f13_browser_viewer_zoom_and_spacebar_controls(cdp_browser):
    """CDP browser verifies Viewer split/side-by-side mode, zoom controls, Spacebar hold, and Reiniciar."""
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    cdp_browser.click_text("Carregar Amostra Promob")
    cdp_browser.wait_for_text("Realismo IA", timeout=15.0)
    cdp_browser.wait_for_condition(
        "Array.from(document.querySelectorAll('img')).some(i => i.complete && i.naturalWidth > 0)",
        timeout=15.0,
    )

    # Test real zoom controls in StatusBar
    cdp_browser.wait_for_text("100%", timeout=10.0)
    cdp_browser.evaluate("document.querySelector('button[title=\"Aumentar Zoom\"]').click()")
    cdp_browser.wait_for_text("125%", timeout=5.0)
    cdp_browser.evaluate("document.querySelector('button[title=\"Reduzir Zoom\"]').click()")
    cdp_browser.wait_for_text("100%", timeout=5.0)

    # Ensure no INPUT/TEXTAREA is focused and verify Spacebar hold toggles original comparison toast
    cdp_browser.evaluate("document.activeElement && document.activeElement.blur && document.activeElement.blur()")
    cdp_browser._send(
        "Input.dispatchKeyEvent",
        {"type": "rawKeyDown", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32},
    )
    cdp_browser.wait_for_text("Exibindo render original [Espaço mantido]", timeout=5.0)
    cdp_browser._send(
        "Input.dispatchKeyEvent",
        {"type": "keyUp", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32},
    )
    cdp_browser.wait_for_condition(
        "!(document.body && document.body.innerText || '').includes('Exibindo render original [Espaço mantido]')",
        timeout=5.0,
    )

    body = cdp_browser.evaluate("document.body.innerText") or ""
    if "Lado a Lado" in body:
        cdp_browser.click_text("Lado a Lado")
        cdp_browser.pump_events(0.15)
    if "Divisor" in body:
        cdp_browser.click_text("Divisor")
        cdp_browser.pump_events(0.15)

    cdp_browser.click_text("Reiniciar")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=10.0)


def test_t1_f14_classic_temperature_and_shadow_lift_preview(client, synthetic_1080p_interior):
    """Classic photometric preview warms R/B under +temperature and lifts 5th-percentile darks."""
    base_resp = client.post(
        "/api/preview",
        files={"file": ("in.jpg", synthetic_1080p_interior, "image/jpeg")},
        data={"render_mode": "classic", "temperature": "0.0", "shadow_lift": "0.0"},
    )
    warm_resp = client.post(
        "/api/preview",
        files={"file": ("in.jpg", synthetic_1080p_interior, "image/jpeg")},
        data={"render_mode": "classic", "temperature": "0.6", "shadow_lift": "0.5"},
    )
    assert base_resp.status_code == 200 and warm_resp.status_code == 200
    b_arr = np.asarray(Image.open(io.BytesIO(base_resp.content)).convert("RGB"), dtype=np.float32)
    w_arr = np.asarray(Image.open(io.BytesIO(warm_resp.content)).convert("RGB"), dtype=np.float32)
    rb_base = float(np.mean(b_arr[:, :, 0]) - np.mean(b_arr[:, :, 2]))
    rb_warm = float(np.mean(w_arr[:, :, 0]) - np.mean(w_arr[:, :, 2]))
    assert rb_warm > rb_base + 5.0
    lum_base = 0.299 * b_arr[:, :, 0] + 0.587 * b_arr[:, :, 1] + 0.114 * b_arr[:, :, 2]
    lum_warm = 0.299 * w_arr[:, :, 0] + 0.587 * w_arr[:, :, 1] + 0.114 * w_arr[:, :, 2]
    assert np.percentile(lum_warm, 5) > np.percentile(lum_base, 5)


def test_t1_f5_led_strips_false_leaves_yellow_marker_untouched_cpu(
    client, synthetic_led_channel_png, job_poller
):
    """led_strips=false leaves Promob yellow marker strips un-converted in classic /api/enhance."""
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_led_channel_png,
        "led_off.png",
        {
            "render_mode": "classic",
            "led_strips": "false",
            "upscale_factor": "1",
            "temperature": "0.0",
            "tint": "0.0",
        },
        timeout=45.0,
    )
    metrics = compute_led_continuity_metrics(synthetic_led_channel_png, out_png)
    assert metrics["yellow_rem"] >= 0.80, f"Expected yellow strip preserved when led_strips=false: {metrics}"


def test_t1_f5_led_strips_true_classic_preview_valid_output(client, synthetic_led_channel_png):
    """led_strips=true on POST /api/preview replaces yellow markers smoothly without NaN."""
    resp = client.post(
        "/api/preview",
        files={"file": ("led.png", synthetic_led_channel_png, "image/png")},
        data={"render_mode": "classic", "led_strips": "true", "specular_bloom": "0.25"},
    )
    assert resp.status_code == 200
    arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
    assert not np.isnan(arr).any() and arr.size > 0
    led = compute_led_continuity_metrics(synthetic_led_channel_png, resp.content)
    assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], led
    assert led["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], led
    assert led["gap_frac"] <= 0.08, led
    assert led["spill_dL"] >= R1_THRESHOLDS["led_spill_dL_min"], led


# ============================================================================
# Group D: GPU Relighting, Emitters, LEDs & Protection Contracts
# ============================================================================

@pytest.mark.gpu
def test_t1_f2_single_generative_render_within_180s_sla(client, promob_reference_image, job_poller):
    """Single native-res generative render finishes within local latency SLA and reports timing."""
    timeout_s = _require_gpu_timeout(client)
    t0 = time.perf_counter()
    out_png = _run_enhance_png(
        client,
        job_poller,
        promob_reference_image,
        "sla.jpg",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "auto",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - t0
    assert Image.open(io.BytesIO(out_png)).size == Image.open(io.BytesIO(promob_reference_image)).size
    assert elapsed <= 180.0, f"Single native-res generative render exceeded 180s SLA: {elapsed:.2f}s"


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f3_spot_relighting_warms_downward_wall_cone(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """Ceiling spots produce warm downward wall-wash illumination spread on the slatted panel below."""
    timeout_s = _require_gpu_timeout(client)
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "spots.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "scene_profile": "quarto_sala",
            "specular_bloom": "0.25",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    spots = compute_spot_cone_metrics(
        synthetic_spot_fireplace_scene_png,
        out_png,
        spot_roi=SPOT_FIREPLACE_ROIS["spot_wall_wash"],
    )
    assert spots["spot_d_r_minus_b"] > 3.0, f"Spot cone failed to warm wall below spots: {spots}"
    assert spots["spot_dL_spread"] > 1.0, f"Spot cone lacked spatial illumination spread: {spots}"


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f3_auto_profile_adapts_lighting_without_clipping(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """scene_profile='auto' adapts lighting smoothly with zero NaN or full-frame clipping."""
    timeout_s = _require_gpu_timeout(client)
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "auto_profile.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "auto",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    arr = np.asarray(Image.open(io.BytesIO(out_png)).convert("RGB"), dtype=np.float32)
    assert not np.isnan(arr).any()
    assert 5.0 < float(np.mean(arr)) < 250.0
    relit = compute_relighting_metrics(
        synthetic_spot_fireplace_scene_png,
        out_png,
        exclude_rois=[SPOT_FIREPLACE_ROIS["firebox"]],
    )
    assert relit["dL_low_spread"] >= 2.0, relit


@pytest.mark.gpu
def test_t1_f4_tv_screen_and_text_roi_preservation(client, synthetic_text_tv_scene_png, job_poller):
    """TV screen ('PROMOB CABINETRY') and framed sign ('WOLF 2026') preserve edges, ZNCC, and SSIM."""
    timeout_s = _require_gpu_timeout(client)
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "text_tv.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "scene_profile": "quarto_sala",
            "preserve_geometry": "true",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    for roi_name, expected in (("tv_screen", "PROMOB"), ("framed_sign", "WOLF")):
        m = compute_roi_preservation(
            synthetic_text_tv_scene_png,
            out_png,
            roi=TEXT_TV_ROIS[roi_name],
            expected_text=expected,
        )
        assert m["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], (roi_name, m)
        assert m["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], (roi_name, m)
        assert m["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], (roi_name, m)
        if m["ocr_available"]:
            assert m["ocr_matched"] is True, (roi_name, m)


@pytest.mark.gpu
def test_t1_f4_preserve_geometry_true_keeps_global_edges(client, synthetic_text_tv_scene_png, job_poller):
    """preserve_geometry=true retains global Sobel correlation, Canny F1, and luminance SSIM."""
    timeout_s = _require_gpu_timeout(client)
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "geom_lock.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "auto",
            "preserve_geometry": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(synthetic_text_tv_scene_png, out_png)
    assert geom["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"], geom
    assert geom["canny_f1_d2"] >= R2_THRESHOLDS["canny_f1_d2_min"], geom
    assert geom["global_ssim_L"] >= 0.78, geom


@pytest.mark.gpu
def test_t1_f5_smooth_continuous_led_replaces_yellow_marker(client, synthetic_led_channel_png, job_poller):
    """Concealed LED pipeline removes yellow guide pixels and renders continuous smooth warm glow."""
    timeout_s = _require_gpu_timeout(client)
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_led_channel_png,
        "led_gen.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "cozinha",
            "led_strips": "true",
            "specular_bloom": "0.25",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    led = compute_led_continuity_metrics(synthetic_led_channel_png, out_png)
    assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], led
    assert led["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], led
    assert led["gap_frac"] <= R1_THRESHOLDS["led_gap_frac_max"], led
    assert led["spill_dL"] >= 1.0, led


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f6_fireplace_emitter_synthesizes_warm_glow_and_spill(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """Unlit dark fireplace cutout gains warm flame dynamic range/chroma and hearth bounce spill."""
    timeout_s = _require_gpu_timeout(client)
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "fireplace.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "scene_profile": "quarto_sala",
            "specular_bloom": "0.25",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    fp = compute_fireplace_metrics(
        synthetic_spot_fireplace_scene_png,
        out_png,
        firebox_roi=SPOT_FIREPLACE_ROIS["firebox"],
        spill_roi=SPOT_FIREPLACE_ROIS["hearth_spill"],
    )
    assert fp["fp_L_rng_ratio"] >= 1.4, f"Firebox lacked flame luminance range expansion: {fp}"
    assert fp["fp_dchr"] >= 8.0, f"Firebox lacked warm flame chroma: {fp}"
    assert fp["spill_d_r_minus_b"] >= 5.0, f"Hearth floor lacked warm bounce spill: {fp}"


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f7_neural_relighting_shifts_lowfreq_illumination_vs_classic(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """Generative relighting introduces genuine low-frequency lighting dynamics beyond classic mode."""
    timeout_s = _require_gpu_timeout(client)
    classic_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "relit_classic.png",
        {"render_mode": "classic", "upscale_factor": "1"},
        timeout=45.0,
    )
    gen_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "relit_gen.png",
        {"render_mode": "generative", "generative_strength": "0.55", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    relit = compute_relighting_metrics(classic_png, gen_png)
    c_arr, g_arr = align_to_reference(classic_png, gen_png)
    mae = float(np.mean(np.abs(g_arr.astype(np.float32) - c_arr.astype(np.float32))))
    assert mae > 2.0 and relit["dL_low_std"] > 0.8, (mae, relit)


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f8_luminance_modulated_tv_and_unclamped_wall_contrast(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """Protected TV/sign stays crisp while architectural wall remains free to relight."""
    timeout_s = _require_gpu_timeout(client)
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "unclamped_wall.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "scene_profile": "quarto_sala",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    tv = compute_roi_preservation(
        synthetic_spot_fireplace_scene_png,
        out_png,
        roi=SPOT_FIREPLACE_ROIS["tv_sign"],
        expected_text="ELLEGANZA",
    )
    spots = compute_spot_cone_metrics(
        synthetic_spot_fireplace_scene_png,
        out_png,
        spot_roi=SPOT_FIREPLACE_ROIS["spot_wall_wash"],
    )
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv
    assert tv["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv
    assert spots["spot_dL_spread"] > 0.8, spots


@pytest.mark.gpu
def test_t1_f9_neutral_wall_hue_fidelity_no_jaundice(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """Unlit neutral grey wall remains free of excessive yellow/green hue corruption."""
    timeout_s = _require_gpu_timeout(client)
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "hue_fidelity.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "auto",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(
        synthetic_spot_fireplace_scene_png,
        out_png,
        exclude_rois=[SPOT_FIREPLACE_ROIS["firebox"]],
    )
    assert geom["dE_ab_norm"] <= R2_THRESHOLDS["dE_ab_norm_max"], geom


@pytest.mark.gpu
def test_t1_f9_all_scene_profiles_execute_generative_cleanly(client, synthetic_png, job_poller):
    """All 4 scene_profile options execute generative mode cleanly and return valid native-res PNGs."""
    timeout_s = _require_gpu_timeout(client)
    for profile in ("auto", "banheiro", "cozinha", "quarto_sala"):
        out_png = _run_enhance_png(
            client,
            job_poller,
            synthetic_png,
            f"gp_{profile}.png",
            {
                "render_mode": "generative",
                "generative_strength": "0.48",
                "scene_profile": profile,
                "upscale_factor": "1",
            },
            timeout=timeout_s,
        )
        assert Image.open(io.BytesIO(out_png)).size == (256, 256)


@pytest.mark.gpu
def test_t1_f10_single_generative_render_vram_under_4600mb(
    client, promob_reference_image, job_poller, vram_monitor
):
    """Single generative render respects <4600 MB PyTorch / <5632 MB nvidia-smi budget and releases VRAM."""
    timeout_s = _require_gpu_timeout(client)
    pre_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    with vram_monitor() as mon:
        _run_enhance_png(
            client,
            job_poller,
            promob_reference_image,
            "vram_single.jpg",
            {
                "render_mode": "generative",
                "generative_strength": "0.48",
                "scene_profile": "auto",
                "upscale_factor": "1",
            },
            timeout=timeout_s,
        )
    mon.assert_within_budget(max_nvidia_mb=5632.0, max_delta_nvidia_mb=4600.0, max_torch_mb=4600.0)
    post_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    mon.assert_health_released(pre_alloc, post_alloc, tolerance_mb=512.0)


@pytest.mark.gpu
def test_t1_f11_photo_finish_applies_filmic_output_without_clipping(
    client, promob_reference_image, job_poller
):
    """photo_finish=0.5 vs 0.0 in generative mode produces distinct non-clipped PNG output."""
    timeout_s = _require_gpu_timeout(client)
    base_params = {
        "render_mode": "generative",
        "generative_strength": "0.48",
        "scene_profile": "auto",
        "upscale_factor": "1",
    }
    pf0 = _run_enhance_png(
        client, job_poller, promob_reference_image, "pf0.jpg", {**base_params, "photo_finish": "0.0"}, timeout=timeout_s
    )
    pf5 = _run_enhance_png(
        client, job_poller, promob_reference_image, "pf5.jpg", {**base_params, "photo_finish": "0.5"}, timeout=timeout_s
    )
    a0 = np.asarray(Image.open(io.BytesIO(pf0)).convert("RGB"), dtype=np.float32)
    a5 = np.asarray(Image.open(io.BytesIO(pf5)).convert("RGB"), dtype=np.float32)
    assert float(np.mean(np.abs(a5 - a0))) > 0.2
    assert 5.0 < float(np.mean(a5)) < 250.0


# ============================================================================
# Group E: Additional Tier 1 Contracts Closing F1-F14 (>=5 per Feature)
# ============================================================================

@pytest.mark.offline
def test_t1_f1_download_models_verify_only_succeeds_offline():
    """F1: download_models.py --verify-only verifies local HF cache offline without network."""
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "backend/scripts/download_models.py", "--verify-only"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )
    assert proc.returncode == 0, f"verify-only failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    assert "ALL VERIFIED IN LOCAL CACHE" in proc.stdout


def test_t1_f2_bench_relight_candidates_cli_help():
    """F2: bench_relight_candidates.py --help exits 0 and documents benchmark flags."""
    proc = subprocess.run(
        [sys.executable, "backend/scripts/bench_relight_candidates.py", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--tag", "--check-flux2", "--working-dim", "--steps", "--cfg", "--strength"):
        assert flag in proc.stdout, f"Missing {flag} in bench_relight_candidates --help"


def test_t1_f2_m1_benchmark_summary_records_rtx2060_feasibility():
    """F2: M1 empirical RTX 2060 summary.json records IC-Light VRAM (<4608 MB) and latency feasibility."""
    summary_path = REPO_ROOT / "comparativos_v3" / "m1_iclight_ab" / "summary.json"
    if not summary_path.is_file():
        pytest.skip(f"M1 summary artifact not present at {summary_path}")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(rows, list) and len(rows) >= 2
    ic_rows = [r for r in rows if str(r.get("candidate", "")).startswith("iclight_")]
    assert ic_rows, f"Expected IC-Light benchmark rows in {summary_path}"
    for r in ic_rows:
        assert float(r["vram_peak_mb"]) <= 4608.0, r
        assert float(r["nvsmi_peak_mb"]) < 5734.0, r
        assert float(r["t_total_s"]) <= 180.0, r
    assert any(r.get("status") == "PASS" for r in ic_rows), ic_rows


def test_t1_f2_job_status_exposes_progress_stage_and_result_url(client, synthetic_png, job_poller):
    """F2: Completed job telemetry exposes progress=100, non-empty stage, processing_time, and result_url."""
    resp = client.post(
        "/api/enhance",
        files={"file": ("telemetry.png", synthetic_png, "image/png")},
        data={"render_mode": "classic", "upscale_factor": "1"},
    )
    assert resp.status_code in (200, 202)
    job_id = resp.json()["job_id"]
    job = job_poller(client, job_id, timeout_seconds=45.0)
    assert float(job.get("progress", -1.0)) == pytest.approx(100.0)
    assert isinstance(job.get("stage"), str) and len(job["stage"].strip()) > 0
    assert float(job.get("processing_time_seconds", -1.0)) >= 0.0
    assert job.get("result_url") == f"/api/download/{job_id}"


def test_t1_f4_preserve_geometry_contract_classic_enhance_cpu(client, synthetic_text_tv_scene_png, job_poller):
    """F4: preserve_geometry=true on strength=0.0 enhance retains Sobel >= 0.90 and TV/sign OCR."""
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "f4_cpu.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.0",
            "preserve_geometry": "true",
            "upscale_factor": "1",
        },
        timeout=45.0,
    )
    geom = compute_geometry_preservation(synthetic_text_tv_scene_png, out)
    assert geom["sobel_corr"] >= 0.90, geom
    tv = compute_roi_preservation(
        synthetic_text_tv_scene_png, out, roi=TEXT_TV_ROIS["tv_screen"], expected_text="PROMOB"
    )
    assert tv["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv


def test_t1_f5_led_strips_true_classic_enhance_continuity_cpu(
    client, synthetic_led_channel_png, job_poller
):
    """F5: Classic /api/enhance with led_strips=true replaces yellow marker smoothly with warm spill."""
    out_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_led_channel_png,
        "led_on.png",
        {
            "render_mode": "classic",
            "led_strips": "true",
            "specular_bloom": "0.25",
            "upscale_factor": "1",
        },
        timeout=45.0,
    )
    led = compute_led_continuity_metrics(synthetic_led_channel_png, out_png)
    assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], led
    assert led["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], led
    assert led["gap_frac"] <= 0.08, led
    assert led["spill_dL"] >= R1_THRESHOLDS["led_spill_dL_min"], led


def test_t1_f5_diagonal_sloped_led_strip_smooth_classic_cpu(client, job_poller):
    """F5: Diagonal yellow LED marker under a sloped ceiling renders smoothly without gaps."""
    bgr = np.full((384, 512, 3), 182, dtype=np.uint8)
    cv2.line(bgr, (60, 120), (450, 260), (0, 238, 255), thickness=6)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    raw = enc.tobytes()

    out_png = _run_enhance_png(
        client,
        job_poller,
        raw,
        "diag_led.png",
        {"render_mode": "classic", "led_strips": "true", "upscale_factor": "1"},
        timeout=45.0,
    )
    led = compute_led_continuity_metrics(raw, out_png)
    assert led["yellow_px"] >= 500.0, led
    assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], led
    assert led["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], led
    assert led["gap_frac"] <= 0.15, led
    assert led["spill_dL"] >= R1_THRESHOLDS["led_spill_dL_min"], led


def test_t1_f10_capabilities_reports_hardware_vram_telemetry(client):
    """F10: GET /api/capabilities exposes hardware telemetry and recommended execution strategy."""
    caps = client.get("/api/capabilities").json()
    hw = caps.get("hardware", {})
    assert isinstance(hw, dict)
    assert isinstance(hw.get("cuda_available"), bool)
    assert float(hw.get("total_mb", -1.0)) >= 0.0, hw
    assert float(hw.get("free_mb", -1.0)) >= 0.0, hw


def test_t1_f10_classic_4x_realesrgan_vram_released_after_job(client, synthetic_png, job_poller, vram_monitor):
    """F10: Classic 4x Real-ESRGAN ONNX upscale stays within VRAM budget and releases memory post-job."""
    pre_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    with vram_monitor() as mon:
        out = _run_enhance_png(
            client,
            job_poller,
            synthetic_png,
            "sr4x_vram.png",
            {"render_mode": "classic", "upscale_factor": "4", "tile_size": "256"},
            timeout=90.0,
        )
    assert Image.open(io.BytesIO(out)).size == (1024, 1024)
    if mon.nvidia_available:
        assert mon.peak_nvidia_mb < 5632.0
    post_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    mon.assert_health_released(pre_alloc, post_alloc, tolerance_mb=512.0)


def test_t1_f11_classic_4x_upscale_exact_dimensions_and_png_format(client, synthetic_png, job_poller):
    """F11: upscale_factor=4 on 256x256 PNG produces exact 1024x1024 lossless PNG."""
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_png,
        "up4x.png",
        {"render_mode": "classic", "upscale_factor": "4", "tile_size": "256"},
        timeout=90.0,
    )
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.size == (1024, 1024)


@pytest.mark.browser
def test_t1_f13_browser_header_vram_telemetry_badge(cdp_browser):
    """F13: Header GPU/VRAM telemetry badge renders live hardware status cleanly without errors."""
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Promob Render Studio", timeout=15.0)
    cdp_browser.wait_for_text("GB VRAM", timeout=15.0)
    body = cdp_browser.evaluate("document.body.innerText") or ""
    assert "NaN" not in body and "undefined" not in body


def test_t1_f14_classic_highlight_recovery_clahe_sharpness_denoise(client, synthetic_1080p_interior, job_poller):
    """F14: Classic highlight_recovery, clahe_clip_limit, sharpness, and denoise_strength alter output deterministically."""
    base = _run_enhance_png(
        client,
        job_poller,
        synthetic_1080p_interior,
        "f14_base.jpg",
        {
            "render_mode": "classic",
            "temperature": "0.0",
            "tint": "0.0",
            "exposure": "0.0",
            "shadow_lift": "0.0",
            "highlight_recovery": "0.0",
            "clahe_clip_limit": "0.0",
            "ambient_occlusion": "0.0",
            "sharpness": "0.0",
            "denoise_strength": "0.0",
            "specular_bloom": "0.0",
            "upscale_factor": "1",
        },
        timeout=45.0,
    )
    tuned = _run_enhance_png(
        client,
        job_poller,
        synthetic_1080p_interior,
        "f14_tuned.jpg",
        {
            "render_mode": "classic",
            "temperature": "0.0",
            "tint": "0.0",
            "exposure": "0.0",
            "shadow_lift": "0.0",
            "highlight_recovery": "0.45",
            "clahe_clip_limit": "2.8",
            "ambient_occlusion": "0.0",
            "sharpness": "0.80",
            "denoise_strength": "0.25",
            "specular_bloom": "0.0",
            "upscale_factor": "1",
        },
        timeout=45.0,
    )
    a_base = np.asarray(Image.open(io.BytesIO(base)).convert("RGB"), dtype=np.float32)
    a_tuned = np.asarray(Image.open(io.BytesIO(tuned)).convert("RGB"), dtype=np.float32)
    assert a_base.shape == a_tuned.shape
    assert float(np.mean(np.abs(a_tuned - a_base))) > 0.5


# --- Additional GPU Tier 1 contracts closing F3, F4, F6, F7, F8, F9, F10, F11 ---

@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f3_window_daylight_directionality_in_generative_relight(client, job_poller):
    """F3: Bright window opening produces coherent directional daylight illumination gradient."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((512, 768, 3), (175, 175, 178), dtype=np.uint8)
    bgr[0:76, :] = (210, 210, 214)
    bgr[400:512, :] = (140, 125, 110)
    # Bright cool daylight window on right
    cv2.rectangle(bgr, (560, 95), (735, 365), (255, 245, 230), -1)
    cv2.rectangle(bgr, (560, 95), (735, 365), (90, 90, 95), 4)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "window_daylight.png",
        {"render_mode": "generative", "generative_strength": "0.52", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    relit = compute_relighting_metrics(enc.tobytes(), out, exclude_rois=[(0.18, 0.72, 0.72, 0.96)])
    assert relit["dL_low_spread"] >= 2.0, relit


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f3_wall_sconce_emits_local_warm_wash(client, job_poller):
    """F3: Wall sconces mounted on neutral wall produce local warm illumination wash."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((512, 512, 3), (185, 185, 188), dtype=np.uint8)
    bgr[0:70, :] = (212, 212, 216)
    bgr[410:512, :] = (145, 132, 118)
    # Two wall sconces
    for cx in (160, 352):
        cv2.rectangle(bgr, (cx - 12, 145), (cx + 12, 185), (225, 244, 255), -1)
        cv2.rectangle(bgr, (cx - 14, 143), (cx + 14, 187), (90, 90, 95), 2)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "sconces.png",
        {"render_mode": "generative", "generative_strength": "0.55", "scene_profile": "banheiro", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    spots = compute_spot_cone_metrics(enc.tobytes(), out, spot_roi=(0.24, 0.56, 0.22, 0.78))
    assert spots["spot_d_r_minus_b"] >= 3.0 and spots["spot_dL_spread"] >= 1.0, spots


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f3_depth_gated_contact_shadow_and_bounce(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F3: Depth-guided scene produces spatially varying low-freq illumination (dL_low_spread >= 2.5)."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "depth_bounce.png",
        {"render_mode": "generative", "generative_strength": "0.55", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    relit = compute_relighting_metrics(
        synthetic_spot_fireplace_scene_png, out, exclude_rois=[SPOT_FIREPLACE_ROIS["firebox"]]
    )
    assert relit["dL_low_std"] >= 0.8 and relit["dL_low_spread"] >= 2.5, relit


@pytest.mark.gpu
def test_t1_f4_decor_object_silhouette_preservation(client, job_poller):
    """F4: Shelf/decor vase and book silhouettes retain ZNCC, Edge F1, and SSIM under relighting."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((512, 512, 3), (188, 188, 190), dtype=np.uint8)
    # Floating shelf + crisp vase & books
    cv2.rectangle(bgr, (120, 290), (392, 308), (115, 95, 78), -1)
    cv2.rectangle(bgr, (165, 195), (195, 290), (50, 95, 145), -1)
    cv2.rectangle(bgr, (200, 210), (225, 290), (140, 80, 60), -1)
    cv2.ellipse(bgr, (315, 250), (28, 38), 0, 0, 360, (65, 115, 90), -1)
    cv2.rectangle(bgr, (305, 195), (325, 222), (65, 115, 90), -1)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "decor_shelf.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "preserve_geometry": "true",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    decor = compute_roi_preservation(enc.tobytes(), out, roi=(0.35, 0.62, 0.22, 0.78))
    assert decor["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], decor
    assert decor["edge_f1_d1"] >= 0.70, decor
    assert decor["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], decor


@pytest.mark.gpu
def test_t1_f4_multi_object_and_tv_simultaneous_protection(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F4: TV logo ('ELLEGANZA') stays OCR-readable and structurally crisp alongside spots & fireplace."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "multi_obj_tv.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.52",
            "scene_profile": "quarto_sala",
            "preserve_geometry": "true",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    tv = compute_roi_preservation(
        synthetic_spot_fireplace_scene_png,
        out,
        roi=SPOT_FIREPLACE_ROIS["tv_sign"],
        expected_text="ELLEGANZA",
    )
    assert tv["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], tv
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f6_fireplace_hearth_bounce_exceeds_neutral_wall(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F6: Hearth floor in front of firebox warms significantly more than distant neutral wall."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "fp_hearth_vs_wall.png",
        {"render_mode": "generative", "generative_strength": "0.60", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    fp = compute_fireplace_metrics(
        synthetic_spot_fireplace_scene_png,
        out,
        firebox_roi=SPOT_FIREPLACE_ROIS["firebox"],
        spill_roi=SPOT_FIREPLACE_ROIS["hearth_spill"],
    )
    wall = compute_relighting_metrics(
        synthetic_spot_fireplace_scene_png,
        out,
        exclude_rois=[
            SPOT_FIREPLACE_ROIS["firebox"],
            SPOT_FIREPLACE_ROIS["hearth_spill"],
            SPOT_FIREPLACE_ROIS["spot_wall_wash"],
        ],
    )
    assert fp["spill_d_r_minus_b"] > wall["d_r_minus_b"] + 3.0, (fp, wall)


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f6_wide_linear_firebox_synthesis(client, job_poller):
    """F6: Wide horizontal linear fireplace cutout gains flame dynamic range and warm chroma."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((512, 768, 3), (182, 182, 185), dtype=np.uint8)
    bgr[380:512, :] = (160, 160, 162)
    # Wide linear dark firebox
    cv2.rectangle(bgr, (180, 290), (588, 376), (72, 72, 76), -1)
    cv2.rectangle(bgr, (192, 300), (576, 366), (22, 22, 24), -1)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "wide_fp.png",
        {"render_mode": "generative", "generative_strength": "0.60", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    fp = compute_fireplace_metrics(
        enc.tobytes(),
        out,
        firebox_roi=(0.58, 0.72, 0.25, 0.75),
        spill_roi=(0.75, 0.94, 0.24, 0.76),
    )
    assert fp["fp_L_rng_ratio"] >= 1.4 and fp["fp_dchr"] >= 8.0, fp


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f6_fireplace_flame_luminance_and_chroma_dynamic_range(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F6: Synthesized flames expand L* dynamic range, chroma, and hearth spill luminance."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "fp_dr.png",
        {"render_mode": "generative", "generative_strength": "0.60", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    fp = compute_fireplace_metrics(
        synthetic_spot_fireplace_scene_png,
        out,
        firebox_roi=SPOT_FIREPLACE_ROIS["firebox"],
        spill_roi=SPOT_FIREPLACE_ROIS["hearth_spill"],
    )
    assert fp["fp_L_rng_out"] > fp["fp_L_rng_in"], fp
    assert fp["fp_chroma_out"] > fp["fp_chroma_in"] + 8.0, fp
    assert fp["spill_dL"] >= 1.5, fp


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f6_fireplace_with_realismo_ia_preset_parameters(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F6: Full realismo_ia preset config (1x) ignites firebox while keeping dE_ab_norm <= 6.5."""
    timeout_s = _require_gpu_timeout(client)
    cfg = client.get("/api/presets").json()["realismo_ia"]["config"]
    form_data = {k: ("true" if v is True else "false" if v is False else str(v)) for k, v in cfg.items()}
    form_data["upscale_factor"] = "1"
    out = _run_enhance_png(
        client, job_poller, synthetic_spot_fireplace_scene_png, "fp_preset.png", form_data, timeout=timeout_s
    )
    fp = compute_fireplace_metrics(
        synthetic_spot_fireplace_scene_png,
        out,
        firebox_roi=SPOT_FIREPLACE_ROIS["firebox"],
        spill_roi=SPOT_FIREPLACE_ROIS["hearth_spill"],
    )
    geom = compute_geometry_preservation(
        synthetic_spot_fireplace_scene_png, out, exclude_rois=[SPOT_FIREPLACE_ROIS["firebox"]]
    )
    assert fp["fp_L_rng_ratio"] >= 1.4 and fp["fp_dchr"] >= 8.0, fp
    assert geom["dE_ab_norm"] <= R2_THRESHOLDS["dE_ab_norm_max"], geom


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f7_ao_contact_shadows_and_bounce_dynamics(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F7: Neural relighting creates meaningful spatial illumination variance (dL_low_spread >= 2.5)."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "f7_ao_bounce.png",
        {"render_mode": "generative", "generative_strength": "0.55", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    relit = compute_relighting_metrics(
        synthetic_spot_fireplace_scene_png, out, exclude_rois=[SPOT_FIREPLACE_ROIS["firebox"]]
    )
    assert relit["dL_low_spread"] >= 2.5 and relit["dL_low_std"] >= 0.8, relit


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f7_pbr_wood_and_stone_material_depth(client, promob_reference_image, job_poller):
    """F7: Neural relighting preserves architectural Sobel/SSIM while deepening PBR light contrast."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        promob_reference_image,
        "f7_pbr.jpg",
        {"render_mode": "generative", "generative_strength": "0.48", "scene_profile": "cozinha", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(promob_reference_image, out)
    relit = compute_relighting_metrics(promob_reference_image, out)
    assert geom["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"], geom
    assert geom["global_ssim_L"] >= 0.75, geom
    assert relit["dL_low_spread"] >= 2.0, relit


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f7_mid_vs_high_strength_relighting_amplitude(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F7: generative_strength=0.65 yields stronger relighting delta than generative_strength=0.25."""
    timeout_s = _require_gpu_timeout(client)
    low = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "f7_025.png",
        {"render_mode": "generative", "generative_strength": "0.25", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    high = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "f7_065.png",
        {"render_mode": "generative", "generative_strength": "0.65", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    in_arr = decode_bgr(synthetic_spot_fireplace_scene_png).astype(np.float32)
    mae_low = float(np.mean(np.abs(decode_bgr(low).astype(np.float32) - in_arr)))
    mae_high = float(np.mean(np.abs(decode_bgr(high).astype(np.float32) - in_arr)))
    assert mae_high > mae_low, f"Expected strength 0.65 MAE ({mae_high:.2f}) > 0.25 MAE ({mae_low:.2f})"


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f8_unclamped_wall_chroma_exceeds_legacy_3_0_cap(
    client, chatgpt_reference_pair, job_poller
):
    """F8: Unclamped architectural blending allows warm spot/bounce chroma shift > legacy 3.0 Lab cap."""
    timeout_s = _require_gpu_timeout(client)
    before = chatgpt_reference_pair["before"]
    rois = chatgpt_reference_pair["rois"]
    out = _run_enhance_png(
        client,
        job_poller,
        before,
        "f8_unclamped.jpg",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "scene_profile": "quarto_sala",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    relit = compute_relighting_metrics(before, out, exclude_rois=[rois["fireplace_firebox"]])
    assert relit["db_mean"] > 3.5 or relit["d_r_minus_b"] >= R1_THRESHOLDS["d_r_minus_b_min"], relit


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f8_luminance_modulated_sign_on_lit_wall(
    client, synthetic_text_tv_scene_png, job_poller
):
    """F8: Protected framed sign retains crisp text & OCR while adapting smoothly to local wall light."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "f8_sign.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "scene_profile": "quarto_sala",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    sign = compute_roi_preservation(
        synthetic_text_tv_scene_png, out, roi=TEXT_TV_ROIS["framed_sign"], expected_text="WOLF"
    )
    assert sign["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], sign
    assert sign["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], sign
    if sign["ocr_available"]:
        assert sign["ocr_matched"] is True, sign
    relit = compute_relighting_metrics(synthetic_text_tv_scene_png, out)
    assert relit["dL_low_spread"] >= 2.0 or abs(relit["d_r_minus_b"]) >= 4.0, relit


@pytest.mark.gpu
def test_t1_f8_protected_object_retains_sharpness_under_high_strength(
    client, synthetic_text_tv_scene_png, job_poller
):
    """F8: TV & sign retain edge_f1 >= 0.75 and ZNCC >= 0.72 at generative_strength=0.65."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "f8_highstr.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.65",
            "latent_object_lock": "true",
            "preserve_geometry": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    tv = compute_roi_preservation(
        synthetic_text_tv_scene_png, out, roi=TEXT_TV_ROIS["tv_screen"], expected_text="PROMOB"
    )
    assert tv["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], tv
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv


@pytest.mark.gpu
@pytest.mark.relighting
def test_t1_f8_luminance_modulated_tv_in_chatgpt_reference(
    client, chatgpt_reference_pair, job_poller
):
    """F8: Real ChatGPT reference TV ('Elleganza') stays OCR-legible while room relights."""
    timeout_s = _require_gpu_timeout(client)
    before = chatgpt_reference_pair["before"]
    rois = chatgpt_reference_pair["rois"]
    out = _run_enhance_png(
        client,
        job_poller,
        before,
        "f8_ref_tv.jpg",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "quarto_sala",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    tv = compute_roi_preservation(before, out, roi=rois["tv_screen"], expected_text="Elleganza")
    assert tv["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], tv
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv
    relit = compute_relighting_metrics(before, out, exclude_rois=[rois["fireplace_firebox"]])
    assert relit["dL_low_spread"] >= 6.0 and relit["d_r_minus_b"] >= 10.0, relit


@pytest.mark.gpu
def test_t1_f9_wood_species_oak_vs_walnut_hue_preservation(client, job_poller):
    """F9: Distinct light oak and dark walnut panels preserve tonal separation and hue fidelity."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((512, 512, 3), (188, 188, 190), dtype=np.uint8)
    bgr[120:390, 60:220] = (95, 140, 185)   # Light honey oak (BGR)
    bgr[120:390, 290:450] = (40, 62, 95)    # Dark walnut (BGR)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "wood_species.png",
        {"render_mode": "generative", "generative_strength": "0.48", "scene_profile": "cozinha", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    out_bgr = decode_bgr(out)
    oak_l = float(np.mean(out_bgr[150:360, 80:200]))
    walnut_l = float(np.mean(out_bgr[150:360, 310:430]))
    assert oak_l > walnut_l + 20.0, f"Oak ({oak_l:.1f}) and walnut ({walnut_l:.1f}) collapsed"
    geom = compute_geometry_preservation(enc.tobytes(), out)
    assert geom["dE_ab_norm"] <= R2_THRESHOLDS["dE_ab_norm_max"], geom


@pytest.mark.gpu
def test_t1_f9_white_and_cool_gray_surfaces_resist_yellowing(client, job_poller):
    """F9: Bright white/cool-gray bathroom walls remain free of false yellowing (dE_ab_norm <= 6.5)."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((512, 512, 3), (236, 234, 232), dtype=np.uint8)
    bgr[140:380, 140:370] = (175, 175, 178)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "white_bath.png",
        {"render_mode": "generative", "generative_strength": "0.48", "scene_profile": "banheiro", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(enc.tobytes(), out)
    assert geom["dE_ab_norm"] <= R2_THRESHOLDS["dE_ab_norm_max"], geom


@pytest.mark.gpu
@pytest.mark.slow
def test_t1_f10_generative_plus_4x_realesrgan_handoff_within_5632mb(
    client, promob_reference_image, job_poller, vram_monitor
):
    """F10: PyTorch -> DirectML ONNX 4x handoff stays <4600 MB PyTorch / <5632 MB nvidia-smi and releases VRAM."""
    timeout_s = _require_gpu_timeout(client)
    pre_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    with vram_monitor() as mon:
        out = _run_enhance_png(
            client,
            job_poller,
            promob_reference_image,
            "f10_handoff.jpg",
            {
                "render_mode": "generative",
                "generative_strength": "0.48",
                "upscale_factor": "4",
                "tile_size": "256",
            },
            timeout=timeout_s,
        )
    assert max(Image.open(io.BytesIO(out)).size) >= 3600
    mon.assert_within_budget(max_nvidia_mb=5632.0, max_delta_nvidia_mb=4600.0, max_torch_mb=4600.0)
    post_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    mon.assert_health_released(pre_alloc, post_alloc, tolerance_mb=512.0)


@pytest.mark.gpu
def test_t1_f10_sequential_native_generative_renders_release_vram(
    client, synthetic_png, job_poller, vram_monitor
):
    """F10: Two consecutive native-res generative renders release VRAM cleanly with no leak."""
    timeout_s = _require_gpu_timeout(client)
    pre_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    peaks = []
    for i in range(2):
        with vram_monitor() as mon:
            _run_enhance_png(
                client,
                job_poller,
                synthetic_png,
                f"seq_nat_{i}.png",
                {"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
                timeout=timeout_s,
            )
        mon.assert_within_budget(max_nvidia_mb=5632.0, max_delta_nvidia_mb=4600.0, max_torch_mb=4600.0)
        peaks.append(mon.peak_torch_mb if mon.peak_torch_mb > 0.0 else mon.peak_nvidia_mb)
    assert peaks[1] <= peaks[0] + 512.0, f"VRAM grew across native renders: {peaks}"
    post_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    vram_monitor.assert_health_released(pre_alloc, post_alloc, tolerance_mb=512.0)


@pytest.mark.gpu
@pytest.mark.slow
def test_t1_f11_generative_4x_upscale_with_photo_finish_preserves_edges(
    client, promob_reference_image, job_poller
):
    """F11: Generative + 4x Real-ESRGAN + photo_finish=0.5 produces exact 4x PNG preserving Sobel >= 0.72."""
    timeout_s = _require_gpu_timeout(client)
    orig_w, orig_h = Image.open(io.BytesIO(promob_reference_image)).size
    out = _run_enhance_png(
        client,
        job_poller,
        promob_reference_image,
        "f11_gen4x.jpg",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "photo_finish": "0.5",
            "upscale_factor": "4",
            "tile_size": "256",
            "preserve_geometry": "true",
        },
        timeout=timeout_s,
    )
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.size == (orig_w * 4, orig_h * 4)
    geom = compute_geometry_preservation(promob_reference_image, out)
    assert geom["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"], geom


@pytest.mark.gpu
def test_t1_f11_photo_finish_zero_vs_one_filmic_contrast_and_grain(
    client, promob_reference_image, job_poller
):
    """F11: photo_finish=1.0 vs 0.0 in generative mode applies filmic finish without highlight blowout."""
    timeout_s = _require_gpu_timeout(client)
    base = {
        "render_mode": "generative",
        "generative_strength": "0.48",
        "scene_profile": "auto",
        "upscale_factor": "1",
    }
    p0 = _run_enhance_png(
        client, job_poller, promob_reference_image, "pf_0.jpg", {**base, "photo_finish": "0.0"}, timeout=timeout_s
    )
    p1 = _run_enhance_png(
        client, job_poller, promob_reference_image, "pf_1.jpg", {**base, "photo_finish": "1.0"}, timeout=timeout_s
    )
    a0 = np.asarray(Image.open(io.BytesIO(p0)).convert("RGB"), dtype=np.float32)
    a1 = np.asarray(Image.open(io.BytesIO(p1)).convert("RGB"), dtype=np.float32)
    assert float(np.mean(np.abs(a1 - a0))) > 0.5
    assert float(np.mean(a1)) < 245.0
