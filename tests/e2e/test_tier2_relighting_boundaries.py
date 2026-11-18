"""Tier 2: Boundary & Degenerate Input E2E Tests for F1-F14 & R1-R4.

Covers:
- CLI invalid flags, offline HF_HOME missing-cache handling, active TCP socket hook blocking
- Schema bounds & HTTP 422/400/404 validation for photo_finish, upscale_factor, led_strips,
  latent_object_lock, freeu, prompt_style, tile_size, RGBA/grayscale/uint16 PNGs
- Degenerate inputs (pitch black, pure white, uniform wall, zero yellow pixels, non-LED yellow blobs,
  L-shaped LEDs, odd dimensions, packed text, dark cabinet non-fireplace guard)
- GPU boundary cases (@pytest.mark.gpu)
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from PIL import Image

from tests.e2e.conftest import (
    OAK_CABINET_ROIS,
    REPO_ROOT,
    SPOT_FIREPLACE_ROIS,
    TEXT_TV_ROIS,
    compute_lab_chroma_stats,
)
from tests.e2e.metrics_helpers import (
    R1_THRESHOLDS,
    R2_THRESHOLDS,
    compute_fireplace_metrics,
    compute_geometry_preservation,
    compute_led_continuity_metrics,
    compute_relighting_metrics,
    compute_roi_preservation,
    decode_bgr,
)


def _run_enhance_png(
    client,
    job_poller,
    image_bytes: bytes,
    filename: str,
    data: dict,
    timeout: float = 120.0,
) -> bytes:
    mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    resp = client.post("/api/enhance", files={"file": (filename, image_bytes, mime)}, data=data)
    assert resp.status_code in (200, 202), f"/api/enhance failed ({resp.status_code}): {resp.text}"
    job_id = resp.json()["job_id"]
    job_data = job_poller(client, job_id, timeout_seconds=timeout)
    assert job_data.get("status") in ("completed", "finished", "success")
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    return dl.content


def _encode_png(rgb_or_gray: np.ndarray, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_or_gray, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _require_gpu_timeout(client) -> float:
    caps = client.get("/api/capabilities").json()
    ge = caps.get("generative_engine", {})
    assert ge.get("available") is True, f"Generative engine unavailable: {caps}"
    return 600.0 if not ge.get("cached_locally") else 300.0


# ============================================================================
# Group A: F1, F2, F11, F12, F13, F14 CLI, Offline & Schema Boundaries (CPU)
# ============================================================================

def test_t2_f1_download_models_invalid_flag_rejected():
    """download_models.py rejects unknown flags or missing mode flags with non-zero exit."""
    for args in ([], ["--bogus-flag"]):
        proc = subprocess.run(
            [sys.executable, "backend/scripts/download_models.py", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
        assert proc.returncode != 0, f"Expected non-zero exit for args={args}"


@pytest.mark.offline
def test_t2_f1_offline_server_invalid_hf_home_fallback(offline_audited_server_factory, tmp_path):
    """Pointing HF_HOME at an empty directory offline reports cached_locally=False without network calls."""
    import httpx

    empty_hf = tmp_path / "empty_hf_home"
    empty_hf.mkdir()
    srv = offline_audited_server_factory({"HF_HOME": str(empty_hf), "HUGGINGFACE_HUB_CACHE": str(empty_hf)})
    with httpx.Client(base_url=srv.base_url, timeout=30.0) as http:
        caps = http.get("/api/capabilities")
    assert caps.status_code == 200
    ge = caps.json().get("generative_engine", {})
    assert ge.get("cached_locally") is False
    assert srv.get_outbound_attempts() == []


@pytest.mark.offline
def test_t2_f1_offline_socket_hook_blocks_non_loopback_tcp(offline_audited_server):
    """Offline socket-audit hook raises PermissionError and logs blocked=True on external TCP attempts."""
    hook_dir = offline_audited_server.log_path.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(hook_dir)
    env["PROMOB_SOCKET_AUDIT_LOG"] = str(offline_audited_server.log_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import socket; socket.create_connection(('93.184.216.34', 80), timeout=0.5)",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )
    assert proc.returncode != 0
    assert "PermissionError" in proc.stderr or "Offline socket audit blocked" in proc.stderr
    assert len(offline_audited_server.get_outbound_attempts()) >= 1


def test_t2_f2_bench_realismo_invalid_flag_rejected():
    """bench_realismo.py rejects unknown CLI flags with non-zero exit."""
    proc = subprocess.run(
        [sys.executable, "backend/scripts/bench_realismo.py", "--not-a-valid-option"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )
    assert proc.returncode != 0


def test_t2_f2_concurrent_preview_during_classic_enhance_stays_fast(
    client, synthetic_1080p_interior, synthetic_jpeg, job_poller
):
    """Interactive /api/preview requests remain responsive (<500ms) alongside an active enhance job."""
    enh = client.post(
        "/api/enhance",
        files={"file": ("bg.jpg", synthetic_1080p_interior, "image/jpeg")},
        data={"render_mode": "classic", "upscale_factor": "1"},
    )
    assert enh.status_code in (200, 202)
    job_id = enh.json()["job_id"]

    latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        p = client.post(
            "/api/preview",
            files={"file": ("fg.jpg", synthetic_jpeg, "image/jpeg")},
            data={"render_mode": "classic"},
        )
        latencies.append(time.perf_counter() - t0)
        assert p.status_code == 200

    job_poller(client, job_id, timeout_seconds=45.0)
    assert min(latencies) < 0.50, f"Preview stalled during background job: {latencies}"


def test_t2_f11_photo_finish_valid_bounds_and_invalid_rejected(client, synthetic_jpeg, job_poller):
    """photo_finish accepts [0.0, 0.5, 1.0] (200) and rejects [-0.01, 1.01, nan, abc] (422)."""
    for valid_pf in ("0.0", "0.5", "1.0"):
        prev = client.post(
            "/api/preview",
            files={"file": ("pf.jpg", synthetic_jpeg, "image/jpeg")},
            data={"photo_finish": valid_pf},
        )
        assert prev.status_code == 200
        enh = client.post(
            "/api/enhance",
            files={"file": ("pf.jpg", synthetic_jpeg, "image/jpeg")},
            data={"photo_finish": valid_pf, "render_mode": "classic", "upscale_factor": "1"},
        )
        assert enh.status_code in (200, 202)
        job_poller(client, enh.json()["job_id"], timeout_seconds=30.0)

    for bad_pf in ("-0.01", "1.01", "nan", "abc"):
        prev = client.post(
            "/api/preview",
            files={"file": ("pf.jpg", synthetic_jpeg, "image/jpeg")},
            data={"photo_finish": bad_pf},
        )
        assert prev.status_code == 422, f"Expected 422 for photo_finish={bad_pf!r}, got {prev.status_code}"
        enh = client.post(
            "/api/enhance",
            files={"file": ("pf.jpg", synthetic_jpeg, "image/jpeg")},
            data={"photo_finish": bad_pf, "upscale_factor": "1"},
        )
        assert enh.status_code == 422


def test_t2_f11_upscale_factor_valid_and_invalid_bounds(client, synthetic_jpeg):
    """upscale_factor accepts [1, 2, 3, 4] (200) and rejects [0, 5, -1, high] (422)."""
    for valid_u in ("1", "2", "3", "4"):
        resp = client.post(
            "/api/preview",
            files={"file": ("u.jpg", synthetic_jpeg, "image/jpeg")},
            data={"upscale_factor": valid_u},
        )
        assert resp.status_code == 200
    for bad_u in ("0", "5", "-1", "high"):
        resp = client.post(
            "/api/preview",
            files={"file": ("u.jpg", synthetic_jpeg, "image/jpeg")},
            data={"upscale_factor": bad_u},
        )
        assert resp.status_code == 422


def test_t2_f12_led_strips_bool_and_invalid_rejected(client, synthetic_jpeg):
    """led_strips accepts true/false (200) and rejects non-boolean strings (422)."""
    for v in ("true", "false"):
        assert client.post(
            "/api/preview", files={"file": ("l.jpg", synthetic_jpeg, "image/jpeg")}, data={"led_strips": v}
        ).status_code == 200
    assert client.post(
        "/api/preview", files={"file": ("l.jpg", synthetic_jpeg, "image/jpeg")}, data={"led_strips": "maybe"}
    ).status_code == 422


def test_t2_f12_latent_object_lock_and_freeu_bool_validation(client, synthetic_jpeg):
    """latent_object_lock and freeu accept booleans (200) and reject invalid values (422)."""
    assert client.post(
        "/api/preview",
        files={"file": ("b.jpg", synthetic_jpeg, "image/jpeg")},
        data={"latent_object_lock": "false", "freeu": "true"},
    ).status_code == 200
    assert client.post(
        "/api/preview",
        files={"file": ("b.jpg", synthetic_jpeg, "image/jpeg")},
        data={"latent_object_lock": "not_bool"},
    ).status_code == 422
    assert client.post(
        "/api/preview",
        files={"file": ("b.jpg", synthetic_jpeg, "image/jpeg")},
        data={"freeu": "not_bool"},
    ).status_code == 422


def test_t2_f12_prompt_style_valid_and_invalid(client, synthetic_jpeg):
    """prompt_style accepts legacy/photo (200) and rejects empty/unknown strings (422)."""
    for valid_ps in ("legacy", "photo"):
        assert client.post(
            "/api/preview", files={"file": ("p.jpg", synthetic_jpeg, "image/jpeg")}, data={"prompt_style": valid_ps}
        ).status_code == 200
    for bad_ps in ("", "chatgpt", "cinematic"):
        assert client.post(
            "/api/preview", files={"file": ("p.jpg", synthetic_jpeg, "image/jpeg")}, data={"prompt_style": bad_ps}
        ).status_code == 422


def test_t2_f12_tile_size_valid_and_invalid_bounds(client, synthetic_jpeg):
    """tile_size accepts [128, 256, 512] (200) and rejects [64, 1024, -1] (422)."""
    for ts in ("128", "256", "512"):
        assert client.post(
            "/api/preview", files={"file": ("t.jpg", synthetic_jpeg, "image/jpeg")}, data={"tile_size": ts}
        ).status_code == 200
    for bad_ts in ("64", "1024", "-1"):
        assert client.post(
            "/api/preview", files={"file": ("t.jpg", synthetic_jpeg, "image/jpeg")}, data={"tile_size": bad_ts}
        ).status_code == 422


def test_t2_f12_rgba_4channel_and_grayscale_1channel_png_accepted(client, job_poller):
    """4-channel RGBA PNG and 1-channel grayscale PNG decode and process cleanly via API."""
    rgba = np.full((128, 128, 4), (160, 140, 120, 255), dtype=np.uint8)
    rgba_bytes = _encode_png(rgba, mode="RGBA")
    gray = np.full((128, 128), 140, dtype=np.uint8)
    gray_bytes = _encode_png(gray, mode="L")

    for name, payload in (("rgba.png", rgba_bytes), ("gray.png", gray_bytes)):
        prev = client.post("/api/preview", files={"file": (name, payload, "image/png")})
        assert prev.status_code == 200
        out = _run_enhance_png(
            client, job_poller, payload, name, {"render_mode": "classic", "upscale_factor": "1"}, timeout=45.0
        )
        assert Image.open(io.BytesIO(out)).size == (128, 128)


def test_t2_f12_uint16_png_accepted_without_error(client, job_poller):
    """16-bit grayscale PNG uploads are normalized cleanly without HTTP 400/500."""
    arr16 = np.full((128, 128), 32768, dtype=np.uint16)
    ok, enc = cv2.imencode(".png", arr16)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "u16.png",
        {"render_mode": "classic", "upscale_factor": "1"},
        timeout=45.0,
    )
    assert Image.open(io.BytesIO(out)).size == (128, 128)


def test_t2_f12_malformed_and_missing_download_job_ids(client):
    """Malformed job_id returns 400 and missing valid-syntax job_id returns 404 on /api/jobs and /api/download."""
    assert client.get("/api/jobs/bad$id;drop").status_code == 400
    assert client.get("/api/download/bad$id;drop").status_code == 400
    missing_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/jobs/{missing_id}").status_code == 404
    assert client.get(f"/api/download/{missing_id}").status_code == 404


@pytest.mark.browser
def test_t2_f13_browser_dropzone_rejects_non_image_file(cdp_browser, tmp_path):
    """Uploading a .txt file into Dropzone <input type='file'> keeps Dropzone intact without crashing."""
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not an image", encoding="utf-8")

    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    cdp_browser.set_file_input("input[type='file']", str(bad_file))
    cdp_browser.pump_events(0.4)
    body = cdp_browser.evaluate("document.body.innerText") or ""
    assert "Carregar Amostra Promob" in body


def test_t2_f14_classic_clahe_and_ao_extreme_bounds(client, synthetic_jpeg):
    """clahe_clip_limit [0.0, 5.0] and ambient_occlusion [0.0, 1.0] pass (200); OOB returns 422."""
    for data in ({"clahe_clip_limit": "0.0"}, {"clahe_clip_limit": "5.0"}, {"ambient_occlusion": "1.0"}):
        assert client.post(
            "/api/preview", files={"file": ("b.jpg", synthetic_jpeg, "image/jpeg")}, data=data
        ).status_code == 200
    for bad in ({"clahe_clip_limit": "-0.1"}, {"clahe_clip_limit": "5.1"}, {"ambient_occlusion": "-0.1"}, {"ambient_occlusion": "1.1"}):
        assert client.post(
            "/api/preview", files={"file": ("b.jpg", synthetic_jpeg, "image/jpeg")}, data=bad
        ).status_code == 422


# ============================================================================
# Group B: F3-F10 Degenerate Image, Emitter & Protection Boundaries (CPU + GPU)
# ============================================================================

def test_t2_f3_pitch_black_image_classic_and_preview_no_nan(client):
    """All-zero (0,0,0) PNG runs /api/preview cleanly with zero NaN or crash."""
    black = _encode_png(np.zeros((256, 256, 3), dtype=np.uint8))
    resp = client.post(
        "/api/preview",
        files={"file": ("black.png", black, "image/png")},
        data={"render_mode": "generative", "scene_profile": "auto"},
    )
    assert resp.status_code == 200
    arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
    assert not np.isnan(arr).any()


def test_t2_f3_pure_white_image_classic_and_preview_no_nan(client):
    """All-255 pure white PNG runs /api/preview cleanly with zero NaN or crash."""
    white = _encode_png(np.full((256, 256, 3), 255, dtype=np.uint8))
    resp = client.post(
        "/api/preview",
        files={"file": ("white.png", white, "image/png")},
        data={"render_mode": "generative", "specular_bloom": "1.0"},
    )
    assert resp.status_code == 200
    arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
    assert not np.isnan(arr).any()


def test_t2_f3_uniform_flat_wall_preview_no_artificial_spot_blobs(client):
    """Uniform flat wall (185,185,185) stays spatially uniform under /api/preview without spot blobs."""
    flat = _encode_png(np.full((256, 256, 3), 185, dtype=np.uint8))
    resp = client.post(
        "/api/preview",
        files={"file": ("flat.png", flat, "image/png")},
        data={"render_mode": "classic", "ambient_occlusion": "0.5", "specular_bloom": "0.5"},
    )
    assert resp.status_code == 200
    arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"), dtype=np.float32)
    assert float(np.std(arr)) < 3.0


def test_t2_f4_empty_scene_zero_objects_zero_text_cpu(client, job_poller):
    """Empty scene with zero objects and zero text completes cleanly at strength=0.0."""
    empty_room = _encode_png(np.full((256, 256, 3), 175, dtype=np.uint8))
    out = _run_enhance_png(
        client,
        job_poller,
        empty_room,
        "empty.png",
        {"render_mode": "generative", "generative_strength": "0.0", "upscale_factor": "1"},
        timeout=45.0,
    )
    assert Image.open(io.BytesIO(out)).size == (256, 256)


def test_t2_f4_tiny_low_contrast_text_does_not_crash_preview(client):
    """Tiny low-contrast text on 256x256 PNG processes cleanly on /api/preview."""
    bgr = np.full((256, 256, 3), 170, dtype=np.uint8)
    cv2.putText(bgr, "mini", (40, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (152, 152, 152), 1)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    resp = client.post("/api/preview", files={"file": ("tiny_txt.png", enc.tobytes(), "image/png")})
    assert resp.status_code == 200


def test_t2_f5_zero_yellow_pixels_no_hallucinated_led_strips(client, synthetic_bloom_png, job_poller):
    """Image without yellow markers receives zero yellow_px detections and zero hallucinated LEDs in out."""
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_bloom_png,
        "no_led.png",
        {"render_mode": "classic", "led_strips": "true", "upscale_factor": "1"},
        timeout=45.0,
    )
    led = compute_led_continuity_metrics(synthetic_bloom_png, out)
    assert led["yellow_px"] == 0.0, led
    assert led["out_yellow_px"] == 0.0, led
    assert led["out_spurious_streak_px"] == 0.0, led


def test_t2_f5_non_led_yellow_decor_blob_not_overblown_preview(client):
    """Large 64x64 yellow decor object is not blown out into a blinding full-frame glare."""
    bgr = np.full((256, 256, 3), 140, dtype=np.uint8)
    bgr[96:160, 96:160] = (0, 230, 250)  # Large yellow decor pillow/vase
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    resp = client.post(
        "/api/preview",
        files={"file": ("yellow_vase.png", enc.tobytes(), "image/png")},
        data={"led_strips": "true", "specular_bloom": "0.5"},
    )
    assert resp.status_code == 200
    arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"), dtype=np.float32)
    assert float(np.mean(arr)) < 245.0


def test_t2_f5_l_shaped_corner_led_strip_preview(client):
    """L-shaped 4px yellow LED strip renders smoothly without yellow remnants or gaps on /api/preview."""
    bgr = np.full((256, 256, 3), 175, dtype=np.uint8)
    bgr[60:64, 50:200] = (0, 238, 255)
    bgr[60:200, 196:200] = (0, 238, 255)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    raw = enc.tobytes()
    resp = client.post(
        "/api/preview",
        files={"file": ("l_led.png", raw, "image/png")},
        data={"led_strips": "true"},
    )
    assert resp.status_code == 200
    led = compute_led_continuity_metrics(raw, resp.content)
    assert led["yellow_px"] >= 500.0, led
    assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], led
    assert led["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], led
    assert led["gap_frac"] <= R1_THRESHOLDS["led_gap_frac_max"], led
    assert led["spill_dL"] >= R1_THRESHOLDS["led_spill_dL_min"], led


def test_t2_f5_odd_dimensions_led_strip_enhance_cpu(client, job_poller):
    """Odd dimensions (501x333) with a yellow LED strip complete cleanly at upscale_factor=1."""
    bgr = np.full((333, 501, 3), 178, dtype=np.uint8)
    bgr[110:115, 60:440] = (0, 238, 255)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    raw = enc.tobytes()
    out = _run_enhance_png(
        client,
        job_poller,
        raw,
        "odd_led.png",
        {"render_mode": "classic", "led_strips": "true", "upscale_factor": "1"},
        timeout=45.0,
    )
    assert Image.open(io.BytesIO(out)).size == (501, 333)
    led = compute_led_continuity_metrics(raw, out)
    assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], led
    assert led["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], led
    assert led["gap_frac"] <= R1_THRESHOLDS["led_gap_frac_max"], led


def test_t2_f6_generative_strength_zero_skips_fireplace_synthesis(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """generative_strength=0.0 bypasses fireplace synthesis so unlit firebox stays unlit."""
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "fp_zero.png",
        {"render_mode": "generative", "generative_strength": "0.0", "upscale_factor": "1"},
        timeout=45.0,
    )
    fp = compute_fireplace_metrics(
        synthetic_spot_fireplace_scene_png,
        out,
        firebox_roi=SPOT_FIREPLACE_ROIS["firebox"],
        spill_roi=SPOT_FIREPLACE_ROIS["hearth_spill"],
    )
    assert fp["fp_dchr"] < 2.0, fp


def test_t2_f6_tiny_dark_rectangle_not_corrupted_at_strength_zero(client, job_poller):
    """Small (<32px) dark cabinet hardware stays intact at generative_strength=0.0."""
    rgb = np.full((256, 256, 3), 185, dtype=np.uint8)
    rgb[120:140, 120:140] = (20, 20, 22)
    raw = _encode_png(rgb)
    out = _run_enhance_png(
        client,
        job_poller,
        raw,
        "tiny_dark.png",
        {"render_mode": "generative", "generative_strength": "0.0", "upscale_factor": "1"},
        timeout=45.0,
    )
    out_arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), dtype=np.float32)
    assert float(np.mean(out_arr[124:136, 124:136])) < 60.0


def test_t2_f7_extreme_aspect_ratios_generative_strength_zero(
    client, synthetic_banner_image, synthetic_tall_image, job_poller
):
    """1920x64 and 64x1920 extreme aspect ratios keep exact geometry at strength=0.0."""
    for name, payload, expected in (
        ("banner.jpg", synthetic_banner_image, (1920, 64)),
        ("tall.jpg", synthetic_tall_image, (64, 1920)),
    ):
        out = _run_enhance_png(
            client,
            job_poller,
            payload,
            name,
            {"render_mode": "generative", "generative_strength": "0.0", "upscale_factor": "1"},
            timeout=45.0,
        )
        assert Image.open(io.BytesIO(out)).size == expected


def test_t2_f8_sharp_boundary_black_tv_bezel_stays_dark_classic(
    client, synthetic_text_tv_scene_png, job_poller
):
    """Dark TV bezel retains dark black boundary without glowing halo bleeding inward."""
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "bezel.png",
        {"render_mode": "classic", "specular_bloom": "0.5", "upscale_factor": "1"},
        timeout=45.0,
    )
    arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"))
    assert float(np.min(arr[114:122, 118:405])) < 65.0


def test_t2_f9_monochrome_room_preserves_neutrality_classic(client, job_poller):
    """Monochrome gray room (170,170,170) stays neutral without inventing false color cast."""
    gray_room = _encode_png(np.full((256, 256, 3), 170, dtype=np.uint8))
    out = _run_enhance_png(
        client,
        job_poller,
        gray_room,
        "mono.png",
        {"render_mode": "classic", "temperature": "0.0", "tint": "0.0", "upscale_factor": "1"},
        timeout=45.0,
    )
    stats = compute_lab_chroma_stats(out)
    assert abs(stats["a"] - 128.0) < 3.0 and abs(stats["b"] - 128.0) < 3.0, stats


def test_t2_f10_two_rapid_concurrent_classic_enhance_jobs_no_error(client, synthetic_png, job_poller):
    """Two rapid concurrent /api/enhance submissions complete safely without lock/race failure."""
    def _submit():
        resp = client.post(
            "/api/enhance",
            files={"file": ("race.png", synthetic_png, "image/png")},
            data={"render_mode": "classic", "upscale_factor": "1"},
        )
        assert resp.status_code in (200, 202)
        return resp.json()["job_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_submit)
        f2 = pool.submit(_submit)
        j1, j2 = f1.result(), f2.result()

    assert job_poller(client, j1, timeout_seconds=45.0).get("status") in ("completed", "finished", "success")
    assert job_poller(client, j2, timeout_seconds=45.0).get("status") in ("completed", "finished", "success")


@pytest.mark.browser
def test_t2_f13_browser_reset_clears_state(cdp_browser):
    """Clicking 'Reiniciar' in browser Sidebar cleanly resets Viewer back to Dropzone."""
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    cdp_browser.click_text("Carregar Amostra Promob")
    cdp_browser.wait_for_text("Reiniciar", timeout=15.0)
    cdp_browser.click_text("Reiniciar")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=10.0)


# ============================================================================
# GPU Degenerate & Boundary Cases (@pytest.mark.gpu)
# ============================================================================

@pytest.mark.gpu
def test_t2_f3_pitch_black_image_generative_no_nan(client, job_poller):
    """All-black 256x256 input under generative relighting produces valid finite PNG without NaN."""
    timeout_s = _require_gpu_timeout(client)
    black = _encode_png(np.zeros((256, 256, 3), dtype=np.uint8))
    out = _run_enhance_png(
        client,
        job_poller,
        black,
        "black_gen.png",
        {"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"))
    assert arr.shape == (256, 256, 3) and not np.isnan(arr).any()


@pytest.mark.gpu
def test_t2_f3_uniform_flat_wall_generative_no_spurious_spots(client, job_poller):
    """Uniform flat gray wall without spots/windows stays smooth under generative mode."""
    timeout_s = _require_gpu_timeout(client)
    flat = _encode_png(np.full((256, 256, 3), 185, dtype=np.uint8))
    out = _run_enhance_png(
        client,
        job_poller,
        flat,
        "flat_gen.png",
        {"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), dtype=np.float32)
    assert float(np.std(arr)) < 25.0, f"Uniform wall produced excessive spatial blobs: std={np.std(arr):.2f}"


@pytest.mark.gpu
def test_t2_f4_packed_text_blocks_generative_no_freeze_crash(client, job_poller):
    """Image packed with 20+ text blocks runs cleanly without OCR/DINO crash or deadlock."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((512, 512, 3), 230, dtype=np.uint8)
    for i in range(20):
        x = 20 + (i % 4) * 120
        y = 45 + (i // 4) * 90
        cv2.putText(bgr, f"LBL{i:02d}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "packed_text.png",
        {"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(enc.tobytes(), out)
    assert geom["sobel_corr"] >= 0.70, geom


@pytest.mark.gpu
def test_t2_f4_preserve_geometry_false_vs_true_generative_boundary(
    client, synthetic_text_tv_scene_png, job_poller
):
    """Both preserve_geometry=true and false succeed in generative mode, with true keeping Sobel >= 0.70."""
    timeout_s = _require_gpu_timeout(client)
    for pg in ("true", "false"):
        out = _run_enhance_png(
            client,
            job_poller,
            synthetic_text_tv_scene_png,
            f"pg_{pg}.png",
            {
                "render_mode": "generative",
                "generative_strength": "0.48",
                "preserve_geometry": pg,
                "upscale_factor": "1",
            },
            timeout=timeout_s,
        )
        if pg == "true":
            geom = compute_geometry_preservation(synthetic_text_tv_scene_png, out)
            assert geom["sobel_corr"] >= 0.70, geom


@pytest.mark.gpu
def test_t2_f5_no_led_markers_generative_no_hallucinated_strips(client, synthetic_bloom_png, job_poller):
    """Generative render on scene with zero yellow LED guides produces zero yellow markers or LED hallucinations."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_bloom_png,
        "gen_no_led.png",
        {"render_mode": "generative", "generative_strength": "0.48", "led_strips": "true", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    led = compute_led_continuity_metrics(synthetic_bloom_png, out)
    assert led["yellow_px"] == 0.0, led
    assert led["out_yellow_px"] == 0.0, led
    assert led["out_spurious_streak_px"] == 0.0, led


@pytest.mark.gpu
@pytest.mark.relighting
def test_t2_f6_non_fireplace_dark_cabinet_not_ignited(client, synthetic_oak_cabinet_views, job_poller):
    """Dark brown cabinet niche (not a fireplace) does not falsely ignite orange flames."""
    timeout_s = _require_gpu_timeout(client)
    _, view_b = synthetic_oak_cabinet_views
    out = _run_enhance_png(
        client,
        job_poller,
        view_b,
        "dark_niche.png",
        {"render_mode": "generative", "generative_strength": "0.48", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    niche_stats = compute_lab_chroma_stats(out, mask_slice=(slice(150, 360), slice(275, 340)))
    assert niche_stats["a"] < 145.0, f"Dark cabinet niche falsely ignited orange: {niche_stats}"
    relit = compute_relighting_metrics(
        view_b, out, exclude_rois=[(150.0 / 512.0, 360.0 / 512.0, 275.0 / 512.0, 340.0 / 512.0)]
    )
    assert relit["dL_low_spread"] >= 2.0 or abs(relit["d_r_minus_b"]) >= 4.0, relit


@pytest.mark.gpu
@pytest.mark.relighting
def test_t2_f7_extreme_strength_one_keeps_structural_edges(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """Extreme generative_strength=1.0 maintains structural Sobel correlation >= 0.55."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "str1.png",
        {"render_mode": "generative", "generative_strength": "1.0", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(
        synthetic_spot_fireplace_scene_png, out, exclude_rois=[SPOT_FIREPLACE_ROIS["firebox"]]
    )
    assert geom["sobel_corr"] >= 0.55, geom
    relit = compute_relighting_metrics(
        synthetic_spot_fireplace_scene_png, out, exclude_rois=[SPOT_FIREPLACE_ROIS["firebox"]]
    )
    assert relit["dL_low_spread"] >= 2.0, relit


@pytest.mark.gpu
def test_t2_f8_latent_object_lock_false_vs_true_boundary(
    client, synthetic_text_tv_scene_png, job_poller
):
    """latent_object_lock=true and false both return valid PNGs; true keeps TV edges >= 0.72."""
    timeout_s = _require_gpu_timeout(client)
    for lock_val in ("true", "false"):
        out = _run_enhance_png(
            client,
            job_poller,
            synthetic_text_tv_scene_png,
            f"lock_{lock_val}.png",
            {
                "render_mode": "generative",
                "generative_strength": "0.55",
                "latent_object_lock": lock_val,
                "upscale_factor": "1",
            },
            timeout=timeout_s,
        )
        if lock_val == "true":
            tv = compute_roi_preservation(synthetic_text_tv_scene_png, out, roi=TEXT_TV_ROIS["tv_screen"])
            assert tv["edge_f1_d1"] >= 0.72, tv


@pytest.mark.gpu
def test_t2_f9_neutral_gray_wall_no_yellowing_under_generative(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """Unlit neutral gray wall avoids severe jaundice/green shift under warm relighting."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "neutral_wall_gen.png",
        {"render_mode": "generative", "generative_strength": "0.48", "scene_profile": "auto", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(
        synthetic_spot_fireplace_scene_png, out, exclude_rois=[SPOT_FIREPLACE_ROIS["firebox"]]
    )
    assert geom["dE_ab_norm"] <= R2_THRESHOLDS["dE_ab_norm_max"], geom


@pytest.mark.gpu
@pytest.mark.slow
def test_t2_f10_odd_dimensions_4k_generative_no_oom(
    client, synthetic_odd_image, job_poller, vram_monitor
):
    """Odd-dimension 1001x733 input with generative + 4x Real-ESRGAN produces exact (4004, 2932) PNG safely."""
    timeout_s = _require_gpu_timeout(client)
    with vram_monitor() as mon:
        out = _run_enhance_png(
            client,
            job_poller,
            synthetic_odd_image,
            "odd_4k.jpg",
            {
                "render_mode": "generative",
                "generative_strength": "0.48",
                "upscale_factor": "4",
                "tile_size": "256",
            },
            timeout=timeout_s,
        )
    assert Image.open(io.BytesIO(out)).size == (4004, 2932)
    mon.assert_within_budget(max_nvidia_mb=5632.0, max_delta_nvidia_mb=4600.0, max_torch_mb=4600.0)


# ============================================================================
# Group C: Additional Tier 2 Boundaries Closing F1-F14 (>=5 per Feature)
# ============================================================================

def test_t2_f1_download_models_verify_only_empty_hf_home_fails(tmp_path):
    """F1: download_models.py --verify-only fails cleanly (exit 1) when HF_HOME is empty."""
    empty_hf = tmp_path / "empty_hf"
    empty_hf.mkdir()
    env = dict(os.environ)
    env["HF_HOME"] = str(empty_hf)
    env["HUGGINGFACE_HUB_CACHE"] = str(empty_hf)
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "backend/scripts/download_models.py", "--verify-only"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert proc.returncode != 0, f"Expected non-zero exit on empty HF_HOME:\n{proc.stdout}"
    assert "[MISSING]" in proc.stdout


@pytest.mark.offline
def test_t2_f1_offline_socket_hook_blocks_connect_ex_and_getaddrinfo(offline_audited_server):
    """F1: Offline socket-audit hook blocks socket.connect_ex and socket.getaddrinfo to non-loopback targets."""
    hook_dir = offline_audited_server.log_path.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(hook_dir)
    env["PROMOB_SOCKET_AUDIT_LOG"] = str(offline_audited_server.log_path)

    for stmt in (
        "import socket; s=socket.socket(); s.connect_ex(('93.184.216.34', 80))",
        "import socket; socket.getaddrinfo('example.com', 80)",
    ):
        proc = subprocess.run(
            [sys.executable, "-c", stmt],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        assert proc.returncode != 0, f"Expected PermissionError for {stmt}"
        assert "PermissionError" in proc.stderr or "Offline socket audit blocked" in proc.stderr, proc.stderr

    events = offline_audited_server.get_outbound_attempts()
    kinds = {ev.get("kind") for ev in events}
    assert "socket.connect_ex" in kinds and "socket.getaddrinfo" in kinds, events


def test_t2_f2_bench_relight_candidates_invalid_flag_rejected():
    """F2: bench_relight_candidates.py rejects unknown or malformed CLI flags with non-zero exit."""
    for args in (["--bogus-option"], ["--steps", "not-an-int"]):
        proc = subprocess.run(
            [sys.executable, "backend/scripts/bench_relight_candidates.py", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
        assert proc.returncode != 0, f"Expected non-zero exit for args={args}"


def test_t2_f2_bench_realismo_invalid_strength_arg_rejected():
    """F2: bench_realismo.py rejects non-numeric --strengths values with non-zero exit."""
    proc = subprocess.run(
        [sys.executable, "backend/scripts/bench_realismo.py", "--strengths", "invalid_float"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )
    assert proc.returncode != 0


def test_t2_f2_enhance_malformed_input_rejects_synchronously_without_stalled_job(client, synthetic_corrupt_bytes):
    """F2: Corrupt image submitted to /api/enhance fails fast with HTTP 400 rather than hanging a job."""
    resp = client.post(
        "/api/enhance",
        files={"file": ("corrupt.jpg", synthetic_corrupt_bytes, "image/jpeg")},
        data={"render_mode": "classic", "upscale_factor": "1"},
    )
    assert resp.status_code == 400, f"Expected HTTP 400 on corrupt upload, got {resp.status_code}: {resp.text}"


def test_t2_f10_small_tile_size_128_4x_classic_odd_dimensions_boundary(client, job_poller):
    """F10: Small tile_size=128 with 4x upscale on odd 257x193 image produces exact 1028x772 PNG."""
    odd_rgb = np.full((193, 257, 3), 165, dtype=np.uint8)
    odd_rgb[40:150, 60:190] = (120, 145, 175)
    raw = _encode_png(odd_rgb)
    out = _run_enhance_png(
        client,
        job_poller,
        raw,
        "tile128.png",
        {"render_mode": "classic", "upscale_factor": "4", "tile_size": "128"},
        timeout=90.0,
    )
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.size == (1028, 772)


def test_t2_f10_tiny_16x16_4x_upscale_boundary(client, synthetic_tiny_image, job_poller):
    """F10: Tiny 16x16 input with upscale_factor=4 and tile_size=128 completes as exact 64x64 PNG."""
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_tiny_image,
        "tiny4x.jpg",
        {"render_mode": "classic", "upscale_factor": "4", "tile_size": "128"},
        timeout=60.0,
    )
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.size == (64, 64)


def test_t2_f11_4x_upscale_tiny_and_wide_banner_exact_dimensions(client, job_poller):
    """F11: Wide banner (512x32) 4x upscale produces exact (2048, 128) lossless PNG without clipping."""
    banner = _encode_png(np.full((32, 512, 3), 160, dtype=np.uint8))
    out = _run_enhance_png(
        client,
        job_poller,
        banner,
        "banner4x.png",
        {"render_mode": "classic", "upscale_factor": "4", "tile_size": "256"},
        timeout=90.0,
    )
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.size == (2048, 128)


def test_t2_f11_photo_finish_ignored_in_classic_mode(client, synthetic_png, job_poller):
    """F11: In classic mode, photo_finish=0.0 and photo_finish=1.0 yield identical pixel output."""
    common = {"render_mode": "classic", "upscale_factor": "1", "temperature": "0.1"}
    out_0 = _run_enhance_png(client, job_poller, synthetic_png, "c0.png", {**common, "photo_finish": "0.0"})
    out_1 = _run_enhance_png(client, job_poller, synthetic_png, "c1.png", {**common, "photo_finish": "1.0"})
    arr_0 = np.asarray(Image.open(io.BytesIO(out_0)).convert("RGB"))
    arr_1 = np.asarray(Image.open(io.BytesIO(out_1)).convert("RGB"))
    assert np.array_equal(arr_0, arr_1), "photo_finish unexpectedly altered classic mode output"


@pytest.mark.browser
def test_t2_f13_browser_enhance_disabled_before_upload(cdp_browser):
    """F13: Before uploading an image, Dropzone is shown and Enhance/Download actions cannot run."""
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    state = cdp_browser.evaluate(
        """
        (() => {
          const btns = Array.from(document.querySelectorAll('button'));
          const enh = btns.find(b => (b.innerText || '').includes('Aprimorar em 4K'));
          const dl = btns.find(b => (b.innerText || '').includes('Download 4K (PNG)'));
          return {
            hasEnhance: !!enh,
            enhanceDisabled: enh ? !!enh.disabled : null,
            hasDownload: !!dl,
            body: document.body ? document.body.innerText : ''
          };
        })()
        """
    )
    assert "Carregar Amostra Promob" in state["body"]
    assert (not state["hasEnhance"]) or (state["enhanceDisabled"] is True), state
    assert state["hasDownload"] is False, state


@pytest.mark.browser
def test_t2_f13_browser_zoom_bounds_disable_at_min_and_max(cdp_browser):
    """F13: StatusBar zoom clamps cleanly at 50% min and 300% max and resets to 100%."""
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    cdp_browser.click_text("Carregar Amostra Promob")
    cdp_browser.wait_for_text("100%", timeout=15.0)

    # Zoom down to 50% (2 clicks from 100%)
    for _ in range(2):
        cdp_browser.evaluate("document.querySelector('button[title=\"Reduzir Zoom\"]').click()")
    cdp_browser.wait_for_text("50%", timeout=5.0)
    assert cdp_browser.evaluate("document.querySelector('button[title=\"Reduzir Zoom\"]').disabled") is True

    # Zoom up to 300% (10 clicks from 50%)
    for _ in range(10):
        cdp_browser.evaluate("document.querySelector('button[title=\"Aumentar Zoom\"]').click()")
    cdp_browser.wait_for_text("300%", timeout=5.0)
    assert cdp_browser.evaluate("document.querySelector('button[title=\"Aumentar Zoom\"]').disabled") is True

    # Reset zoom to 100%
    cdp_browser.evaluate("document.querySelector('button[title=\"Ajustar à Janela (100%)\"]').click()")
    cdp_browser.wait_for_text("100%", timeout=5.0)


@pytest.mark.browser
def test_t2_f13_browser_header_hardware_badge_no_nan(cdp_browser):
    """F13: Header GPU/VRAM badge stays valid and non-empty across viewport/load transitions."""
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("GB VRAM", timeout=15.0)
    header_text = cdp_browser.evaluate("document.querySelector('header') ? document.querySelector('header').innerText : ''")
    assert "GB VRAM" in header_text and "NaN" not in header_text, header_text


# --- Additional GPU Tier 2 Boundaries closing F3, F4, F6, F7, F8, F9, F10, F11 ---

@pytest.mark.gpu
def test_t2_f3_pure_white_image_generative_no_nan_or_blowout(client, job_poller):
    """F3: All-255 pure white PNG in generative mode returns finite PNG without NaN or inversion."""
    timeout_s = _require_gpu_timeout(client)
    white = _encode_png(np.full((256, 256, 3), 255, dtype=np.uint8))
    out = _run_enhance_png(
        client,
        job_poller,
        white,
        "white_gen.png",
        {"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), dtype=np.float32)
    assert arr.shape == (256, 256, 3) and not np.isnan(arr).any()
    assert float(np.mean(arr)) > 150.0, f"Pure white input inverted dark: mean={arr.mean():.1f}"


@pytest.mark.gpu
def test_t2_f4_rotated_or_edge_clipped_text_no_crash(client, job_poller):
    """F4: Edge-clipped / rotated sign text does not crash OCR/DINO/SAM and preserves Sobel >= 0.70."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((384, 384, 3), 210, dtype=np.uint8)
    cv2.putText(bgr, "PROMOB EDGE", (-18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2)
    cv2.putText(bgr, "STUDIO", (280, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 2)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "edge_text.png",
        {"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(enc.tobytes(), out)
    assert geom["sobel_corr"] >= 0.70, geom


@pytest.mark.gpu
@pytest.mark.relighting
def test_t2_f4_dark_tv_with_logo_vs_fireplace_disambiguation(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F4: Dark TV with white text next to dark firebox is protected (not ignited as a fireplace)."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "tv_vs_fp.png",
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
        out,
        roi=SPOT_FIREPLACE_ROIS["tv_sign"],
        expected_text="ELLEGANZA",
    )
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv
    fp = compute_fireplace_metrics(
        synthetic_spot_fireplace_scene_png,
        out,
        firebox_roi=SPOT_FIREPLACE_ROIS["firebox"],
        spill_roi=SPOT_FIREPLACE_ROIS["hearth_spill"],
    )
    assert fp["fp_dchr"] >= 8.0 and fp["fp_L_rng_ratio"] >= 1.4, fp


@pytest.mark.gpu
@pytest.mark.relighting
def test_t2_f6_already_lit_fireplace_not_overblown(client, chatgpt_reference_pair, job_poller):
    """F6: Already-lit fireplace target does not invert or clip solidly white under generative mode."""
    timeout_s = _require_gpu_timeout(client)
    lit_after = chatgpt_reference_pair["after"]
    out = _run_enhance_png(
        client,
        job_poller,
        lit_after,
        "already_lit.webp",
        {"render_mode": "generative", "generative_strength": "0.40", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), dtype=np.float32)
    assert not np.isnan(arr).any() and float(np.mean(arr)) < 245.0
    in_arr = np.asarray(Image.open(io.BytesIO(lit_after)).convert("RGB"), dtype=np.float32)
    mae = float(np.mean(np.abs(arr - in_arr)))
    assert mae > 1.0, f"Expected non-trivial relighting pass without blowout, got mae={mae}"
    fp = compute_fireplace_metrics(lit_after, out)
    assert fp["fp_chroma_out"] >= 15.0, fp


@pytest.mark.gpu
def test_t2_f6_tiny_sub_24px_dark_cutout_not_ignited(client, job_poller):
    """F6: Tiny <24px dark handle cutout on cabinetry is not misclassified as a fireplace."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((256, 256, 3), 185, dtype=np.uint8)
    bgr[120:136, 120:136] = (18, 18, 20)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "sub24_dark.png",
        {"render_mode": "generative", "generative_strength": "0.48", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    stats = compute_lab_chroma_stats(out, mask_slice=(slice(120, 136), slice(120, 136)))
    assert stats["a"] < 145.0, f"Sub-24px dark handle falsely ignited orange: {stats}"


@pytest.mark.gpu
def test_t2_f7_extreme_aspect_ratios_generative_active_gpu(
    client, synthetic_banner_image, synthetic_tall_image, job_poller
):
    """F7: 1920x64 banner and 64x1920 tall sliver at generative_strength=0.48 return exact sizes without NaN."""
    timeout_s = _require_gpu_timeout(client)
    for name, payload, expected in (
        ("banner_gpu.jpg", synthetic_banner_image, (1920, 64)),
        ("tall_gpu.jpg", synthetic_tall_image, (64, 1920)),
    ):
        out = _run_enhance_png(
            client,
            job_poller,
            payload,
            name,
            {"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
            timeout=timeout_s,
        )
        img = Image.open(io.BytesIO(out))
        assert img.size == expected
        assert not np.isnan(np.asarray(img.convert("RGB"), dtype=np.float32)).any()


@pytest.mark.gpu
def test_t2_f7_near_zero_strength_005_bounded_delta(client, synthetic_text_tv_scene_png, job_poller):
    """F7: Near-zero generative_strength=0.05 preserves high Sobel (>= 0.80) and SSIM (>= 0.80)."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "str_005.png",
        {"render_mode": "generative", "generative_strength": "0.05", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    geom = compute_geometry_preservation(synthetic_text_tv_scene_png, out)
    assert geom["sobel_corr"] >= 0.80 and geom["global_ssim_L"] >= 0.80, geom


@pytest.mark.gpu
def test_t2_f8_luminance_modulated_protection_dark_and_bright_signs(
    client, synthetic_text_tv_scene_png, job_poller
):
    """F8: Both white-on-dark TV and dark-on-white framed sign preserve OCR & Edge F1 simultaneously."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "dark_bright_ocr.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    for roi_name, txt in (("tv_screen", "PROMOB"), ("framed_sign", "WOLF")):
        m = compute_roi_preservation(
            synthetic_text_tv_scene_png, out, roi=TEXT_TV_ROIS[roi_name], expected_text=txt
        )
        assert m["edge_f1_d1"] >= 0.72, (roi_name, m)
        if m["ocr_available"]:
            assert m["ocr_matched"] is True, (roi_name, m)


@pytest.mark.gpu
def test_t2_f8_preserve_geometry_true_vs_false_edge_preservation_delta(
    client, synthetic_text_tv_scene_png, job_poller
):
    """F8: preserve_geometry=true retains Sobel >= 0.72 and matches/exceeds preserve_geometry=false."""
    timeout_s = _require_gpu_timeout(client)
    out_t = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "pg_t.png",
        {"render_mode": "generative", "generative_strength": "0.65", "preserve_geometry": "true", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    out_f = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "pg_f.png",
        {"render_mode": "generative", "generative_strength": "0.65", "preserve_geometry": "false", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    gt = compute_geometry_preservation(synthetic_text_tv_scene_png, out_t)
    gf = compute_geometry_preservation(synthetic_text_tv_scene_png, out_f)
    assert gt["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"], gt
    assert gt["sobel_corr"] >= gf["sobel_corr"] - 0.05, (gt, gf)


@pytest.mark.gpu
def test_t2_f8_latent_object_lock_true_preserves_tv_roi_strictly(
    client, synthetic_text_tv_scene_png, job_poller
):
    """F8: latent_object_lock=true keeps TV screen ZNCC >= 0.72 and Edge F1 >= 0.75 at strength=0.65."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "lock_strict.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.65",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    tv = compute_roi_preservation(synthetic_text_tv_scene_png, out, roi=TEXT_TV_ROIS["tv_screen"])
    assert tv["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], tv
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv


@pytest.mark.gpu
def test_t2_f9_pure_white_wall_non_yellowing_under_generative(client, job_poller):
    """F9: Pure white bathroom wall (240,240,240) avoids severe yellow cast under generative relighting."""
    timeout_s = _require_gpu_timeout(client)
    white_room = _encode_png(np.full((256, 256, 3), 240, dtype=np.uint8))
    out = _run_enhance_png(
        client,
        job_poller,
        white_room,
        "pure_white_wall.png",
        {"render_mode": "generative", "generative_strength": "0.48", "scene_profile": "banheiro", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    stats = compute_lab_chroma_stats(out)
    assert abs(stats["b"] - 128.0) <= 12.0, f"Pure white wall yellowed excessively: {stats}"


@pytest.mark.gpu
def test_t2_f9_dark_charcoal_cabinetry_not_crushed_or_shifted(client, job_poller):
    """F9: Dark charcoal (42,44,48) cabinetry avoids crushing to pure black or strong color cast."""
    timeout_s = _require_gpu_timeout(client)
    bgr = np.full((256, 256, 3), (185, 185, 185), dtype=np.uint8)
    bgr[60:220, 60:196] = (48, 44, 42)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    out = _run_enhance_png(
        client,
        job_poller,
        enc.tobytes(),
        "charcoal.png",
        {"render_mode": "generative", "generative_strength": "0.48", "scene_profile": "quarto_sala", "upscale_factor": "1"},
        timeout=timeout_s,
    )
    stats = compute_lab_chroma_stats(out, mask_slice=(slice(80, 200), slice(80, 176)))
    assert stats["L"] > 8.0, f"Dark charcoal crushed: {stats}"
    assert abs(stats["a"] - 128.0) < 12.0 and abs(stats["b"] - 128.0) < 14.0, stats


@pytest.mark.gpu
def test_t2_f10_concurrent_generative_jobs_serialized_within_vram_budget(
    client, synthetic_png, job_poller, vram_monitor
):
    """F10: Two concurrent generative /api/enhance submissions serialize safely within 6 GB VRAM budget."""
    timeout_s = _require_gpu_timeout(client)

    def _submit(tag: str) -> str:
        r = client.post(
            "/api/enhance",
            files={"file": (f"conc_{tag}.png", synthetic_png, "image/png")},
            data={"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
        )
        assert r.status_code in (200, 202), r.text
        return r.json()["job_id"]

    with vram_monitor() as mon:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_submit, "a")
            f2 = pool.submit(_submit, "b")
            j1, j2 = f1.result(), f2.result()
        assert job_poller(client, j1, timeout_seconds=timeout_s).get("status") in ("completed", "finished", "success")
        assert job_poller(client, j2, timeout_seconds=timeout_s).get("status") in ("completed", "finished", "success")

    mon.assert_within_budget(max_nvidia_mb=5632.0, max_delta_nvidia_mb=4600.0, max_torch_mb=4600.0)


@pytest.mark.gpu
def test_t2_f11_photo_finish_1_0_generative_no_white_or_black_clipping(
    client, promob_reference_image, job_poller
):
    """F11: Maximum photo_finish=1.0 in generative mode avoids extreme white/black pixel clipping."""
    timeout_s = _require_gpu_timeout(client)
    out = _run_enhance_png(
        client,
        job_poller,
        promob_reference_image,
        "pf_max.jpg",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "photo_finish": "1.0",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"))
    frac_black = float(np.mean(np.all(arr == 0, axis=2)))
    frac_white = float(np.mean(np.all(arr == 255, axis=2)))
    assert frac_black < 0.05 and frac_white < 0.05, (frac_black, frac_white)
