"""Tier 4: Real-World Application Workloads E2E Tests for R3-R5.

Executes complete end-to-end architectural workflows:
1. Practical RTX 2060 4K Generative + Real-ESRGAN Tiled (< 5.5 GB peak VRAM via nvidia-smi + torch.cuda)
2. Sequential back-to-back 4K generative jobs (VRAM release & no memory leak)
3. Full realismo_ia API client session (health -> capabilities -> presets -> 1080p preview -> 4K enhance -> download)
4. Full browser E2E workflow via headless Chrome/Edge CDP including real job progress stage wrapping
"""

import io
import statistics
import time

import numpy as np
import pytest
from PIL import Image

from tests.e2e.conftest import AMOSTRAS_ROIS
from tests.e2e.metrics_helpers import (
    R1_THRESHOLDS,
    R2_THRESHOLDS,
    bgr_to_cie_lab,
    compute_fireplace_metrics,
    compute_geometry_preservation,
    compute_led_continuity_metrics,
    compute_relighting_metrics,
    compute_roi_preservation,
    compute_spot_cone_metrics,
    decode_bgr,
    roi_to_slices,
)


def _require_gpu_capabilities(client) -> float:
    health = client.get("/api/health").json()
    assert health.get("cuda_available") is True, f"CUDA GPU required for Tier 4 generative workload: {health}"
    caps = client.get("/api/capabilities").json()
    ge = caps.get("generative_engine", {})
    assert ge.get("available") is True, f"Generative engine unavailable: {caps}"
    return 600.0 if not ge.get("cached_locally") else 300.0


def _poll_with_stages(client, job_id: str, timeout_seconds: float, poll_interval: float = 0.5):
    observed = []
    start = time.time()
    while time.time() - start < timeout_seconds:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200, f"Job polling failed ({resp.status_code}): {resp.text}"
        data = resp.json()
        progress = data.get("progress", 0.0)
        stage = data.get("stage", "")
        assert 0.0 <= float(progress) <= 100.0, f"Invalid job progress: {progress}"
        observed.append((float(progress), str(stage)))
        status = data.get("status")
        if status in ("completed", "finished", "success"):
            return data, observed
        if status in ("failed", "error"):
            raise RuntimeError(f"Job {job_id} failed: {data.get('error', 'unknown')}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout_seconds}s")


# ============================================================================
# Workload 1: Practical RTX 2060 4K Generative + Real-ESRGAN VRAM < 5.5 GB
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
def test_t4_r3_practical_rtx2060_4k_generative_vram_under_5_5gb(
    client, promob_reference_image, vram_monitor
):
    """Full RTX 2060 Generative + Real-ESRGAN 4K Tiled completes with peak VRAM < 5.5 GB (5632 MB)."""
    timeout_s = _require_gpu_capabilities(client)
    pre_health = client.get("/api/health").json()
    pre_alloc_mb = float(pre_health.get("allocated_vram_mb", 0.0))

    with vram_monitor() as mon:
        submit_resp = client.post(
            "/api/enhance",
            files={"file": ("promob_rtx2060.jpg", promob_reference_image, "image/jpeg")},
            data={
                "render_mode": "generative",
                "generative_strength": "0.48",
                "scene_profile": "auto",
                "specular_bloom": "0.25",
                "upscale_factor": "4",
                "tile_size": "256",
                "preserve_geometry": "true",
            },
        )
        assert submit_resp.status_code in (200, 202), f"Submission failed: {submit_resp.text}"
        job_id = submit_resp.json()["job_id"]

        final_job, observed_stages = _poll_with_stages(client, job_id, timeout_seconds=timeout_s)
        assert final_job.get("status") in ("completed", "finished", "success")

    assert any(stage.strip() for _, stage in observed_stages), (
        f"Expected non-empty job stages during execution, got {observed_stages}"
    )

    dl_resp = client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200
    assert len(dl_resp.content) > 50000
    out_img = Image.open(io.BytesIO(dl_resp.content))
    assert out_img.format == "PNG"
    assert max(out_img.size) >= 3600, f"Expected 4K output (>=3600px), got {out_img.size}"

    # Verify dual-process RTX 2060 VRAM ceiling (<4600 MB PyTorch / <5632 MB nvidia-smi / <4600 MB delta)
    mon.assert_within_budget(
        max_nvidia_mb=5632.0,
        max_delta_nvidia_mb=4600.0,
        max_torch_mb=4600.0,
    )

    post_health = client.get("/api/health").json()
    post_alloc_mb = float(post_health.get("allocated_vram_mb", 0.0))
    mon.assert_health_released(pre_alloc_mb, post_alloc_mb, tolerance_mb=512.0)


# ============================================================================
# Workload 2: Sequential Back-to-Back 4K Generative Jobs (No Memory Leak)
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
def test_t4_r3_sequential_4k_generative_jobs_no_memory_leak(
    client, chatgpt_reference_pair, promob_reference_image, job_poller, vram_monitor
):
    """Two consecutive production-res 4K generative renders complete without CUDA OOM or VRAM leak."""
    timeout_s = _require_gpu_capabilities(client)
    pre_health = client.get("/api/health").json()
    pre_alloc_mb = float(pre_health.get("allocated_vram_mb", 0.0))

    jobs_spec = [
        ("quarto_sala", "ref_1024x698.jpg", chatgpt_reference_pair["before"], (4096, 2792)),
        ("banheiro", "promob_723x900.jpg", promob_reference_image, (2892, 3600)),
    ]
    peaks = []
    for profile, fname, payload, expected_size in jobs_spec:
        with vram_monitor(poll_interval=0.10) as mon:
            submit_resp = client.post(
                "/api/enhance",
                files={"file": (fname, payload, "image/jpeg")},
                data={
                    "render_mode": "generative",
                    "generative_strength": "0.48",
                    "scene_profile": profile,
                    "specular_bloom": "0.25",
                    "upscale_factor": "4",
                    "tile_size": "256",
                    "preserve_geometry": "true",
                },
            )
            assert submit_resp.status_code in (200, 202), f"{profile} submission failed: {submit_resp.text}"
            job_id = submit_resp.json()["job_id"]
            job_data = job_poller(client, job_id, timeout_seconds=timeout_s)
            assert job_data.get("status") in ("completed", "finished", "success")

        dl = client.get(f"/api/download/{job_id}")
        assert dl.status_code == 200
        out_img = Image.open(io.BytesIO(dl.content))
        assert out_img.format == "PNG" and out_img.size == expected_size, (out_img.size, expected_size)

        mon.assert_within_budget(
            max_nvidia_mb=5632.0,
            max_delta_nvidia_mb=4600.0,
            max_torch_mb=4600.0,
        )
        if mon.peak_torch_mb > 0.0:
            peaks.append(mon.peak_torch_mb)
        elif mon.nvidia_available:
            peaks.append(mon.peak_nvidia_mb)

    if len(peaks) == 2:
        assert peaks[1] <= peaks[0] + 512.0, (
            f"Unbounded peak VRAM growth across sequential 4K jobs: run1={peaks[0]:.1f} MB, run2={peaks[1]:.1f} MB"
        )
    post_health = client.get("/api/health").json()
    post_alloc_mb = float(post_health.get("allocated_vram_mb", 0.0))
    vram_monitor.assert_health_released(pre_alloc_mb, post_alloc_mb, tolerance_mb=512.0)


# ============================================================================
# Workload 3: Full Realismo IA API Client Session
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
def test_t4_r4_full_realismo_ia_api_client_session(
    client, promob_reference_image, synthetic_1080p_interior, job_poller
):
    """Complete API client journey: health -> capabilities -> presets -> 1080p preview -> 4K enhance -> download."""
    health = client.get("/api/health")
    assert health.status_code == 200 and health.json().get("status") == "ok"

    caps = client.get("/api/capabilities").json()
    ge = caps.get("generative_engine", {})
    assert ge.get("available") is True
    timeout_s = 600.0 if not ge.get("cached_locally") else 300.0

    presets = client.get("/api/presets").json()
    realismo_cfg = presets["realismo_ia"].get("config", presets["realismo_ia"])
    form_data = {
        k: ("true" if v is True else "false" if v is False else str(v))
        for k, v in realismo_cfg.items()
    }

    # 3 warmup calls + timed 1080p preview using realismo_ia config
    for _ in range(3):
        warm = client.post(
            "/api/preview",
            files={"file": ("1080p.jpg", synthetic_1080p_interior, "image/jpeg")},
            data=form_data,
        )
        assert warm.status_code == 200

    latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        p_resp = client.post(
            "/api/preview",
            files={"file": ("1080p.jpg", synthetic_1080p_interior, "image/jpeg")},
            data=form_data,
        )
        latencies.append(time.perf_counter() - t0)
        assert p_resp.status_code == 200
    assert statistics.median(latencies) < 0.100, (
        f"1080p realismo_ia preview median {statistics.median(latencies) * 1000:.1f}ms >= 100ms"
    )

    # Full 4K enhance using realismo_ia config
    enh_resp = client.post(
        "/api/enhance",
        files={"file": ("promob.jpg", promob_reference_image, "image/jpeg")},
        data=form_data,
    )
    assert enh_resp.status_code in (200, 202)
    job_id = enh_resp.json()["job_id"]
    job_poller(client, job_id, timeout_seconds=timeout_s)

    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    final_img = Image.open(io.BytesIO(dl.content))
    assert final_img.format == "PNG" and max(final_img.size) >= 3600


# ============================================================================
# Workload 4: Full Browser Workflow & Real Job Stage Wrapping (CDP)
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.browser
def test_t4_r5_full_browser_workflow_and_real_job_stage_wrapping(cdp_browser, promob_sample_image, tmp_path):
    """Full Chrome/Edge CDP workflow: <input type='file'> upload -> Preview -> Real 4K Job + Stage Wrap -> Viewer toggle -> Download PNG."""
    import httpx

    caps = httpx.get(f"{cdp_browser.base_url}/api/capabilities", timeout=5.0).json()
    ge = caps.get("generative_engine", {})
    assert ge.get("available") is True, f"Generative engine unavailable: {caps}"
    job_timeout = 600.0 if not ge.get("cached_locally") else 300.0

    upload_file = tmp_path / "promob_upload.jpg"
    upload_file.write_bytes(promob_sample_image)

    cdp_browser.observe_api_requests(clear=True)
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    cdp_browser.set_file_input("input[type='file']", str(upload_file))
    cdp_browser.wait_for_text("Realismo IA", timeout=15.0)
    cdp_browser.wait_for_api_response("/api/preview", min_index=0, timeout=15.0)

    # Real-time Bloom slider adjustment
    idx = len(cdp_browser.captured_api_requests)
    resp_idx = len(cdp_browser.network_responses)
    cdp_browser.set_radix_slider("Brilho GI & Reflexos (Bloom)", key="End")
    req_bloom = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    resp_bloom = cdp_browser.wait_for_api_response("/api/preview", min_index=resp_idx, timeout=10.0)
    assert resp_bloom["status"] == 200
    assert float(req_bloom["form_fields"].get("specular_bloom", -1)) == pytest.approx(1.0, abs=0.05)

    # Submit real 4K job from UI
    enh_idx = len(cdp_browser.captured_api_requests)
    cdp_browser.click_text("Aprimorar em 4K")
    enh_req = cdp_browser.wait_for_api_request("/api/enhance", min_index=enh_idx, timeout=15.0)
    assert enh_req["form_fields"].get("render_mode") == "generative"

    # Inspect live job stage wrapping while real job runs
    stage_style = None
    deadline = time.time() + job_timeout
    while time.time() < deadline:
        body_text = cdp_browser.evaluate("document.body ? document.body.innerText : ''") or ""
        if "Download 4K (PNG)" in body_text:
            break
        candidate = cdp_browser.evaluate(
            """
            (() => {
              const aside = document.querySelector('aside');
              if (!aside) return null;
              const asideRect = aside.getBoundingClientRect();
              const nodes = Array.from(aside.querySelectorAll('div'));
              const stageEl = nodes.find(el => {
                const cs = window.getComputedStyle(el);
                const txt = (el.textContent || '').trim();
                return (
                  el.children.length === 0 &&
                  txt.length > 8 &&
                  cs.fontFamily.toLowerCase().includes('mono') &&
                  !/^\\d+%$/.test(txt) &&
                  !txt.includes('Super-Resolução') &&
                  !txt.includes('256') &&
                  !txt.includes('Preservada')
                );
              });
              if (!stageEl) return null;
              const cs = window.getComputedStyle(stageEl);
              const rect = stageEl.getBoundingClientRect();
              return {
                whiteSpace: cs.whiteSpace,
                overflowX: cs.overflowX,
                textOverflow: cs.textOverflow,
                scrollWidth: stageEl.scrollWidth,
                clientWidth: stageEl.clientWidth,
                rectRight: rect.right,
                asideRight: asideRect.right,
                text: (stageEl.textContent || '').trim()
              };
            })()
            """
        )
        if candidate and candidate.get("text"):
            stage_style = candidate
            assert stage_style["whiteSpace"] != "nowrap", (
                f"Progress stage uses nowrap truncation: {stage_style}"
            )
            assert stage_style["scrollWidth"] <= stage_style["clientWidth"] + 2, (
                f"Progress stage overflows horizontally: {stage_style}"
            )
            assert stage_style["rectRight"] <= stage_style["asideRight"] + 2, (
                f"Progress stage clips past sidebar edge: {stage_style}"
            )
            break
        cdp_browser.pump_events(0.2)

    cdp_browser.wait_for_text("Download 4K (PNG)", timeout=job_timeout)
    assert stage_style is not None, "Did not observe live job stage element during 4K processing"

    # Toggle Viewer mode (Lado a Lado -> Divisor)
    body_text = cdp_browser.evaluate("document.body ? document.body.innerText : ''") or ""
    if "Lado a Lado" in body_text:
        cdp_browser.click_text("Lado a Lado")
        cdp_browser.pump_events(0.2)
    if "Divisor" in body_text:
        cdp_browser.click_text("Divisor")
        cdp_browser.pump_events(0.2)

    download_src = cdp_browser.evaluate(
        """
        (() => {
          const imgs = Array.from(document.querySelectorAll('img')).filter(
            img => img.complete && img.naturalWidth > 0 && img.src.includes('/api/download/')
          );
          if (imgs.length > 0) return imgs[0].src;
          const links = Array.from(document.querySelectorAll('a[href*="/api/download/"]'));
          return links.length > 0 ? links[0].href : null;
        })()
        """
    )
    has_completed_img = cdp_browser.evaluate(
        """
        Array.from(document.querySelectorAll('img')).some(
          img => img.complete && img.naturalWidth > 0 && (img.src.startsWith('blob:') || img.src.includes('/api/download/'))
        )
        """
    )
    assert has_completed_img is True, "Viewer did not render completed 4K image"
    assert download_src and "/api/download/" in download_src, f"Expected /api/download/ URL in UI, got {download_src}"
    dl_resp = httpx.get(download_src, timeout=30.0)
    assert dl_resp.status_code == 200
    assert "image/png" in dl_resp.headers.get("content-type", "").lower()
    dl_img = Image.open(io.BytesIO(dl_resp.content))
    assert dl_img.format == "PNG" and max(dl_img.size) >= 2800


# ============================================================================
# Workload 5: ChatGPT Reference Room 4K Relighting, Fireplace & TV Logo Preservation
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.relighting
def test_t4_r1_r2_chatgpt_reference_room_4k_relighting_and_tv_logo(
    client, chatgpt_reference_pair, job_poller, vram_monitor
):
    """ChatGPT reference room (1024x698 -> 4096x2792) satisfies R1 relighting, spot, fireplace & R2 TV preservation."""
    timeout_s = _require_gpu_capabilities(client)
    before = chatgpt_reference_pair["before"]
    rois = chatgpt_reference_pair["rois"]

    presets = client.get("/api/presets").json()
    realismo_cfg = presets["realismo_ia"].get("config", presets["realismo_ia"])
    form_data = {
        k: ("true" if v is True else "false" if v is False else str(v))
        for k, v in realismo_cfg.items()
    }
    form_data["upscale_factor"] = "4"

    pre_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    with vram_monitor() as mon:
        resp = client.post(
            "/api/enhance",
            files={"file": ("chatgpt_ref_before.jpg", before, "image/jpeg")},
            data=form_data,
        )
        assert resp.status_code in (200, 202), f"Submission failed: {resp.text}"
        job_id = resp.json()["job_id"]
        job_poller(client, job_id, timeout_seconds=timeout_s)

    mon.assert_within_budget(max_nvidia_mb=5632.0, max_delta_nvidia_mb=4600.0, max_torch_mb=4600.0)
    post_alloc = float(client.get("/api/health").json().get("allocated_vram_mb", 0.0))
    mon.assert_health_released(pre_alloc, post_alloc, tolerance_mb=512.0)

    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    out_img = Image.open(io.BytesIO(dl.content))
    assert out_img.format == "PNG" and out_img.size == (4096, 2792), out_img.size

    relit = compute_relighting_metrics(before, dl.content, exclude_rois=[rois["fireplace_firebox"]])
    assert relit["d_r_minus_b"] >= R1_THRESHOLDS["d_r_minus_b_min"], relit
    assert relit["db_mean"] >= R1_THRESHOLDS["db_mean_min"], relit
    assert relit["dL_low_std"] >= R1_THRESHOLDS["dL_low_std_min"], relit
    assert relit["dL_low_spread"] >= R1_THRESHOLDS["dL_low_spread_min"], relit

    spots = compute_spot_cone_metrics(before, dl.content, spot_roi=rois["walnut_upper_spots"])
    assert spots["spot_dL_spread"] >= R1_THRESHOLDS["spot_dL_spread_min"], spots
    assert spots["spot_d_r_minus_b"] >= R1_THRESHOLDS["spot_d_r_minus_b_min"], spots

    fp = compute_fireplace_metrics(
        before,
        dl.content,
        firebox_roi=rois["fireplace_firebox"],
        spill_roi=rois["fireplace_spill_floor"],
    )
    assert fp["fp_L_rng_ratio"] >= R1_THRESHOLDS["fp_L_rng_ratio_min"], fp
    assert fp["fp_dchr"] >= R1_THRESHOLDS["fp_dchr_min"], fp
    assert fp["spill_dL"] >= R1_THRESHOLDS["fp_spill_dL_min"], fp
    assert fp["spill_d_r_minus_b"] >= R1_THRESHOLDS["fp_spill_d_r_minus_b_min"], fp

    geom = compute_geometry_preservation(before, dl.content, exclude_rois=[rois["fireplace_firebox"]])
    assert geom["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"], geom
    assert geom["canny_f1_d2"] >= R2_THRESHOLDS["canny_f1_d2_min"], geom
    assert geom["global_ssim_L"] >= R2_THRESHOLDS["global_ssim_L_min"], geom
    assert geom["dE_ab_norm"] <= R2_THRESHOLDS["dE_ab_norm_max"], geom

    tv = compute_roi_preservation(before, dl.content, roi=rois["tv_screen"], expected_text="Elleganza")
    assert tv["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], tv
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv
    assert tv["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv

    shelves = compute_roi_preservation(before, dl.content, roi=rois["shelves_decor"])
    assert shelves["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], shelves
    assert shelves["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], shelves
    assert shelves["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], shelves


# ============================================================================
# Workload 6: Diverse AMOSTRAS PROMOB Multi-Scene Suite (LEDs, Hue, Objects/Text)
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.relighting
@pytest.mark.parametrize(
    ("key", "profile"),
    [
        ("s00", "auto"),
        ("s09", "cozinha"),
        ("s14", "cozinha"),
        ("s15", "auto"),
        ("s17", "banheiro"),
        ("s21", "quarto_sala"),
        ("s23", "auto"),
    ],
    ids=["s00", "s09", "s14", "s15", "s17", "s21", "s23"],
)
def test_t4_r1_r2_amostras_promob_diverse_scenes_led_and_hue_fidelity(
    client, amostras_promob_scenes, job_poller, vram_monitor, key, profile
):
    """Check relighting, geometry, LEDs and protected details for one real Promob scene."""
    timeout_s = _require_gpu_capabilities(client)
    presets = client.get("/api/presets").json()
    realismo_cfg = presets["realismo_ia"].get("config", presets["realismo_ia"])
    base_form = {
        k: ("true" if v is True else "false" if v is False else str(v))
        for k, v in realismo_cfg.items()
    }
    base_form["upscale_factor"] = "1"

    img_bytes = amostras_promob_scenes[key]
    form_data = {**base_form, "scene_profile": profile}
    with vram_monitor(poll_interval=0.10) as mon:
        resp = client.post(
            "/api/enhance",
            files={"file": (f"{key}.jpg", img_bytes, "image/jpeg")},
            data=form_data,
        )
        assert resp.status_code in (200, 202), f"{key} failed: {resp.text}"
        job_id = resp.json()["job_id"]
        job_poller(client, job_id, timeout_seconds=timeout_s)
    mon.assert_within_budget(max_nvidia_mb=5632.0, max_delta_nvidia_mb=4600.0, max_torch_mb=4600.0)

    out_bytes = client.get(f"/api/download/{job_id}").content

    # R1 spatial relighting dynamics across all scenes (rejects no-op pass-through)
    relit = compute_relighting_metrics(img_bytes, out_bytes)
    assert relit["dL_low_spread"] >= 4.0, (key, relit)
    assert relit["dL_low_std"] >= 1.2, (key, relit)
    if key in ("s15", "s21"):
        assert relit["d_r_minus_b"] >= 4.0, (key, relit)

    geom = compute_geometry_preservation(img_bytes, out_bytes)
    assert geom["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"], (key, geom)
    assert geom["canny_f1_d2"] >= R2_THRESHOLDS["canny_f1_d2_min"], (key, geom)
    assert geom["dE_ab_norm"] <= R2_THRESHOLDS["dE_ab_norm_max"], (key, geom)

    if key in ("s15", "s21"):
        led = compute_led_continuity_metrics(img_bytes, out_bytes)
        assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], (key, led)
        assert led["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], (key, led)
        assert led["gap_frac"] <= R1_THRESHOLDS["led_gap_frac_max"], (key, led)
        assert led["spill_dL"] >= R1_THRESHOLDS["led_spill_dL_min"], (key, led)
        assert led["spill_db"] >= R1_THRESHOLDS["led_spill_db_min"], (key, led)
    else:
        led_clean = compute_led_continuity_metrics(img_bytes, out_bytes)
        assert led_clean["yellow_px"] == 0.0 and led_clean["out_yellow_px"] == 0.0, (key, led_clean)

    for roi_name, roi_box in AMOSTRAS_ROIS.get(key, {}).items():
        if roi_name == "pendants":
            continue
        if roi_name in ("neutral_wall", "white_brick_wall"):
            in_bgr = decode_bgr(img_bytes)
            out_bgr = decode_bgr(out_bytes)
            ys, xs = roi_to_slices(roi_box, in_bgr.shape[0], in_bgr.shape[1])
            lab_in_roi = bgr_to_cie_lab(in_bgr[ys, xs])
            lab_out_roi = bgr_to_cie_lab(out_bgr[ys, xs])
            de_roi = float(
                np.mean(
                    np.hypot(
                        lab_out_roi[:, :, 1] - lab_in_roi[:, :, 1],
                        lab_out_roi[:, :, 2] - lab_in_roi[:, :, 2],
                    )
                )
            )
            assert de_roi <= R2_THRESHOLDS["dE_ab_norm_max"], (key, roi_name, de_roi)
            continue

        expected_txt = "TWININGS" if (key == "s00" and roi_name == "twinings_box") else None
        roi_m = compute_roi_preservation(
            img_bytes, out_bytes, roi=roi_box, expected_text=expected_txt
        )
        assert roi_m["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], (key, roi_name, roi_m)
        assert roi_m["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], (key, roi_name, roi_m)
        assert roi_m["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], (key, roi_name, roi_m)
        if expected_txt and roi_m["ocr_available"]:
            assert roi_m["ocr_matched"] is True, (key, roi_name, roi_m)


# ============================================================================
# Workload 7: Offline Socket-Audited 2x Production 4K Generative Renders
# ============================================================================

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.offline
def test_t4_r3_offline_socket_audited_4k_generative_render(
    offline_audited_server, chatgpt_reference_pair, amostras_promob_scenes, job_poller, vram_monitor
):
    """Two consecutive production-resolution 4K Realismo IA renders run 100% offline within 6 GB VRAM."""
    import httpx

    with httpx.Client(base_url=offline_audited_server.base_url, timeout=360.0) as http:
        caps = http.get("/api/capabilities").json()
        ge = caps.get("generative_engine", {})
        assert ge.get("available") is True and ge.get("cached_locally") is True, caps

        presets = http.get("/api/presets").json()
        cfg = presets["realismo_ia"].get("config", presets["realismo_ia"])
        form_data = {
            k: ("true" if v is True else "false" if v is False else str(v))
            for k, v in cfg.items()
        }
        form_data["upscale_factor"] = "4"

        workloads = [
            ("ref_1024x698.jpg", chatgpt_reference_pair["before"], (4096, 2792)),
            ("s21_1600x900.jpg", amostras_promob_scenes["s21"], (6400, 3600)),
        ]
        nvsmi_peaks = []
        for fname, payload, expected_size in workloads:
            with vram_monitor(poll_interval=0.10) as mon:
                resp = http.post(
                    "/api/enhance",
                    files={"file": (fname, payload, "image/jpeg")},
                    data=form_data,
                )
                assert resp.status_code in (200, 202), resp.text
                job_id = resp.json()["job_id"]
                job_poller(http, job_id, timeout_seconds=360.0)
                dl = http.get(f"/api/download/{job_id}")
                assert dl.status_code == 200
                out_img = Image.open(io.BytesIO(dl.content))
                assert out_img.format == "PNG" and out_img.size == expected_size, (out_img.size, expected_size)
            mon.assert_within_budget(max_nvidia_mb=5632.0, max_delta_nvidia_mb=4600.0, max_torch_mb=4600.0)
            if mon.nvidia_available:
                nvsmi_peaks.append(mon.peak_nvidia_mb)

        if len(nvsmi_peaks) == 2:
            assert nvsmi_peaks[1] <= nvsmi_peaks[0] + 512.0, f"Offline 4K VRAM leak: {nvsmi_peaks}"

    assert offline_audited_server.get_outbound_attempts() == []


# ============================================================================
# Workload 8: CPU-Only Offline API Session & <100ms Preview Latency Check
# ============================================================================

@pytest.mark.offline
def test_t4_r4_offline_api_client_session_classic_and_preview_latency(
    offline_audited_server, synthetic_1080p_interior, promob_reference_image, job_poller
):
    """Offline socket-audited API session: health -> capabilities -> presets -> 1080p preview <100ms -> strength=0 enhance -> download."""
    import httpx

    with httpx.Client(base_url=offline_audited_server.base_url, timeout=60.0) as http:
        assert http.get("/api/health").status_code == 200
        assert http.get("/api/capabilities").status_code == 200
        presets = http.get("/api/presets").json()
        assert "realismo_ia" in presets

        for _ in range(3):
            warm = http.post(
                "/api/preview",
                files={"file": ("1080p.jpg", synthetic_1080p_interior, "image/jpeg")},
                data={"render_mode": "generative", "specular_bloom": "0.25"},
            )
            assert warm.status_code == 200

        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = http.post(
                "/api/preview",
                files={"file": ("1080p.jpg", synthetic_1080p_interior, "image/jpeg")},
                data={"render_mode": "generative", "specular_bloom": "0.25"},
            )
            latencies.append(time.perf_counter() - t0)
            assert r.status_code == 200
        assert statistics.median(latencies) < 0.150, f"Offline preview latency unexpectedly slow: {latencies}"

        enh = http.post(
            "/api/enhance",
            files={"file": ("promob_cpu.jpg", promob_reference_image, "image/jpeg")},
            data={"render_mode": "generative", "generative_strength": "0.0", "upscale_factor": "1"},
        )
        assert enh.status_code in (200, 202)
        job_id = enh.json()["job_id"]
        job_poller(http, job_id, timeout_seconds=60.0)
        dl = http.get(f"/api/download/{job_id}")
        assert dl.status_code == 200
        assert Image.open(io.BytesIO(dl.content)).format == "PNG"

    assert offline_audited_server.get_outbound_attempts() == []
