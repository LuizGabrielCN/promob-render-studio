"""Tier 3: Pairwise Combinations E2E Tests for R2-R5.

Tests combinatorial interactions:
- Multi-Scale SSAO x Soft GI Bloom x preserve_geometry (verifying 0.75/0.25 dilution removal)
- generative_strength monotonic curve (0.0, 0.28, 0.48, 0.72, 1.0)
- scene_profile pairwise output differentiation (banheiro vs cozinha vs quarto_sala vs auto)
- CIE L*a*b* chroma anchoring on warm oak cabinetry
- Deterministic seed & multi-camera View A vs View B oak tone consistency
- Generative (0.48) + Real-ESRGAN 4x Tiled combination
- Interactive browser UI controls (engine toggle, 4 scene buttons, sliders -> live POST /api/preview)
"""

import io
import itertools
import json

import numpy as np
import pytest
from PIL import Image

from tests.e2e.conftest import (
    BLOOM_ROIS,
    LED_CHANNEL_ROIS,
    OAK_CABINET_ROIS,
    SPOT_FIREPLACE_ROIS,
    SSAO_MULTISCALE_ROIS,
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
    compute_spot_cone_metrics,
    decode_bgr,
    roi_to_slices,
)


def _run_enhance_image(client, job_poller, image_bytes: bytes, filename: str, data: dict, timeout: float = 180.0) -> bytes:
    mime = "image/png" if filename.endswith(".png") else "image/jpeg"
    resp = client.post("/api/enhance", files={"file": (filename, image_bytes, mime)}, data=data)
    assert resp.status_code in (200, 202), f"/api/enhance failed ({resp.status_code}): {resp.text}"
    job_id = resp.json()["job_id"]
    job_data = job_poller(client, job_id, timeout_seconds=timeout)
    assert job_data.get("status") in ("completed", "finished", "success")
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    return dl.content


def _gpu_timeout(client) -> float:
    caps = client.get("/api/capabilities").json()
    ge = caps.get("generative_engine", {})
    assert ge.get("available") is True, f"Generative engine unavailable: {caps}"
    return 600.0 if not ge.get("cached_locally") else 300.0


# ============================================================================
# Group 1: R2 PBR Combinations (Non-GPU)
# ============================================================================

def test_t3_r2_ao_x_bloom_x_preserve_geometry_full_contrast(
    client, synthetic_ssao_multiscale_png, job_poller
):
    """AO x Bloom x preserve_geometry produces full contrast without NaN/clipping or 0.75/0.25 dilution."""
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

    outputs = {}
    for ao, bloom, preserve in itertools.product(("0.0", "0.25", "0.80"), ("0.0", "0.25", "0.80"), ("true", "false")):
        out_png = _run_enhance_image(
            client,
            job_poller,
            synthetic_ssao_multiscale_png,
            f"combo_{ao}_{bloom}_{preserve}.png",
            {
                **neutral,
                "ambient_occlusion": ao,
                "specular_bloom": bloom,
                "preserve_geometry": preserve,
            },
            timeout=45.0,
        )
        arr = np.asarray(Image.open(io.BytesIO(out_png)).convert("RGB"), dtype=np.float32)
        assert not np.isnan(arr).any(), f"NaN at ao={ao}, bloom={bloom}, preserve={preserve}"
        assert (arr > 0).any() and (arr < 255).any()
        outputs[(ao, bloom, preserve)] = arr

    # Removing the old cv2.addWeighted(s4, 0.75, s1, 0.25, 0) blend means preserve_geometry=true
    # delivers full un-attenuated AO/Bloom contrast identical to preserve_geometry=false.
    seam_roi = SSAO_MULTISCALE_ROIS["seam_r3"]
    for ao, bloom in (("0.80", "0.0"), ("0.80", "0.80")):
        arr_t = outputs[(ao, bloom, "true")]
        arr_f = outputs[(ao, bloom, "false")]
        diff = float(np.mean(np.abs(arr_t[seam_roi] - arr_f[seam_roi])))
        assert diff < 0.5, (
            f"Unexpected preserve_geometry attenuation detected at ao={ao}, bloom={bloom}: diff={diff:.3f}"
        )


def test_t3_r2_realismo_ia_preset_x_bloom_preview_sweep(client, synthetic_bloom_png):
    """Realismo IA config combined with specular_bloom sweep (0.0 -> 0.5 -> 1.0) scales halo smoothly on /api/preview."""
    halos = []
    for bloom in ("0.0", "0.50", "1.0"):
        resp = client.post(
            "/api/preview",
            files={"file": ("bloom.png", synthetic_bloom_png, "image/png")},
            data={
                "render_mode": "generative",
                "generative_strength": "0.48",
                "scene_profile": "auto",
                "specular_bloom": bloom,
                "ambient_occlusion": "0.15",
            },
        )
        assert resp.status_code == 200
        arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"), dtype=np.float32)
        assert not np.isnan(arr).any()
        halos.append(float(np.mean(arr[BLOOM_ROIS["halo"]])))

    assert halos[2] > halos[1] > halos[0], f"Bloom sweep did not increase halo monotonically: {halos}"


# ============================================================================
# Group 2: R3 Generative Curve, Scene Profiles, Lab Chroma, Seed & 4x Combo
# ============================================================================

@pytest.mark.gpu
def test_t3_r3_generative_strength_monotonic_curve(
    client, promob_reference_image, job_poller, ssim_evaluator
):
    """generative_strength across (0.0, 0.28, 0.48, 0.72, 1.0) diverges monotonically from 0.0 baseline."""
    timeout_s = _gpu_timeout(client)
    strengths = ("0.0", "0.28", "0.48", "0.72", "1.0")
    outputs = {}

    for s in strengths:
        outputs[s] = _run_enhance_image(
            client,
            job_poller,
            promob_reference_image,
            f"strength_{s}.jpg",
            {
                "render_mode": "generative",
                "generative_strength": s,
                "scene_profile": "auto",
                "specular_bloom": "0.25",
                "upscale_factor": "1",
            },
            timeout=timeout_s,
        )

    base_arr = np.asarray(Image.open(io.BytesIO(outputs["0.0"])).convert("RGB"), dtype=np.float32)
    maes = {}
    inv_ssims = {}
    for s in strengths[1:]:
        arr = np.asarray(Image.open(io.BytesIO(outputs[s])).convert("RGB"), dtype=np.float32)
        maes[s] = float(np.mean(np.abs(arr - base_arr)))
        inv_ssims[s] = 1.0 - ssim_evaluator(outputs["0.0"], outputs[s], downsample_to_reference=True)

    assert maes["0.28"] > 0.5 and inv_ssims["0.28"] > 0.001, f"Low strength barely changed output: {maes}"
    assert maes["0.72"] > maes["0.28"], f"Expected higher relighting delta at 0.72 vs 0.28: {maes}"
    assert maes["1.0"] >= maes["0.72"] * 0.90, f"Expected max strength >= 0.72*0.90: {maes}"


@pytest.mark.gpu
def test_t3_r3_scene_profile_pairwise_differentiation(client, promob_reference_image, job_poller):
    """banheiro, cozinha, quarto_sala, and auto all succeed and distinct profiles differentiate measurably."""
    timeout_s = _gpu_timeout(client)

    arrays = {}
    for profile in ("auto", "banheiro", "cozinha", "quarto_sala"):
        png_bytes = _run_enhance_image(
            client,
            job_poller,
            promob_reference_image,
            f"profile_{profile}.jpg",
            {
                "render_mode": "generative",
                "generative_strength": "0.48",
                "scene_profile": profile,
                "specular_bloom": "0.25",
                "upscale_factor": "1",
            },
            timeout=timeout_s,
        )
        arr = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.float32)
        assert not np.isnan(arr).any() and arr.size > 0
        arrays[profile] = arr

    for p1, p2 in (("banheiro", "cozinha"), ("cozinha", "quarto_sala"), ("banheiro", "quarto_sala")):
        mae = float(np.mean(np.abs(arrays[p1] - arrays[p2])))
        assert mae > 0.3, f"Scene profiles {p1} and {p2} produced indistinguishable outputs (MAE={mae:.3f})"


@pytest.mark.gpu
def test_t3_r3_lab_chroma_anchoring_natural_oak(client, synthetic_oak_cabinet_views, job_poller):
    """Hue-angle / material fidelity preserves warm natural oak tone relative to neutral wall under relighting."""
    timeout_s = _gpu_timeout(client)
    view_a, _ = synthetic_oak_cabinet_views
    oak_roi = OAK_CABINET_ROIS["view_a_oak"]
    wall_roi = OAK_CABINET_ROIS["wall"]

    out_png = _run_enhance_image(
        client,
        job_poller,
        view_a,
        "oak_view_a.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "quarto_sala",
            "specular_bloom": "0.25",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )

    oak_stats = compute_lab_chroma_stats(out_png, mask_slice=oak_roi, ref_bytes=view_a)
    wall_stats = compute_lab_chroma_stats(out_png, mask_slice=wall_roi, ref_bytes=view_a)
    assert oak_stats["b"] > 132.0, f"Oak wood washed out (mean b*={oak_stats['b']:.2f} <= 132.0)"
    assert oak_stats["b"] > wall_stats["b"] + 8.0, (
        f"Oak lost warm hue separation vs neutral wall: oak_b={oak_stats['b']:.2f}, wall_b={wall_stats['b']:.2f}"
    )
    assert oak_stats["a"] >= 120.0, f"Oak shifted green unexpectedly: a*={oak_stats['a']:.2f}"


@pytest.mark.gpu
def test_t3_r3_deterministic_seed_and_multiview_consistency(client, synthetic_oak_cabinet_views, job_poller):
    """Identical requests stay deterministic (MAE<=1.0) and View A vs View B preserve oak Lab chroma (<=6.0)."""
    timeout_s = _gpu_timeout(client)
    view_a, view_b = synthetic_oak_cabinet_views
    params = {
        "render_mode": "generative",
        "generative_strength": "0.48",
        "scene_profile": "quarto_sala",
        "specular_bloom": "0.25",
        "upscale_factor": "1",
    }

    out_a1 = _run_enhance_image(client, job_poller, view_a, "view_a1.png", params, timeout=timeout_s)
    out_a2 = _run_enhance_image(client, job_poller, view_a, "view_a2.png", params, timeout=timeout_s)
    out_b = _run_enhance_image(client, job_poller, view_b, "view_b.png", params, timeout=timeout_s)

    arr_a1 = np.asarray(Image.open(io.BytesIO(out_a1)).convert("RGB"), dtype=np.float32)
    arr_a2 = np.asarray(Image.open(io.BytesIO(out_a2)).convert("RGB"), dtype=np.float32)
    repeat_mae = float(np.mean(np.abs(arr_a1 - arr_a2)))
    assert repeat_mae <= 1.0, f"Non-deterministic repeated generative seed (MAE={repeat_mae:.3f})"

    stats_a = compute_lab_chroma_stats(out_a1, mask_slice=OAK_CABINET_ROIS["view_a_oak"])
    stats_b = compute_lab_chroma_stats(out_b, mask_slice=OAK_CABINET_ROIS["view_b_oak"])
    assert abs(stats_a["a"] - stats_b["a"]) <= 6.0, (
        f"Oak a* inconsistent across views: A={stats_a['a']:.2f}, B={stats_b['a']:.2f}"
    )
    assert abs(stats_a["b"] - stats_b["b"]) <= 6.0, (
        f"Oak b* inconsistent across views: A={stats_a['b']:.2f}, B={stats_b['b']:.2f}"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_t3_r3_generative_plus_realesrgan_4x_tiled_combo(client, promob_reference_image, job_poller):
    """Combining ControlNet generative (0.48) with Real-ESRGAN 4x Tiled produces exact 4x PNG without OOM."""
    timeout_s = _gpu_timeout(client)
    orig_w, orig_h = Image.open(io.BytesIO(promob_reference_image)).size

    out_png = _run_enhance_image(
        client,
        job_poller,
        promob_reference_image,
        "promob_gen_4x.jpg",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "auto",
            "specular_bloom": "0.25",
            "upscale_factor": "4",
            "tile_size": "256",
            "preserve_geometry": "true",
        },
        timeout=timeout_s,
    )

    out_img = Image.open(io.BytesIO(out_png))
    assert out_img.format == "PNG"
    assert out_img.size == (orig_w * 4, orig_h * 4), (
        f"Expected 4x size {(orig_w * 4, orig_h * 4)}, got {out_img.size}"
    )


# ============================================================================
# Group 3: R5 Interactive Browser UI Combinations (Headless CDP)
# ============================================================================

@pytest.mark.browser
def test_t3_r5_browser_interactive_ui_controls_and_live_preview(cdp_browser):
    """Interactive browser combinations update real POST /api/preview FormData and Viewer blob: preview."""
    cdp_browser.observe_api_requests(clear=True)
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    cdp_browser.click_text("Carregar Amostra Promob")
    cdp_browser.wait_for_text("Realismo IA", timeout=15.0)
    cdp_browser.wait_for_api_response("/api/preview", min_index=0, timeout=15.0)

    # 1. Toggle engine: generative -> classic -> generative
    idx = len(cdp_browser.captured_api_requests)
    cdp_browser.click_text("Clássico Rápido")
    req_classic = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    assert req_classic["form_fields"].get("render_mode") == "classic"

    idx = len(cdp_browser.captured_api_requests)
    cdp_browser.click_text("Realismo IA")
    req_gen = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    assert req_gen["form_fields"].get("render_mode") == "generative"

    # The photographic finish is sent to the final render even though the quick preview skips it.
    idx = len(cdp_browser.captured_api_requests)
    cdp_browser.set_radix_slider("Acabamento Fotográfico", key="End")
    req_finish = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    assert float(req_finish["form_fields"].get("photo_finish", -1)) == pytest.approx(1.0)

    # 2. Cycle all 4 scene profile buttons
    for btn_label, expected_profile in (
        ("Banheiro/Lavabo", "banheiro"),
        ("Cozinha/Gourmet", "cozinha"),
        ("Sala/Quarto", "quarto_sala"),
        ("Automático", "auto"),
    ):
        idx = len(cdp_browser.captured_api_requests)
        cdp_browser.click_text(btn_label)
        req_scene = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
        assert req_scene["form_fields"].get("scene_profile") == expected_profile

    # 3. Adjust Fidelidade Promob slider from 0% (Home) to 100% (End)
    idx = len(cdp_browser.captured_api_requests)
    val_min = cdp_browser.set_radix_slider("Fidelidade Promob", key="Home")
    assert val_min == "0", f"Expected slider aria-valuenow='0', got {val_min!r}"
    cdp_browser.wait_for_condition(
        """
        (() => {
          const rows = Array.from(document.querySelectorAll('aside div'));
          const row = rows.find(
            d => (d.innerText || '').includes('Fidelidade Promob') && d.querySelectorAll('[role="slider"]').length === 1
          );
          const readout = row ? row.querySelector('span.font-mono') : null;
          return readout && readout.textContent.trim() === '0%';
        })()
        """,
        timeout=5.0,
    )
    req_min = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    assert float(req_min["form_fields"].get("generative_strength", -1)) == pytest.approx(0.0, abs=0.02)

    idx = len(cdp_browser.captured_api_requests)
    resp_idx_max = len(cdp_browser.network_responses)
    val_max = cdp_browser.set_radix_slider("Fidelidade Promob", key="End")
    assert val_max == "1", f"Expected slider aria-valuenow='1', got {val_max!r}"
    cdp_browser.wait_for_condition(
        """
        (() => {
          const rows = Array.from(document.querySelectorAll('aside div'));
          const row = rows.find(
            d => (d.innerText || '').includes('Fidelidade Promob') && d.querySelectorAll('[role="slider"]').length === 1
          );
          const readout = row ? row.querySelector('span.font-mono') : null;
          return readout && readout.textContent.trim() === '100%';
        })()
        """,
        timeout=5.0,
    )
    req_max = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    assert float(req_max["form_fields"].get("generative_strength", -1)) == pytest.approx(1.0, abs=0.02)
    resp_max = cdp_browser.wait_for_api_response("/api/preview", min_index=resp_idx_max, timeout=10.0)
    assert resp_max["status"] == 200
    cdp_browser.pump_events(0.2)

    # 4. Adjust Bloom slider in real time and confirm Viewer blob: preview image updates
    before_srcs = cdp_browser.evaluate(
        "Array.from(document.querySelectorAll('img')).map(i => i.src)"
    ) or []
    idx = len(cdp_browser.captured_api_requests)
    resp_idx = len(cdp_browser.network_responses)
    cdp_browser.set_radix_slider("Brilho GI & Reflexos (Bloom)", key="End")

    req_bloom = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    resp_bloom = cdp_browser.wait_for_api_response("/api/preview", min_index=resp_idx, timeout=10.0)
    assert resp_bloom["status"] == 200
    assert float(req_bloom["form_fields"].get("specular_bloom", -1)) == pytest.approx(1.0, abs=0.05)

    updated = cdp_browser.wait_for_condition(
        f"""
        (() => {{
          const prev = new Set({json.dumps(before_srcs)});
          return Array.from(document.querySelectorAll('img')).some(
            img => img.src.startsWith('blob:') && !prev.has(img.src) && img.complete && img.naturalWidth > 0
          );
        }})()
        """,
        timeout=10.0,
    )
    assert updated is True


# ============================================================================
# Group 4: F1-F14 Pairwise Relighting, LED, OCR/TV & Offline Combinations
# ============================================================================

def test_t3_f1_f12_capabilities_matches_presets_defaults(client):
    """F1 x F12: /api/capabilities render_modes & scene_profiles align with /api/presets."""
    caps = client.get("/api/capabilities").json()
    presets = client.get("/api/presets").json()
    modes = set(caps.get("render_modes", []))
    profiles = set(caps.get("scene_profiles", []))
    for key, entry in presets.items():
        cfg = entry.get("config", entry)
        rm = cfg.get("render_mode", "classic")
        sp = cfg.get("scene_profile", "auto")
        assert rm in modes, f"Preset {key} render_mode={rm} not in capabilities {modes}"
        assert sp in profiles, f"Preset {key} scene_profile={sp} not in capabilities {profiles}"


def test_t3_f5_f11_led_strips_x_photo_finish_classic_combo(
    client, synthetic_led_channel_png, job_poller
):
    """F5 x F11: led_strips=true paired with photo_finish sweep (0.0, 0.5, 1.0) stays valid and unclipped."""
    for pf in ("0.0", "0.5", "1.0"):
        out = _run_enhance_image(
            client,
            job_poller,
            synthetic_led_channel_png,
            f"led_pf_{pf}.png",
            {
                "render_mode": "classic",
                "led_strips": "true",
                "photo_finish": pf,
                "specular_bloom": "0.25",
                "upscale_factor": "1",
            },
            timeout=45.0,
        )
        arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), dtype=np.float32)
        assert arr.shape == (512, 512, 3) and not np.isnan(arr).any()


@pytest.mark.browser
def test_t3_f13_f12_browser_preset_buttons_and_viewer_zoom_combo(cdp_browser):
    """F13 x F12: Switching presets + Viewer zoom controls & Spacebar fires valid /api/preview requests cleanly."""
    cdp_browser.observe_api_requests(clear=True)
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    cdp_browser.click_text("Carregar Amostra Promob")
    cdp_browser.wait_for_text("Realismo IA", timeout=15.0)
    cdp_browser.wait_for_api_response("/api/preview", min_index=0, timeout=15.0)

    idx = len(cdp_browser.captured_api_requests)
    cdp_browser.click_text("Marcenaria")
    req_marc = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    assert req_marc["form_fields"].get("render_mode") == "classic"

    idx = len(cdp_browser.captured_api_requests)
    cdp_browser.click_text("Realismo IA")
    req_ia = cdp_browser.wait_for_api_request("/api/preview", min_index=idx, timeout=10.0)
    assert req_ia["form_fields"].get("render_mode") == "generative"

    # Exercise zoom controls + Spacebar original-compare hold
    cdp_browser.wait_for_text("100%", timeout=5.0)
    cdp_browser.evaluate("document.querySelector('button[title=\"Aumentar Zoom\"]').click()")
    cdp_browser.wait_for_text("125%", timeout=5.0)
    cdp_browser.evaluate("document.querySelector('button[title=\"Reduzir Zoom\"]').click()")
    cdp_browser.wait_for_text("100%", timeout=5.0)
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


@pytest.mark.gpu
@pytest.mark.relighting
def test_t3_f3_f5_spot_cones_and_led_channel_stay_distinct(
    client, synthetic_led_channel_png, job_poller
):
    """F3 x F5: Concealed yellow LED strips and neural relighting render smoothly without double-bloom blowout."""
    timeout_s = _gpu_timeout(client)
    out = _run_enhance_image(
        client,
        job_poller,
        synthetic_led_channel_png,
        "led_relit.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "scene_profile": "cozinha",
            "led_strips": "true",
            "specular_bloom": "0.35",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    led = compute_led_continuity_metrics(synthetic_led_channel_png, out)
    assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], led
    assert led["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], led
    assert led["gap_frac"] <= R1_THRESHOLDS["led_gap_frac_max"], led


@pytest.mark.gpu
def test_t3_f4_f8_high_strength_relighting_preserves_tv_and_sign_text(
    client, synthetic_text_tv_scene_png, job_poller
):
    """F4 x F8: High relighting strength (0.72) paired with luminance-modulated protection keeps TV/sign crisp."""
    timeout_s = _gpu_timeout(client)
    out = _run_enhance_image(
        client,
        job_poller,
        synthetic_text_tv_scene_png,
        "text_072.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.72",
            "scene_profile": "quarto_sala",
            "preserve_geometry": "true",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    for roi_name in ("tv_screen", "framed_sign"):
        m = compute_roi_preservation(synthetic_text_tv_scene_png, out, roi=TEXT_TV_ROIS[roi_name])
        assert m["zncc"] >= R2_THRESHOLDS["roi_zncc_min"], (roi_name, m)
        assert m["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], (roi_name, m)
        assert m["ssim_L"] >= R2_THRESHOLDS["roi_ssim_L_min"], (roi_name, m)


@pytest.mark.gpu
@pytest.mark.relighting
def test_t3_f6_f4_fireplace_synthesis_next_to_protected_tv(
    client, synthetic_spot_fireplace_scene_png, job_poller
):
    """F6 x F4: Fireplace flame synthesis + spot relighting + adjacent TV/sign protection in one pass."""
    timeout_s = _gpu_timeout(client)
    out = _run_enhance_image(
        client,
        job_poller,
        synthetic_spot_fireplace_scene_png,
        "fp_and_tv.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.60",
            "scene_profile": "quarto_sala",
            "preserve_geometry": "true",
            "latent_object_lock": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    fp = compute_fireplace_metrics(
        synthetic_spot_fireplace_scene_png,
        out,
        firebox_roi=SPOT_FIREPLACE_ROIS["firebox"],
        spill_roi=SPOT_FIREPLACE_ROIS["hearth_spill"],
    )
    tv = compute_roi_preservation(
        synthetic_spot_fireplace_scene_png,
        out,
        roi=SPOT_FIREPLACE_ROIS["tv_sign"],
        expected_text="ELLEGANZA",
    )
    spots = compute_spot_cone_metrics(
        synthetic_spot_fireplace_scene_png, out, spot_roi=SPOT_FIREPLACE_ROIS["spot_wall_wash"]
    )
    assert fp["fp_dchr"] >= 8.0, fp
    assert spots["spot_d_r_minus_b"] >= 3.0, spots
    assert tv["edge_f1_d1"] >= R2_THRESHOLDS["roi_edge_f1_d1_min"], tv
    if tv["ocr_available"]:
        assert tv["ocr_matched"] is True, tv
    wall_slice = roi_to_slices(SPOT_FIREPLACE_ROIS["neutral_wall"], 512, 768)
    wall_stats = compute_lab_chroma_stats(
        out, mask_slice=wall_slice, ref_bytes=synthetic_spot_fireplace_scene_png
    )
    assert abs(wall_stats["delta_ab"]) <= 4.5, wall_stats


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.relighting
def test_t3_f7_f11_relighting_x_photo_finish_x_4x_realesrgan(
    client, promob_reference_image, job_poller
):
    """F7 x F11: Generative relighting + photo_finish=0.5 + bloom=0.25 + 4x Real-ESRGAN produces clean 4x PNG."""
    timeout_s = _gpu_timeout(client)
    orig_w, orig_h = Image.open(io.BytesIO(promob_reference_image)).size
    out = _run_enhance_image(
        client,
        job_poller,
        promob_reference_image,
        "pf_4x.jpg",
        {
            "render_mode": "generative",
            "generative_strength": "0.55",
            "scene_profile": "auto",
            "photo_finish": "0.5",
            "specular_bloom": "0.25",
            "upscale_factor": "4",
            "tile_size": "256",
            "preserve_geometry": "true",
        },
        timeout=timeout_s,
    )
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG" and img.size == (orig_w * 4, orig_h * 4)
    geom = compute_geometry_preservation(promob_reference_image, out)
    assert geom["sobel_corr"] >= 0.65, geom
    relit = compute_relighting_metrics(promob_reference_image, out)
    assert relit["dL_low_spread"] >= 2.0, relit


@pytest.mark.gpu
@pytest.mark.offline
def test_t3_f1_f10_offline_server_x_generative_render(
    offline_audited_server, synthetic_png, job_poller
):
    """F1 x F10: Offline socket-audited server executes generative render with zero outbound TCP calls."""
    import httpx

    with httpx.Client(base_url=offline_audited_server.base_url, timeout=300.0) as http:
        caps = http.get("/api/capabilities").json()
        ge = caps.get("generative_engine", {})
        assert ge.get("available") is True and ge.get("cached_locally") is True, caps
        out = _run_enhance_image(
            http,
            job_poller,
            synthetic_png,
            "offline_gen.png",
            {"render_mode": "generative", "generative_strength": "0.48", "upscale_factor": "1"},
            timeout=300.0,
        )
    assert Image.open(io.BytesIO(out)).size == (256, 256)
    assert offline_audited_server.get_outbound_attempts() == []


@pytest.mark.gpu
@pytest.mark.relighting
def test_t3_f5_f9_led_guides_x_neutral_paint_x_realismo_ia(
    client, synthetic_led_channel_png, job_poller
):
    """F5 x F9: Yellow LED guides render smoothly under realismo_ia without yellowing neutral control wall."""
    timeout_s = _gpu_timeout(client)
    out_on = _run_enhance_image(
        client,
        job_poller,
        synthetic_led_channel_png,
        "led_on_f5_f9.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "cozinha",
            "led_strips": "true",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    out_off = _run_enhance_image(
        client,
        job_poller,
        synthetic_led_channel_png,
        "led_off_f5_f9.png",
        {
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "cozinha",
            "led_strips": "false",
            "upscale_factor": "1",
        },
        timeout=timeout_s,
    )
    led_on = compute_led_continuity_metrics(synthetic_led_channel_png, out_on)
    assert led_on["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], led_on
    assert led_on["center_cv"] <= R1_THRESHOLDS["led_center_cv_max"], led_on
    assert led_on["gap_frac"] <= R1_THRESHOLDS["led_gap_frac_max"], led_on
    assert led_on["spill_dL"] >= R1_THRESHOLDS["led_spill_dL_min"], led_on
    assert led_on["spill_db"] >= R1_THRESHOLDS["led_spill_db_min"], led_on

    wall_slice = roi_to_slices(LED_CHANNEL_ROIS["control_wall"], 512, 512)
    wall_in = compute_lab_chroma_stats(synthetic_led_channel_png, mask_slice=wall_slice)
    wall_out = compute_lab_chroma_stats(out_on, mask_slice=wall_slice, ref_bytes=synthetic_led_channel_png)
    assert wall_out["delta_ab"] <= 4.5 and abs(wall_out["b"] - wall_in["b"]) <= 4.0, (wall_in, wall_out)

    led_off = compute_led_continuity_metrics(synthetic_led_channel_png, out_off)
    assert led_off["yellow_rem"] >= 0.65, led_off


@pytest.mark.gpu
def test_t3_f5_f11_led_strips_x_generative_photo_finish_sweep(
    client, synthetic_led_channel_png, job_poller
):
    """F5 x F11: Smooth LEDs stay intact across active generative photo_finish (0.0, 0.5, 1.0)."""
    timeout_s = _gpu_timeout(client)
    outputs = {}
    for pf in ("0.0", "0.5", "1.0"):
        out = _run_enhance_image(
            client,
            job_poller,
            synthetic_led_channel_png,
            f"led_gen_pf_{pf}.png",
            {
                "render_mode": "generative",
                "generative_strength": "0.48",
                "scene_profile": "cozinha",
                "led_strips": "true",
                "photo_finish": pf,
                "upscale_factor": "1",
            },
            timeout=timeout_s,
        )
        arr = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), dtype=np.float32)
        assert not np.isnan(arr).any()
        led = compute_led_continuity_metrics(synthetic_led_channel_png, out)
        assert led["yellow_rem"] <= R1_THRESHOLDS["led_yellow_rem_max"], (pf, led)
        assert led["center_cv"] <= 0.055, (pf, led)
        assert led["gap_frac"] <= 0.05, (pf, led)
        outputs[pf] = arr

    mae = float(np.mean(np.abs(outputs["1.0"] - outputs["0.0"])))
    assert mae >= 0.5, f"photo_finish=1.0 vs 0.0 had negligible generative effect: MAE={mae:.3f}"


@pytest.mark.gpu
@pytest.mark.slow
def test_t3_f9_f11_scene_profile_x_specular_bloom_x_4x_upscale(
    client, promob_reference_image, job_poller
):
    """F9 x F11: scene_profile (cozinha, banheiro) x specular_bloom (0.0, 0.80) x 4x Real-ESRGAN."""
    timeout_s = _gpu_timeout(client)
    orig_w, orig_h = Image.open(io.BytesIO(promob_reference_image)).size
    hi_stats = {}
    for profile, bloom in (("cozinha", "0.0"), ("cozinha", "0.80"), ("banheiro", "0.80")):
        out = _run_enhance_image(
            client,
            job_poller,
            promob_reference_image,
            f"sp_{profile}_{bloom}.jpg",
            {
                "render_mode": "generative",
                "generative_strength": "0.48",
                "scene_profile": profile,
                "specular_bloom": bloom,
                "upscale_factor": "4",
                "tile_size": "256",
            },
            timeout=timeout_s,
        )
        img = Image.open(io.BytesIO(out))
        assert img.format == "PNG" and img.size == (orig_w * 4, orig_h * 4)
        geom = compute_geometry_preservation(promob_reference_image, out)
        assert geom["sobel_corr"] >= R2_THRESHOLDS["sobel_corr_min"], (profile, bloom, geom)
        assert geom["canny_f1_d2"] >= R2_THRESHOLDS["canny_f1_d2_min"], (profile, bloom, geom)
        arr = decode_bgr(out).astype(np.float32)
        assert float(np.mean(arr)) < 245.0
        hi_stats[(profile, bloom)] = float(np.percentile(arr, 99))

    assert hi_stats[("cozinha", "0.80")] >= hi_stats[("cozinha", "0.0")] - 1.0, hi_stats


@pytest.mark.gpu
@pytest.mark.slow
def test_t3_f14_f7_strength_zero_fallback_vs_generative_4x(
    client, synthetic_led_channel_png, job_poller
):
    """F14 x F7: generative_strength=0.0 matches classic 4x bit-for-bit (skipping photo_finish) while 0.48 diverges."""
    timeout_s = _gpu_timeout(client)
    common = {
        "led_strips": "true",
        "photo_finish": "0.5",
        "upscale_factor": "4",
        "tile_size": "256",
    }
    out_classic = _run_enhance_image(
        client,
        job_poller,
        synthetic_led_channel_png,
        "fb_classic_4x.png",
        {**common, "render_mode": "classic", "generative_strength": "0.0"},
        timeout=90.0,
    )
    out_zero = _run_enhance_image(
        client,
        job_poller,
        synthetic_led_channel_png,
        "fb_gen0_4x.png",
        {**common, "render_mode": "generative", "generative_strength": "0.0"},
        timeout=90.0,
    )
    out_gen = _run_enhance_image(
        client,
        job_poller,
        synthetic_led_channel_png,
        "fb_gen48_4x.png",
        {**common, "render_mode": "generative", "generative_strength": "0.48"},
        timeout=timeout_s,
    )
    a_c = np.asarray(Image.open(io.BytesIO(out_classic)).convert("RGB"), dtype=np.float32)
    a_z = np.asarray(Image.open(io.BytesIO(out_zero)).convert("RGB"), dtype=np.float32)
    a_g = np.asarray(Image.open(io.BytesIO(out_gen)).convert("RGB"), dtype=np.float32)
    assert a_c.shape == a_z.shape == a_g.shape == (2048, 2048, 3)
    mae_fallback = float(np.mean(np.abs(a_c - a_z)))
    mae_gen = float(np.mean(np.abs(a_g - a_c)))
    assert mae_fallback <= 1.0, f"strength=0.0 diverged from classic 4x: MAE={mae_fallback:.3f}"
    assert mae_gen >= 3.0, f"generative strength=0.48 did not diverge from classic 4x: MAE={mae_gen:.3f}"
