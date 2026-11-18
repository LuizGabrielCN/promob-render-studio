"""Tier 1: Feature Coverage E2E Tests for R1-R5 (Generative ControlNet, PBR & UI).

Opaque-box tests exercising:
- /api/presets ('realismo_ia' + classic backward-compatible presets)
- /api/capabilities (legacy contract + generative_engine metadata)
- 1080p interactive preview latency (< 100 ms) with Multi-Scale SSAO + Soft GI Bloom
- Multi-Scale SSAO (r=3, r=11, r=35) & uniform surface invariance via lossless PNG
- Soft GI Bloom & selective specular highlighting via /api/preview and /api/enhance
- scene_profile options & generative_strength=0.0 short-circuit equivalence
- Local ControlNet generative execution (@pytest.mark.gpu)
- Frontend vitest, production build, served HTML/assets/sample, and headless CDP UI defaults
"""

import io
import os
import re
import statistics
import subprocess
import time
from typing import Dict

import numpy as np
import pytest
from PIL import Image

from tests.e2e.conftest import BLOOM_ROIS, REPO_ROOT, SSAO_MULTISCALE_ROIS
from tests.e2e.metrics_helpers import compute_geometry_preservation


def _run_enhance_png(client, job_poller, image_bytes: bytes, filename: str, data: Dict[str, str], timeout: float = 60.0) -> bytes:
    """Helper submitting POST /api/enhance (upscale_factor=1 lossless PNG) and returning bytes."""
    files = {"file": (filename, image_bytes, "image/png")}
    resp = client.post("/api/enhance", files=files, data=data)
    assert resp.status_code in (200, 202), f"/api/enhance failed ({resp.status_code}): {resp.text}"
    job_id = resp.json()["job_id"]
    job_data = job_poller(client, job_id, timeout_seconds=timeout)
    assert job_data.get("status") in ("completed", "finished", "success")
    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200, f"/api/download/{job_id} failed ({dl.status_code}): {dl.text}"
    return dl.content


# ============================================================================
# Group 1: R4 API Presets & Capabilities Contracts
# ============================================================================

def test_t1_r4_presets_include_realismo_ia_and_classic_modes(client):
    """GET /api/presets must expose 'realismo_ia' and preserve all 4 classic presets."""
    resp = client.get("/api/presets")
    assert resp.status_code == 200, f"GET /api/presets failed: {resp.text}"
    presets = resp.json()

    expected_classic = ("natural", "marcenaria", "luz_quente", "ultra_detalhe")
    for key in expected_classic:
        assert key in presets, f"Missing classic preset '{key}'"
        cfg = presets[key].get("config", presets[key])
        assert cfg.get("render_mode") == "classic", (
            f"Classic preset '{key}' expected render_mode='classic', got {cfg.get('render_mode')}"
        )

    assert "realismo_ia" in presets, f"Missing 'realismo_ia' preset in {list(presets.keys())}"
    realismo_cfg = presets["realismo_ia"].get("config", presets["realismo_ia"])
    assert realismo_cfg.get("render_mode") == "generative"
    strength = float(realismo_cfg.get("generative_strength", 0.0))
    assert 0.0 < strength <= 1.0, f"Unexpected realismo_ia generative_strength={strength}"
    assert realismo_cfg.get("scene_profile") == "auto"
    bloom = float(realismo_cfg.get("specular_bloom", -1.0))
    assert 0.0 <= bloom <= 1.0


def test_t1_r4_capabilities_backward_compat_and_generative_engine(client):
    """GET /api/capabilities must preserve legacy fields and expose generative_engine metadata."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200, f"GET /api/capabilities failed: {resp.text}"
    caps = resp.json()

    # Legacy contract
    assert caps.get("engine") == "ClassicEnhancer"
    assert caps.get("version") == "1.0.0"
    assert caps.get("max_scale") == 4
    assert caps.get("preserves_geometry") is True
    assert caps.get("deterministic") is True
    assert isinstance(caps.get("photometric_stages"), list) and len(caps["photometric_stages"]) >= 4
    assert caps.get("super_resolution_engine") == "Real-ESRGAN-ONNX"
    assert isinstance(caps.get("hardware"), dict)

    # New R4 generative contract
    assert caps.get("render_modes") == ["classic", "generative"]
    assert caps.get("scene_profiles") == ["auto", "banheiro", "cozinha", "quarto_sala"]

    ge = caps.get("generative_engine")
    assert isinstance(ge, dict), "Missing 'generative_engine' dict in /api/capabilities"
    assert isinstance(ge.get("available"), bool)
    assert isinstance(ge.get("base_model"), str) and len(ge["base_model"].strip()) > 0
    has_control_or_relight = (
        (isinstance(ge.get("controlnet_model"), str) and len(ge["controlnet_model"].strip()) > 0)
        or (isinstance(ge.get("controlnet_models"), list) and len(ge["controlnet_models"]) > 0)
        or (isinstance(ge.get("relighting_model"), str) and len(ge["relighting_model"].strip()) > 0)
    )
    assert has_control_or_relight, f"Missing relighting/controlnet model metadata in capabilities: {ge}"
    assert ge.get("dtype") == "float16"
    assert isinstance(ge.get("cached_locally"), bool)


def test_t1_r4_realismo_ia_preset_preview_roundtrip(client, synthetic_jpeg):
    """Submitting 'realismo_ia' preset config to POST /api/preview stays fast (<1.0s) photometric."""
    presets = client.get("/api/presets").json()
    cfg = presets["realismo_ia"].get("config", presets["realismo_ia"])
    form_data = {
        k: ("true" if v is True else "false" if v is False else str(v))
        for k, v in cfg.items()
    }

    t0 = time.perf_counter()
    resp = client.post(
        "/api/preview",
        files={"file": ("render.jpg", synthetic_jpeg, "image/jpeg")},
        data=form_data,
    )
    elapsed = time.perf_counter() - t0

    assert resp.status_code == 200, f"Preview with realismo_ia failed: {resp.text}"
    assert elapsed < 1.0, f"Preview with render_mode='generative' blocked too long: {elapsed:.3f}s"
    img = Image.open(io.BytesIO(resp.content))
    assert img.size[0] > 0 and img.size[1] > 0


def test_t1_r4_classic_presets_preview_roundtrip(client, synthetic_jpeg):
    """All 4 classic presets fetched from /api/presets must execute cleanly on POST /api/preview."""
    presets = client.get("/api/presets").json()
    for key in ("natural", "marcenaria", "luz_quente", "ultra_detalhe"):
        cfg = presets[key].get("config", presets[key])
        form_data = {
            k: ("true" if v is True else "false" if v is False else str(v))
            for k, v in cfg.items()
        }
        resp = client.post(
            "/api/preview",
            files={"file": ("render.jpg", synthetic_jpeg, "image/jpeg")},
            data=form_data,
        )
        assert resp.status_code == 200, f"Classic preset '{key}' preview failed: {resp.text}"
        img = Image.open(io.BytesIO(resp.content))
        assert img.size[0] > 0 and img.size[1] > 0


# ============================================================================
# Group 2: R2 PBR Pipeline (1080p Preview < 100ms, Multi-Scale SSAO, Bloom)
# ============================================================================

def test_t1_r2_preview_1080p_latency_under_100ms(client, synthetic_1080p_interior):
    """1920x1080 POST /api/preview with Multi-Scale SSAO + Soft GI Bloom must have median < 100 ms."""
    payload = {
        "ambient_occlusion": "0.25",
        "specular_bloom": "0.25",
        "render_mode": "classic",
    }

    # 3 warmup calls to eliminate cold threadpool/OpenCV/font-cache jitter under full-suite load
    for _ in range(3):
        warmup = client.post(
            "/api/preview",
            files={"file": ("interior_1080p.jpg", synthetic_1080p_interior, "image/jpeg")},
            data=payload,
        )
        assert warmup.status_code == 200, f"Warmup 1080p preview failed: {warmup.text}"

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = client.post(
            "/api/preview",
            files={"file": ("interior_1080p.jpg", synthetic_1080p_interior, "image/jpeg")},
            data=payload,
        )
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 200, f"Timed 1080p preview failed: {resp.text}"
        img = Image.open(io.BytesIO(resp.content))
        assert img.size[0] > 0 and img.size[1] > 0
        latencies.append(elapsed)

    median_s = statistics.median(latencies)
    assert median_s < 0.100, (
        f"1080p preview median latency {median_s * 1000:.1f}ms exceeded 100ms budget "
        f"(samples ms: {[round(x * 1000, 1) for x in latencies]})"
    )


def test_t1_r2_multiscale_ssao_three_scales_and_uniform_invariance(
    client, synthetic_ssao_multiscale_png, job_poller
):
    """Multi-Scale SSAO must darken 2px seam (r=3), 10px joint (r=11), and 30px recess (r=35) while keeping uniform wall invariant."""
    neutral_base = {
        "temperature": "0.0",
        "tint": "0.0",
        "exposure": "0.0",
        "shadow_lift": "0.0",
        "highlight_recovery": "0.0",
        "clahe_clip_limit": "0.0",
        "sharpness": "0.0",
        "denoise_strength": "0.0",
        "specular_bloom": "0.0",
        "render_mode": "classic",
        "upscale_factor": "1",
        "preserve_geometry": "true",
    }

    base_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_ssao_multiscale_png,
        "ssao_base.png",
        {**neutral_base, "ambient_occlusion": "0.0"},
    )
    ao_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_ssao_multiscale_png,
        "ssao_active.png",
        {**neutral_base, "ambient_occlusion": "0.80"},
    )

    base_arr = np.asarray(Image.open(io.BytesIO(base_png)).convert("RGB"), dtype=np.float32)
    ao_arr = np.asarray(Image.open(io.BytesIO(ao_png)).convert("RGB"), dtype=np.float32)
    assert base_arr.shape == ao_arr.shape == (512, 512, 3)

    # 1. Uniform wall interior invariance
    wall_roi = SSAO_MULTISCALE_ROIS["wall"]
    wall_diff = float(np.mean(np.abs(ao_arr[wall_roi] - base_arr[wall_roi])))
    assert wall_diff < 0.5, f"Uniform wall drifted under SSAO: mean(|diff|)={wall_diff:.3f}"

    # 2. Fine seam (r=3), Medium joint (r=11), Broad recessed corner (r=35) all darken
    for scale_name in ("seam_r3", "joint_r11", "corner_r35"):
        roi = SSAO_MULTISCALE_ROIS[scale_name]
        mean_base = float(np.mean(base_arr[roi]))
        mean_ao = float(np.mean(ao_arr[roi]))
        assert mean_ao < mean_base - 0.5, (
            f"Multi-scale SSAO failed to darken {scale_name}: base={mean_base:.2f}, ao={mean_ao:.2f}"
        )


def test_t1_r2_soft_gi_bloom_preview_and_enhance(client, synthetic_bloom_png, job_poller):
    """Soft GI Bloom must brighten adjacent lamp halo monotonically while keeping dark corners non-blown."""
    neutral = {
        "temperature": "0.0",
        "tint": "0.0",
        "exposure": "0.0",
        "shadow_lift": "0.0",
        "highlight_recovery": "0.0",
        "clahe_clip_limit": "0.0",
        "ambient_occlusion": "0.0",
        "sharpness": "0.0",
        "denoise_strength": "0.0",
        "render_mode": "classic",
        "upscale_factor": "1",
        "preserve_geometry": "true",
    }

    # Lossless /api/enhance monotonic curve
    halo_means = {}
    dark_means = {}
    for bloom in ("0.0", "0.25", "0.80"):
        out_png = _run_enhance_png(
            client,
            job_poller,
            synthetic_bloom_png,
            f"bloom_{bloom}.png",
            {**neutral, "specular_bloom": bloom},
        )
        arr = np.asarray(Image.open(io.BytesIO(out_png)).convert("RGB"), dtype=np.float32)
        halo_means[bloom] = float(np.mean(arr[BLOOM_ROIS["halo"]]))
        dark_means[bloom] = float(np.mean(arr[BLOOM_ROIS["dark"]]))

    assert halo_means["0.80"] > halo_means["0.25"] > halo_means["0.0"], (
        f"Expected monotonic Soft GI Bloom halo brightening, got {halo_means}"
    )
    assert dark_means["0.80"] < 60.0, f"Dark corner blown out by bloom: {dark_means['0.80']:.2f}"

    # Interactive /api/preview bloom responsiveness
    preview_halos = {}
    for bloom in ("0.0", "0.80"):
        resp = client.post(
            "/api/preview",
            files={"file": ("bloom.png", synthetic_bloom_png, "image/png")},
            data={**neutral, "specular_bloom": bloom},
        )
        assert resp.status_code == 200, f"Preview bloom={bloom} failed: {resp.text}"
        arr = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"), dtype=np.float32)
        preview_halos[bloom] = float(np.mean(arr[BLOOM_ROIS["halo"]]))

    assert preview_halos["0.80"] > preview_halos["0.0"], (
        f"Preview Soft GI Bloom did not brighten halo: {preview_halos}"
    )


# ============================================================================
# Group 3: R3 Generative Engine Controls & Execution
# ============================================================================

def test_t1_r3_scene_profile_all_valid_options_preview(client, synthetic_jpeg):
    """All valid scene_profile options (auto, banheiro, cozinha, quarto_sala) return HTTP 200 on /api/preview."""
    for profile in ("auto", "banheiro", "cozinha", "quarto_sala"):
        resp = client.post(
            "/api/preview",
            files={"file": ("scene.jpg", synthetic_jpeg, "image/jpeg")},
            data={
                "render_mode": "generative",
                "generative_strength": "0.48",
                "scene_profile": profile,
                "specular_bloom": "0.25",
            },
        )
        assert resp.status_code == 200, f"scene_profile={profile!r} failed: {resp.text}"
        img = Image.open(io.BytesIO(resp.content))
        assert img.size[0] > 0 and img.size[1] > 0


def test_t1_r3_generative_strength_zero_matches_classic_enhance(client, synthetic_png, job_poller):
    """render_mode='generative' with generative_strength=0.0 must short-circuit ControlNet and match 'classic' bit-for-bit."""
    common = {
        "temperature": "0.10",
        "shadow_lift": "0.20",
        "ambient_occlusion": "0.15",
        "specular_bloom": "0.25",
        "upscale_factor": "1",
        "preserve_geometry": "true",
    }
    classic_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_png,
        "classic.png",
        {**common, "render_mode": "classic", "generative_strength": "0.0"},
    )
    gen_zero_png = _run_enhance_png(
        client,
        job_poller,
        synthetic_png,
        "gen_zero.png",
        {**common, "render_mode": "generative", "generative_strength": "0.0"},
    )

    arr_classic = np.asarray(Image.open(io.BytesIO(classic_png)).convert("RGB"))
    arr_gen_zero = np.asarray(Image.open(io.BytesIO(gen_zero_png)).convert("RGB"))
    assert np.array_equal(arr_classic, arr_gen_zero), (
        f"generative_strength=0.0 diverged from classic mode (max diff: {np.max(np.abs(arr_classic.astype(int) - arr_gen_zero.astype(int)))})"
    )


@pytest.mark.gpu
def test_t1_r3_local_controlnet_generative_execution(
    client, promob_reference_image, job_poller, ssim_evaluator
):
    """Local ControlNet generative execution produces authentic synthesis while preserving Promob layout."""
    caps = client.get("/api/capabilities").json()
    ge = caps.get("generative_engine", {})
    assert ge.get("available") is True, f"Generative engine unavailable: {caps}"
    timeout_s = 600.0 if not ge.get("cached_locally") else 300.0

    common = {
        "ambient_occlusion": "0.15",
        "specular_bloom": "0.25",
        "upscale_factor": "1",
        "preserve_geometry": "true",
    }
    classic_png = _run_enhance_png(
        client,
        job_poller,
        promob_reference_image,
        "promob_classic.jpg",
        {**common, "render_mode": "classic"},
        timeout=90.0,
    )
    gen_png = _run_enhance_png(
        client,
        job_poller,
        promob_reference_image,
        "promob_gen.jpg",
        {
            **common,
            "render_mode": "generative",
            "generative_strength": "0.48",
            "scene_profile": "quarto_sala",
        },
        timeout=timeout_s,
    )

    classic_arr = np.asarray(Image.open(io.BytesIO(classic_png)).convert("RGB"), dtype=np.float32)
    gen_arr = np.asarray(Image.open(io.BytesIO(gen_png)).convert("RGB"), dtype=np.float32)
    assert classic_arr.shape == gen_arr.shape

    mae = float(np.mean(np.abs(gen_arr - classic_arr)))
    ssim_vs_classic = ssim_evaluator(classic_png, gen_png, downsample_to_reference=True)
    ssim_vs_input = ssim_evaluator(promob_reference_image, gen_png, downsample_to_reference=True)

    geom = compute_geometry_preservation(promob_reference_image, gen_png)
    assert mae > 2.0 and ssim_vs_classic < 0.995, (
        f"Generative mode produced near-identical output to classic (MAE={mae:.2f}, SSIM={ssim_vs_classic:.4f})"
    )
    assert ssim_vs_input >= 0.45, (
        f"Generative structural floor too low against input: SSIM={ssim_vs_input:.4f}"
    )
    assert geom["sobel_corr"] >= 0.65, (
        f"Structural edge preservation too low under relighting: {geom}"
    )


# ============================================================================
# Group 4: R5 Frontend Build, Served Assets & Browser Default UI Controls
# ============================================================================

def test_t1_r5_frontend_vitest_and_production_build(frontend_built):
    """Frontend Vitest suite and TypeScript/Vite production build must both pass cleanly."""
    test_cmd = "npm --prefix frontend test" if os.name == "nt" else ["npm", "--prefix", "frontend", "test"]
    vitest_proc = subprocess.run(
        test_cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120.0,
        shell=(os.name == "nt"),
        check=False,
    )
    assert vitest_proc.returncode == 0, (
        f"Frontend vitest failed (exit {vitest_proc.returncode}):\n{vitest_proc.stdout}\n{vitest_proc.stderr}"
    )

    dist_dir = frontend_built
    assert (dist_dir / "index.html").is_file()
    js_assets = list((dist_dir / "assets").glob("*.js"))
    css_assets = list((dist_dir / "assets").glob("*.css"))
    assert len(js_assets) >= 1 and js_assets[0].stat().st_size > 0
    assert len(css_assets) >= 1 and css_assets[0].stat().st_size > 0
    assert (dist_dir / "samples" / "promob_sample.jpg").is_file()


def test_t1_r5_served_html_assets_and_sample(ephemeral_server):
    """FastAPI serves built frontend HTML, JS/CSS bundle assets, and Promob sample image."""
    import httpx

    with httpx.Client(base_url=ephemeral_server, timeout=30.0) as http_client:
        html_resp = http_client.get("/", headers={"Accept": "text/html"})
        assert html_resp.status_code == 200
        assert "text/html" in html_resp.headers.get("content-type", "").lower()
        assert '<div id="root"' in html_resp.text

        js_paths = re.findall(r'src="(/assets/[^"]+\.js)"', html_resp.text)
        css_paths = re.findall(r'href="(/assets/[^"]+\.css)"', html_resp.text)
        assert js_paths, f"No /assets/*.js reference found in index.html:\n{html_resp.text}"
        assert css_paths, f"No /assets/*.css reference found in index.html:\n{html_resp.text}"

        for asset_path in js_paths + css_paths:
            asset_resp = http_client.get(asset_path)
            assert asset_resp.status_code == 200, f"Asset {asset_path} returned {asset_resp.status_code}"
            assert len(asset_resp.content) > 100

        sample_resp = http_client.get("/samples/promob_sample.jpg")
        assert sample_resp.status_code == 200
        sample_img = Image.open(io.BytesIO(sample_resp.content))
        assert sample_img.format == "JPEG" and sample_img.size[0] > 0 and sample_img.size[1] > 0


@pytest.mark.browser
def test_t1_r5_browser_default_realismo_ia_controls(cdp_browser):
    """Headless Chrome/Edge verifies Realismo IA default controls and initial POST /api/preview wire payload."""
    import httpx

    presets = httpx.get(f"{cdp_browser.base_url}/api/presets", timeout=5.0).json()
    realismo_cfg = presets["realismo_ia"].get("config", presets["realismo_ia"])
    expected_strength = float(realismo_cfg.get("generative_strength", 0.48))
    expected_bloom = float(realismo_cfg.get("specular_bloom", 0.25))

    cdp_browser.observe_api_requests(clear=True)
    cdp_browser.goto(f"{cdp_browser.base_url}/")
    cdp_browser.wait_for_text("Carregar Amostra Promob", timeout=15.0)
    cdp_browser.click_text("Carregar Amostra Promob")
    cdp_browser.wait_for_text("Realismo IA", timeout=15.0)

    body_text = cdp_browser.evaluate("document.body.innerText")
    for expected_label in (
        "Realismo IA",
        "Clássico Rápido",
        "Automático",
        "Banheiro/Lavabo",
        "Cozinha/Gourmet",
        "Sala/Quarto",
        "Fidelidade Promob",
        "Brilho GI & Reflexos (Bloom)",
    ):
        assert expected_label in body_text, f"Missing expected UI text {expected_label!r}"

    req = cdp_browser.wait_for_api_request("/api/preview", min_index=0, timeout=15.0)
    resp = cdp_browser.wait_for_api_response("/api/preview", min_index=0, timeout=15.0)
    assert resp["status"] == 200, f"Initial /api/preview failed with status {resp['status']}"

    fields = req["form_fields"]
    assert fields.get("render_mode") == "generative", f"Expected render_mode=generative, got {fields}"
    assert float(fields.get("generative_strength", -1)) == pytest.approx(expected_strength, abs=0.05)
    assert fields.get("scene_profile") == "auto", f"Expected scene_profile=auto, got {fields}"
    assert float(fields.get("specular_bloom", -1)) == pytest.approx(expected_bloom, abs=0.05)
