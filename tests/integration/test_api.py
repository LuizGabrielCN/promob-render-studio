"""
Integration tests for Promob Render Enhancer FastAPI REST API.
Verifies endpoints: /health, /preview, /enhance, /jobs/{id}, /download/{id}, /presets, /capabilities, /
"""
import io
from pathlib import Path
import time
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.storage import get_default_storage


@pytest.fixture(scope="module")
def api_client():
    """Provides a TestClient context with lifespan initialized."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generates a small 128x128 valid JPEG image for fast integration checks."""
    img = Image.new("RGB", (128, 128), color=(180, 160, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ============================================================================
# 1. Health & Root Tests
# ============================================================================

def test_api_health_endpoint(api_client):
    """GET /api/health returns HTTP 200, status 'ok', and valid GPU/VRAM telemetry."""
    response = api_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu" in data and isinstance(data["gpu"], str)
    assert "vram_free_mb" in data and data["vram_free_mb"] >= 0


def test_api_root_endpoint(api_client):
    """GET / returns server metadata and online status."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_api_presets_endpoint(api_client):
    """GET /api/presets returns standard architectural presets plus realismo_ia."""
    response = api_client.get("/api/presets")
    assert response.status_code == 200
    presets = response.json()
    assert "realismo_ia" in presets
    assert presets["realismo_ia"]["config"]["render_mode"] == "generative"
    assert presets["realismo_ia"]["config"]["generative_strength"] == pytest.approx(0.48)
    assert presets["realismo_ia"]["config"]["scene_profile"] == "auto"
    assert presets["realismo_ia"]["config"]["specular_bloom"] == pytest.approx(0.25)
    for key in ("natural", "marcenaria", "luz_quente", "ultra_detalhe"):
        assert key in presets
        assert presets[key]["config"]["render_mode"] == "classic"


def test_api_capabilities_endpoint(api_client):
    """GET /api/capabilities returns engine specifications and generative_engine metadata."""
    response = api_client.get("/api/capabilities")
    assert response.status_code == 200
    caps = response.json()
    assert caps["engine"] == "ClassicEnhancer"
    assert caps["preserves_geometry"] is True
    assert "photometric_stages" in caps
    assert caps["render_modes"] == ["classic", "generative"]
    assert caps["scene_profiles"] == ["auto", "banheiro", "cozinha", "quarto_sala"]
    assert "generative_engine" in caps
    assert caps["generative_engine"]["base_model"] == "emilianJR/epiCRealism"
    assert caps["generative_engine"]["controlnet_model"] == "lllyasviel/control_v11p_sd15_canny"


# ============================================================================
# 2. Preview Endpoint Tests
# ============================================================================

def test_api_preview_success(api_client, sample_jpeg_bytes):
    """POST /api/preview returns processed WebP image bytes."""
    files = {"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
    response = api_client.post("/api/preview", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] in ("image/webp", "image/jpeg")
    img = Image.open(io.BytesIO(response.content))
    assert img.size[0] > 0 and img.size[1] > 0


def test_api_preview_with_parameters(api_client, sample_jpeg_bytes):
    """POST /api/preview accepts photometric form parameters."""
    files = {"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
    data = {
        "temperature": "0.30",
        "tint": "0.10",
        "exposure": "0.20",
        "shadow_lift": "0.35",
        "highlight_recovery": "0.15",
        "sharpness": "0.50",
    }
    response = api_client.post("/api/preview", files=files, data=data)
    assert response.status_code == 200


def test_api_preview_missing_file_returns_422(api_client):
    """POST /api/preview without 'file' returns 422 Unprocessable Entity."""
    response = api_client.post("/api/preview", data={"temperature": "0.5"})
    assert response.status_code == 422


def test_api_preview_out_of_bounds_param_returns_422(api_client, sample_jpeg_bytes):
    """POST /api/preview with parameter outside physical range returns 422."""
    files = {"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
    response = api_client.post("/api/preview", files=files, data={"temperature": "2.5"})
    assert response.status_code == 422


def test_api_preview_empty_file_returns_400(api_client):
    """POST /api/preview with 0-byte file returns 400 Bad Request."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = api_client.post("/api/preview", files=files)
    assert response.status_code == 400


def test_api_preview_corrupt_file_returns_400(api_client):
    """POST /api/preview with corrupt payload returns 400 Bad Request."""
    files = {"file": ("corrupt.jpg", b"NOT_AN_IMAGE_PAYLOAD", "image/jpeg")}
    response = api_client.post("/api/preview", files=files)
    assert response.status_code == 400


# ============================================================================
# 3. Enhance, Job Polling & Download Tests
# ============================================================================

def test_api_enhance_lifecycle_and_download(api_client, sample_jpeg_bytes):
    """Tests the full asynchronous lifecycle: enhance submission, status polling, and download."""
    # 1. Submit enhancement job (scale=1 for speed in integration test)
    files = {"file": ("render.jpg", sample_jpeg_bytes, "image/jpeg")}
    data = {
        "upscale_factor": "1",
        "temperature": "0.10",
        "shadow_lift": "0.20",
    }
    submit_resp = api_client.post("/api/enhance", files=files, data=data)
    assert submit_resp.status_code in (200, 202)
    job_info = submit_resp.json()
    assert "job_id" in job_info
    job_id = job_info["job_id"]
    assert job_info["status"] in ("queued", "processing", "completed")

    # 2. Poll until completed
    max_wait = 15.0
    start = time.time()
    final_job = None
    while time.time() - start < max_wait:
        poll_resp = api_client.get(f"/api/jobs/{job_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        if poll_data.get("status") == "completed":
            final_job = poll_data
            break
        time.sleep(0.3)

    assert final_job is not None, f"Job {job_id} did not finish within {max_wait}s"
    assert final_job["status"] == "completed"
    assert final_job["progress"] == 100.0
    assert final_job["width"] == 128
    assert final_job["height"] == 128
    assert "result_url" in final_job
    assert "processing_time_seconds" in final_job
    assert final_job["processing_time_seconds"] == final_job["processing_time"]
    assert final_job["processing_time_seconds"] is not None and final_job["processing_time_seconds"] >= 0

    # 3. Download the enhanced asset
    dl_resp = api_client.get(f"/api/download/{job_id}")
    assert dl_resp.status_code == 200
    assert "image/png" in dl_resp.headers.get("content-type", "").lower()
    assert "attachment" in dl_resp.headers.get("content-disposition", "").lower()

    # Verify downloaded image content
    dl_img = Image.open(io.BytesIO(dl_resp.content))
    assert dl_img.size == (128, 128)
    assert dl_img.format == "PNG"


def test_api_job_not_found(api_client):
    """GET /api/jobs/{id} returns 404 for unknown job."""
    response = api_client.get("/api/jobs/unknown-job-id-12345")
    assert response.status_code == 404


def test_api_download_not_found(api_client):
    """GET /api/download/{id} returns 404 for unknown job."""
    response = api_client.get("/api/download/unknown-job-id-12345")
    assert response.status_code == 404


def test_api_download_in_progress_job_returns_400(api_client):
    """GET /api/download/{id} returns 400 if job is still in processing status."""
    storage = get_default_storage()
    job = storage.create_job()
    storage.update_job(job.job_id, status="processing")

    response = api_client.get(f"/api/download/{job.job_id}")
    assert response.status_code == 400


def test_api_preview_16bit_png_succeeds(api_client):
    """POST /api/preview normalizes valid 16-bit (uint16) PNG uploads instead of returning HTTP 500."""
    # 3-channel uint16 PNG
    arr_rgb = np.full((64, 64, 3), 30000, dtype=np.uint16)
    ok_rgb, enc_rgb = cv2.imencode(".png", arr_rgb)
    assert ok_rgb
    resp_rgb = api_client.post(
        "/api/preview",
        files={"file": ("u16_rgb.png", enc_rgb.tobytes(), "image/png")},
    )
    assert resp_rgb.status_code == 200
    assert resp_rgb.headers["content-type"] in ("image/webp", "image/jpeg")

    # Grayscale uint16 PNG
    arr_gray = np.full((64, 64), 30000, dtype=np.uint16)
    ok_gray, enc_gray = cv2.imencode(".png", arr_gray)
    assert ok_gray
    resp_gray = api_client.post(
        "/api/preview",
        files={"file": ("u16_gray.png", enc_gray.tobytes(), "image/png")},
    )
    assert resp_gray.status_code == 200


def test_api_enhance_16bit_png_lifecycle(api_client):
    """POST /api/enhance completes cleanly for valid 16-bit (uint16) PNG uploads."""
    arr = np.full((64, 64, 3), 30000, dtype=np.uint16)
    ok, enc = cv2.imencode(".png", arr)
    assert ok

    submit_resp = api_client.post(
        "/api/enhance",
        files={"file": ("u16.png", enc.tobytes(), "image/png")},
        data={"upscale_factor": "1"},
    )
    assert submit_resp.status_code in (200, 202)
    job_id = submit_resp.json()["job_id"]

    final_job = None
    start = time.time()
    while time.time() - start < 15.0:
        poll_resp = api_client.get(f"/api/jobs/{job_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        if poll_data.get("status") in ("completed", "failed"):
            final_job = poll_data
            break
        time.sleep(0.2)

    assert final_job is not None
    assert final_job["status"] == "completed", f"16-bit PNG enhance failed: {final_job}"
    assert final_job["width"] == 64
    assert final_job["height"] == 64


def test_api_unsupported_bit_depths_return_400(api_client):
    """Unsupported 1-channel bit depths (int32, int16, float32 TIFFs) must return HTTP 400, never 500."""
    # int32 1-channel TIFF via PIL mode='I'
    bio_i32 = io.BytesIO()
    Image.fromarray(np.full((32, 32), 100000, dtype=np.int32), mode="I").save(bio_i32, format="TIFF")
    payload_i32 = bio_i32.getvalue()

    # float32 1-channel TIFF via PIL mode='F'
    bio_f32 = io.BytesIO()
    Image.fromarray(np.full((32, 32), 0.5, dtype=np.float32), mode="F").save(bio_f32, format="TIFF")
    payload_f32 = bio_f32.getvalue()

    # int16 1-channel TIFF via OpenCV
    ok_i16, enc_i16 = cv2.imencode(".tiff", np.full((32, 32), 1000, dtype=np.int16))
    assert ok_i16
    payload_i16 = enc_i16.tobytes()

    for label, payload, filename in (
        ("int32", payload_i32, "i32.tiff"),
        ("int16", payload_i16, "i16.tiff"),
        ("float32", payload_f32, "f32.tiff"),
    ):
        for ep in ("preview", "enhance"):
            resp = api_client.post(
                f"/api/{ep}",
                files={"file": (filename, payload, "image/tiff")},
            )
            assert resp.status_code == 400, (
                f"{label} TIFF on /api/{ep} expected HTTP 400, got {resp.status_code}: {resp.text}"
            )
            assert "Profundidade de bits não suportada" in resp.json().get("detail", "")


def test_api_preview_16bit_rgba_png_succeeds(api_client):
    """POST /api/preview handles 4-channel 16-bit (uint16) PNG uploads cleanly."""
    arr_rgba = np.full((64, 64, 4), 30000, dtype=np.uint16)
    ok, enc = cv2.imencode(".png", arr_rgba)
    assert ok
    resp = api_client.post(
        "/api/preview",
        files={"file": ("u16_rgba.png", enc.tobytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] in ("image/webp", "image/jpeg")


def test_api_new_realism_form_fields_are_validated(api_client, sample_jpeg_bytes):
    """photo_finish [0, 1], latent_object_lock/freeu booleans and prompt_style (photo|legacy) are accepted
    and bounded by the form."""
    files = {"file": ("render.jpg", sample_jpeg_bytes, "image/jpeg")}
    ok = api_client.post("/api/preview", files=files, data={
        "photo_finish": "0.7", "latent_object_lock": "false", "freeu": "true", "prompt_style": "legacy"})
    assert ok.status_code == 200
    for bad in ({"photo_finish": "1.5"}, {"prompt_style": "vray"}, {"prompt_style": ""}):
        files = {"file": ("render.jpg", sample_jpeg_bytes, "image/jpeg")}
        assert api_client.post("/api/preview", files=files, data=bad).status_code == 422, bad


@pytest.mark.parametrize("endpoint", ["preview", "enhance"])
def test_realism_form_fields_reach_the_enhancer(api_client, sample_jpeg_bytes, monkeypatch, endpoint):
    from types import SimpleNamespace
    from app.api.routes import get_enhancer

    received = []

    class RecordingEnhancer:
        def process_preview(self, img, config):
            received.append(config)
            return img

        def process(self, img, config, progress_callback=None):
            received.append(config)
            return img, SimpleNamespace(
                output_width=img.shape[1], output_height=img.shape[0], processing_time_seconds=0.01,
            )

    enhancer = RecordingEnhancer()
    monkeypatch.setitem(api_client.app.dependency_overrides, get_enhancer, lambda: enhancer)
    response = api_client.post(f"/api/{endpoint}",
        files={"file": ("render.jpg", sample_jpeg_bytes, "image/jpeg")},
        data={"render_mode": "generative", "photo_finish": "0.73", "latent_object_lock": "false",
              "freeu": "true", "prompt_style": "photo", "led_strips": "false", "upscale_factor": "1"})
    assert response.status_code == 200, response.text
    if endpoint == "enhance":
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = api_client.get(f"/api/jobs/{response.json()['job_id']}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(0.02)
        assert job["status"] == "completed", job
    assert len(received) == 1
    config = received[0]
    assert config.photo_finish == pytest.approx(0.73)
    assert config.latent_object_lock is False and config.freeu is True
    assert config.prompt_style == "photo" and config.led_strips is False


def test_api_preview_returns_jpeg_with_photometric_fidelity(api_client):
    """POST /api/preview returns image/jpeg bytes and preserves warm temperature + exposure shifts."""
    base = np.full((256, 256, 3), 128, dtype=np.uint8)
    ok, enc = cv2.imencode(".png", base)
    assert ok
    png_bytes = enc.tobytes()

    resp_neutral = api_client.post(
        "/api/preview",
        files={"file": ("neutral.png", png_bytes, "image/png")},
        data={"temperature": "0.0", "exposure": "0.0", "specular_bloom": "0.0", "ambient_occlusion": "0.0"},
    )
    assert resp_neutral.status_code == 200
    assert resp_neutral.headers["content-type"] == "image/jpeg"
    img_neutral = Image.open(io.BytesIO(resp_neutral.content))
    assert img_neutral.format == "JPEG"
    arr_neutral = np.asarray(img_neutral.convert("RGB"), dtype=np.float32)

    resp_warm = api_client.post(
        "/api/preview",
        files={"file": ("warm.png", png_bytes, "image/png")},
        data={"temperature": "0.60", "exposure": "0.40", "specular_bloom": "0.0", "ambient_occlusion": "0.0"},
    )
    assert resp_warm.status_code == 200
    assert resp_warm.headers["content-type"] == "image/jpeg"
    arr_warm = np.asarray(Image.open(io.BytesIO(resp_warm.content)).convert("RGB"), dtype=np.float32)

    assert arr_warm.mean() > arr_neutral.mean() + 10.0
    # Warm temperature raises R - B difference
    rb_warm = float((arr_warm[:, :, 0] - arr_warm[:, :, 2]).mean())
    rb_neutral = float((arr_neutral[:, :, 0] - arr_neutral[:, :, 2]).mean())
    assert rb_warm > rb_neutral + 15.0


def test_api_preview_4k_jpeg_fast_decode_and_exif_orientation_consistency(api_client):
    """4K JPEG preview uses fast decode with <=960px proxy and matches /enhance orientation on EXIF tags."""
    # 1. Standard 4K JPEG (3840x2160) -> 960x540 JPEG preview
    img_4k = Image.new("RGB", (3840, 2160), (140, 150, 160))
    buf_4k = io.BytesIO()
    img_4k.save(buf_4k, format="JPEG", quality=85)

    t0 = time.perf_counter()
    resp_4k = api_client.post(
        "/api/preview",
        files={"file": ("4k.jpg", buf_4k.getvalue(), "image/jpeg")},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert resp_4k.status_code == 200
    assert resp_4k.headers["content-type"] == "image/jpeg"
    out_4k = Image.open(io.BytesIO(resp_4k.content))
    assert out_4k.size == (960, 540)
    assert elapsed_ms < 250.0, f"4K JPEG preview took {elapsed_ms:.1f}ms"

    # 2. EXIF-rotated >1920px JPEG must bypass IMREAD_REDUCED_COLOR_2 to match IMREAD_UNCHANGED orientation
    img_exif = Image.new("RGB", (2400, 1200), (120, 130, 140))
    exif = img_exif.getexif()
    exif[274] = 6  # Rotate 90 CW in EXIF metadata
    buf_exif = io.BytesIO()
    img_exif.save(buf_exif, format="JPEG", quality=85, exif=exif)
    resp_exif = api_client.post(
        "/api/preview",
        files={"file": ("exif.jpg", buf_exif.getvalue(), "image/jpeg")},
    )
    assert resp_exif.status_code == 200
    out_exif = Image.open(io.BytesIO(resp_exif.content))
    assert out_exif.size == (960, 480)


def test_api_preview_rejects_truncated_jpeg_and_corrupt_png(api_client, sample_jpeg_bytes):
    """Preview fast-decode path preserves HTTP 400 rejection on truncated JPEG and corrupt PNG."""
    trunc_jpeg = sample_jpeg_bytes[: max(16, len(sample_jpeg_bytes) // 2)]
    resp_jpg = api_client.post(
        "/api/preview",
        files={"file": ("trunc.jpg", trunc_jpeg, "image/jpeg")},
    )
    assert resp_jpg.status_code == 400

    ok, enc_png = cv2.imencode(".png", np.full((64, 64, 3), 120, dtype=np.uint8))
    assert ok
    trunc_png = enc_png.tobytes()[: len(enc_png.tobytes()) // 2]
    resp_png = api_client.post(
        "/api/preview",
        files={"file": ("trunc.png", trunc_png, "image/png")},
    )
    assert resp_png.status_code == 400


def test_api_preview_does_not_block_event_loop_for_health(api_client, sample_jpeg_bytes, monkeypatch):
    """Slow CPU preview processing is offloaded via threadpool so concurrent /api/health stays responsive."""
    import asyncio
    import threading
    import httpx
    from app.api.routes import get_enhancer

    entered = threading.Event()
    release = threading.Event()

    class BlockingEnhancer:
        def process_preview(self, img, config):
            entered.set()
            release.wait(timeout=3.0)
            return img

    monkeypatch.setitem(api_client.app.dependency_overrides, get_enhancer, lambda: BlockingEnhancer())

    async def _run_concurrent():
        transport = httpx.ASGITransport(app=api_client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            preview_task = asyncio.create_task(
                async_client.post(
                    "/api/preview",
                    files={"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")},
                )
            )
            # Yield to the event loop until worker thread enters process_preview
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.005)
            assert entered.is_set(), "Preview worker did not enter process_preview"

            t0 = time.perf_counter()
            health_resp = await async_client.get("/api/health")
            health_ms = (time.perf_counter() - t0) * 1000.0
            assert health_resp.status_code == 200
            assert health_ms < 100.0, f"/api/health blocked behind /api/preview ({health_ms:.1f}ms)"

            release.set()
            preview_resp = await preview_task
            assert preview_resp.status_code == 200

    asyncio.run(_run_concurrent())


# ============================================================================
# Primeira experiência do instalador Windows: amostra no modo Clássico, 1×, CPU
# ============================================================================

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "frontend" / "public" / "samples" / "promob_sample.jpg"


def test_health_identifies_the_app(api_client):
    data = api_client.get("/api/health").json()
    assert data["app"] == "promob-render-studio"


def test_sample_classic_original_resolution_roundtrip(api_client):
    """O botão 'Experimentar com uma amostra' usa a rota real: classic + upscale_factor=1, sem pesos de IA."""
    sample = SAMPLE_PATH.read_bytes()
    src = cv2.imdecode(np.frombuffer(sample, np.uint8), cv2.IMREAD_COLOR)
    response = api_client.post(
        "/api/enhance",
        files={"file": ("promob_sample.jpg", sample, "image/jpeg")},
        data={"render_mode": "classic", "upscale_factor": "1"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    deadline = time.time() + 120
    job = {}
    while time.time() < deadline:
        job = api_client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert job["status"] == "completed", job
    assert (job["output_width"], job["output_height"]) == (src.shape[1], src.shape[0])

    png = api_client.get(f"/api/download/{job_id}")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"
    out = cv2.imdecode(np.frombuffer(png.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == src.shape


def test_features_status_shape(api_client):
    data = api_client.get("/api/features").json()
    assert set(data["features"]) == {"upscale_4x", "realismo_ia"}
    for feature in data["features"].values():
        assert {"ready", "available", "download_bytes", "task"} <= set(feature)


def test_feature_activation_requires_json(api_client):
    assert api_client.post("/api/features/upscale_4x/activate", data={"x": "1"}).status_code == 415


def test_unknown_feature_is_404(api_client):
    assert api_client.post("/api/features/nada/activate", json={}).status_code == 404


def test_shutdown_is_hidden_without_launcher_token(api_client):
    assert api_client.post("/api/app/shutdown", json={}).status_code == 404
