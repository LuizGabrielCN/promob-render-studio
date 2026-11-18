"""Pytest fixtures and test utilities for Promob Render Enhancer E2E test suite.

Provides:
- Configurable test client fixture (Live HTTP server via httpx.Client or in-process FastAPI TestClient)
- Synthetic test image generators (JPEG, PNG, corrupt bytes, extreme aspect ratios)
- Promob reference image loader with realistic fallback
- Structural Similarity Index (SSIM) calculation utility
- Hardware VRAM and resource monitoring utilities
- Async job polling helper
"""

import contextlib
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

# Optional imports for SSIM calculation
try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Optional import for torch GPU metrics
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# Pytest Marker Registration
# ============================================================================

def pytest_configure(config):
    """Register custom markers so CPU-only, relighting, browser, offline, and GPU runs stay warning-free."""
    config.addinivalue_line(
        "markers",
        "gpu: requires local CUDA GPU + ControlNet generative engine",
    )
    config.addinivalue_line(
        "markers",
        "slow: long-running 4K + diffusion E2E workload",
    )
    config.addinivalue_line(
        "markers",
        "relighting: requires M2 architectural relighting leap (R1 spot cones, fireplace flames, dL_low dynamics)",
    )
    config.addinivalue_line(
        "markers",
        "browser: requires built frontend/dist and headless Chrome/Edge CDP",
    )
    config.addinivalue_line(
        "markers",
        "offline: launches isolated OfflineAuditedServer with TCP socket audit",
    )


# ============================================================================
# Configuration & Constants
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
PROMOB_IMAGE_PRIMARY_PATH = os.environ.get("PROMOB_REFERENCE_PATH", "")
CHATGPT_REF_BEFORE_DEFAULT = str(FIXTURES_DIR / "reference_before.jpg")
CHATGPT_REF_AFTER_DEFAULT = str(FIXTURES_DIR / "reference_after.webp")
AMOSTRAS_PROMOB_DEFAULT_DIR = str(FIXTURES_DIR / "promob")
AMOSTRAS_SCENE_FILENAMES: Dict[str, str] = {
    "s00": "WhatsApp Image 2026-11-17 at 15.08.43 (1).jpeg",
    "s09": "WhatsApp Image 2026-11-17 at 15.08.45 (1).jpeg",
    "s14": "WhatsApp Image 2026-11-17 at 15.08.45.jpeg",
    "s15": "WhatsApp Image 2026-11-17 at 15.08.46 (1).jpeg",
    "s17": "WhatsApp Image 2026-11-17 at 15.08.46 (3).jpeg",
    "s21": "WhatsApp Image 2026-11-17 at 15.08.47 (3).jpeg",
    "s23": "WhatsApp Image 2026-11-17 at 15.08.48 (1).jpeg",
}

CHATGPT_REF_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "walnut_upper_spots": (0.04, 0.38, 0.30, 0.52),
    "fireplace_firebox": (0.68, 0.80, 0.355, 0.470),
    "fireplace_spill_floor": (0.81, 0.94, 0.30, 0.52),
    "tv_screen": (0.446, 0.586, 0.340, 0.487),
    "tv_logo_tight": (0.454, 0.552, 0.382, 0.447),
    "shelves_decor": (0.38, 0.71, 0.53, 0.69),
    "right_wall_wash": (0.18, 0.78, 0.84, 0.99),
    "left_wall_bounce": (0.44, 0.77, 0.01, 0.14),
}

AMOSTRAS_ROIS: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "s00": {"twinings_box": (0.09, 0.19, 0.44, 0.51)},
    # Center the WOLF crops on the badges and exclude the upper edge of the oven glass.
    "s09": {
        "wolf_top": (0.442, 0.467, 0.902, 0.923),
        "wolf_bottom": (0.735, 0.755, 0.901, 0.925),
    },
    "s14": {
        "wolf_top": (0.442, 0.467, 0.902, 0.923),
        "wolf_bottom": (0.735, 0.755, 0.901, 0.925),
    },
    "s17": {
        "neutral_wall": (0.20, 0.42, 0.56, 0.78),
        "oak_vanity": (0.62, 0.80, 0.40, 0.60),
    },
    "s21": {
        "bottles_top": (0.30, 0.39, 0.37, 0.52),
        "bottles_mid": (0.40, 0.48, 0.40, 0.55),
    },
    "s23": {
        "tv_lake": (0.37, 0.59, 0.00, 0.07),
        "pendants": (0.08, 0.39, 0.30, 0.68),
        "white_brick_wall": (0.20, 0.48, 0.16, 0.42),
    },
}


# ============================================================================
# Test Client Fixture
# ============================================================================

@pytest.fixture(scope="session")
def client():
    """Provides a unified HTTP client for E2E opaque-box testing.

    Attempts to connect to:
    1. Live running server at BASE_URL (default: http://127.0.0.1:8000)
    2. In-process FastAPI TestClient via `from backend.app.main import app`
    3. If neither is available, skips with an actionable diagnostic message.
    """
    import httpx

    # Mode 1: Check if live server is reachable
    live_url = DEFAULT_BASE_URL.rstrip("/")
    try:
        probe = httpx.get(f"{live_url}/api/health", timeout=1.5)
        if probe.status_code == 200:
            with httpx.Client(base_url=live_url, timeout=300.0) as http_client:
                yield http_client
                return
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        pass

    # Mode 2: In-process FastAPI TestClient fallback
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app
        with TestClient(app) as test_client:
            yield test_client
            return
    except ImportError:
        pass

    # If neither is reachable, skip gracefully
    pytest.skip(
        f"E2E Test Client could not connect to live server at {live_url} "
        f"and 'backend.app.main:app' could not be imported. "
        f"Please start the backend server ('uvicorn backend.app.main:app') or verify backend installation."
    )


# ============================================================================
# Image Generation Utilities & Fixtures
# ============================================================================

def generate_image_bytes(
    width: int,
    height: int,
    format: str = "JPEG",
    style: str = "interior",
    base_color: Tuple[int, int, int] = (195, 190, 185)
) -> bytes:
    """Generates synthetic image bytes in memory with authentic architectural textures."""
    image = Image.new("RGB", (width, height), color=base_color)
    draw = ImageDraw.Draw(image)

    if style == "interior" and width >= 32 and height >= 32:
        # Simulate ceiling, walls, floor, and cabinetry
        ceiling_h = max(int(height * 0.25), 4)
        floor_h = max(int(height * 0.75), 8)

        # Ceiling (slightly brighter cool white)
        draw.rectangle([0, 0, width, ceiling_h], fill=(225, 225, 230))
        # Floor (wood/parquet warm brownish tone)
        draw.rectangle([0, floor_h, width, height], fill=(160, 125, 95))
        # Wall baseboard / line
        draw.line([0, floor_h, width, floor_h], fill=(70, 60, 50), width=max(1, height // 100))

        # Cabinet / island geometry in the center
        box_left = int(width * 0.25)
        box_right = int(width * 0.75)
        box_top = int(height * 0.40)
        box_bottom = floor_h
        if box_right > box_left and box_bottom > box_top:
            draw.rectangle([box_left, box_top, box_right, box_bottom], fill=(130, 120, 115), outline=(50, 50, 50))
            # Countertop line (marble light gray)
            draw.rectangle([box_left - 5, box_top, box_right + 5, box_top + max(3, height // 40)], fill=(235, 235, 235))
    elif style == "gradient":
        for y in range(height):
            ratio = y / max(height - 1, 1)
            r = int(base_color[0] * (1 - ratio * 0.5))
            g = int(base_color[1] * (1 - ratio * 0.5))
            b = int(base_color[2] * (1 - ratio * 0.3))
            draw.line([(0, y), (width, y)], fill=(r, g, b))

    buf = io.BytesIO()
    if format.upper() in ("JPG", "JPEG"):
        image.save(buf, format="JPEG", quality=90)
    elif format.upper() == "PNG":
        image.save(buf, format="PNG")
    elif format.upper() == "WEBP":
        image.save(buf, format="WEBP")
    else:
        image.save(buf, format=format)
    return buf.getvalue()


@pytest.fixture
def synthetic_jpeg() -> bytes:
    """Returns valid 256x256 RGB JPEG image bytes."""
    return generate_image_bytes(256, 256, format="JPEG", style="interior")


@pytest.fixture
def synthetic_png() -> bytes:
    """Returns valid 256x256 RGB PNG image bytes."""
    return generate_image_bytes(256, 256, format="PNG", style="interior")


@pytest.fixture
def synthetic_banner_image() -> bytes:
    """Returns extreme wide aspect ratio banner (1920x64)."""
    return generate_image_bytes(1920, 64, format="JPEG", style="gradient")


@pytest.fixture
def synthetic_tall_image() -> bytes:
    """Returns extreme tall aspect ratio sliver (64x1920)."""
    return generate_image_bytes(64, 1920, format="JPEG", style="gradient")


@pytest.fixture
def synthetic_tiny_image() -> bytes:
    """Returns tiny 16x16 icon-sized image."""
    return generate_image_bytes(16, 16, format="JPEG", style="gradient")


@pytest.fixture
def synthetic_odd_image() -> bytes:
    """Returns non-standard odd dimensions (1001x733)."""
    return generate_image_bytes(1001, 733, format="JPEG", style="interior")


@pytest.fixture
def synthetic_4k_image() -> bytes:
    """Returns 3840x2160 high resolution image."""
    return generate_image_bytes(3840, 2160, format="JPEG", style="interior")


@pytest.fixture
def synthetic_corrupt_bytes() -> bytes:
    """Returns corrupt JPEG bytes with truncated payload."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\xde\xad\xbe\xef" * 8


@pytest.fixture
def synthetic_text_file() -> bytes:
    """Returns plain text bytes representing non-image upload."""
    return b"Architectural specification report: This is not an image file.\nProject: Promob Kitchen."


@pytest.fixture
def synthetic_empty_bytes() -> bytes:
    """Returns 0-byte payload."""
    return b""


@pytest.fixture
def synthetic_exe_file() -> bytes:
    """Returns simulated binary executable payload."""
    return b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00\xb8\x00\x00\x00"


# ============================================================================
# Promob Reference Image Fixture
# ============================================================================

@pytest.fixture(scope="session")
def promob_reference_image() -> bytes:
    """Loads a locally supplied render or generates a synthetic fixture."""
    if PROMOB_IMAGE_PRIMARY_PATH:
        path = Path(PROMOB_IMAGE_PRIMARY_PATH)
        if path.is_file():
            data = path.read_bytes()
            if len(data) > 1000:
                return data

    # Fallback: realistic 723x900 Promob kitchen interior sample
    return generate_image_bytes(723, 900, format="JPEG", style="interior")


@pytest.fixture(scope="session")
def chatgpt_reference_pair() -> Dict[str, Any]:
    """Loads the ChatGPT before/after architectural relighting reference pair.

    Env overrides: CHATGPT_REF_BEFORE_PATH, CHATGPT_REF_AFTER_PATH.
    Skips cleanly if either asset is missing from disk.
    """
    before_path = Path(os.environ.get("CHATGPT_REF_BEFORE_PATH", CHATGPT_REF_BEFORE_DEFAULT))
    after_path = Path(os.environ.get("CHATGPT_REF_AFTER_PATH", CHATGPT_REF_AFTER_DEFAULT))
    if not before_path.is_file() or not after_path.is_file():
        pytest.skip(
            f"ChatGPT reference pair not found on disk: before={before_path} after={after_path}"
        )
    return {
        "before": before_path.read_bytes(),
        "after": after_path.read_bytes(),
        "before_path": str(before_path),
        "after_path": str(after_path),
        "rois": CHATGPT_REF_ROIS,
    }


@pytest.fixture(scope="session")
def amostras_promob_scenes() -> Dict[str, bytes]:
    """Loads key real Promob samples (s00, s09, s14, s15, s17, s21, s23) by exact filename.

    Env override: AMOSTRAS_PROMOB_DIR. Skips cleanly if folder or expected files are absent.
    """
    base_dir = Path(os.environ.get("AMOSTRAS_PROMOB_DIR", AMOSTRAS_PROMOB_DEFAULT_DIR))
    if not base_dir.is_dir():
        pytest.skip(f"AMOSTRAS PROMOB directory not found: {base_dir}")

    scenes: Dict[str, bytes] = {}
    missing = []
    for key, filename in AMOSTRAS_SCENE_FILENAMES.items():
        path = base_dir / filename
        if not path.is_file():
            missing.append(f"{key}={path}")
        else:
            scenes[key] = path.read_bytes()

    if missing:
        pytest.skip(f"Missing expected AMOSTRAS PROMOB scenes: {', '.join(missing)}")
    return scenes


# ============================================================================
# SSIM Utility
# ============================================================================

def compute_ssim(
    image1_bytes: Union[bytes, Image.Image, np.ndarray],
    image2_bytes: Union[bytes, Image.Image, np.ndarray],
    downsample_to_reference: bool = True
) -> float:
    """Computes Structural Similarity Index (SSIM) between two images.

    If dimensions differ, image2 is downsampled to match image1's geometry.
    Uses scikit-image when available, with a mathematically faithful NumPy fallback
    (Wang et al., 2004).
    """
    # Convert image 1 to numpy RGB
    if isinstance(image1_bytes, bytes):
        pil1 = Image.open(io.BytesIO(image1_bytes)).convert("RGB")
    elif isinstance(image1_bytes, Image.Image):
        pil1 = image1_bytes.convert("RGB")
    else:
        pil1 = Image.fromarray(image1_bytes).convert("RGB")

    # Convert image 2 to numpy RGB
    if isinstance(image2_bytes, bytes):
        pil2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB")
    elif isinstance(image2_bytes, Image.Image):
        pil2 = image2_bytes.convert("RGB")
    else:
        pil2 = Image.fromarray(image2_bytes).convert("RGB")

    # Align dimensions if necessary
    if pil1.size != pil2.size and downsample_to_reference:
        pil2 = pil2.resize(pil1.size, Image.Resampling.LANCZOS)

    arr1 = np.asarray(pil1, dtype=np.float64)
    arr2 = np.asarray(pil2, dtype=np.float64)

    # Use scikit-image if available
    if HAS_SKIMAGE:
        return float(skimage_ssim(arr1, arr2, channel_axis=2, data_range=255.0))

    # NumPy Fallback: Wang et al. (2004) SSIM formulation
    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    channels = []
    for c in range(3):
        x = arr1[:, :, c]
        y = arr2[:, :, c]
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x2 = np.var(x)
        sigma_y2 = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))

        num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_channel = num / max(den, 1e-12)
        channels.append(ssim_channel)

    return float(np.mean(channels))


@pytest.fixture
def ssim_evaluator() -> Callable[[Any, Any], float]:
    """Fixture returning the SSIM calculation function."""
    return compute_ssim


# ============================================================================
# Hardware VRAM & Resource Monitor Utility
# ============================================================================

def inspect_vram_and_hardware() -> Dict[str, Any]:
    """Checks current GPU VRAM utilization and hardware safety envelope."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_name": "CPU",
        "peak_vram_mb": 0.0,
        "safe_budget_mb": 5632.0,  # 5.5 GB safety envelope on 6GB card
        "is_safe": True
    }

    if HAS_TORCH and torch.cuda.is_available():
        info["cuda_available"] = True
        info["device_name"] = torch.cuda.get_device_name(0)
        peak_bytes = torch.cuda.max_memory_allocated(0)
        info["peak_vram_mb"] = peak_bytes / (1024.0 * 1024.0)
        info["is_safe"] = info["peak_vram_mb"] <= info["safe_budget_mb"]
    else:
        # Running on CPU is safe by definition regarding GPU VRAM
        info["is_safe"] = True

    return info


@pytest.fixture
def hardware_monitor() -> Callable[[], Dict[str, Any]]:
    """Fixture returning the hardware VRAM inspector function."""
    return inspect_vram_and_hardware


# ============================================================================
# Job Polling Helper
# ============================================================================

def poll_job_until_completion(
    client,
    job_id: str,
    timeout_seconds: float = 120.0,
    poll_interval: float = 0.5
) -> Dict[str, Any]:
    """Polls GET /api/jobs/{job_id} until status reaches 'completed' or 'failed'.

    Raises TimeoutError if execution exceeds timeout_seconds.
    """
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, f"Polling failed with HTTP {response.status_code}: {response.text}"
        data = response.json()
        status = data.get("status")
        if status in ("completed", "finished", "success"):
            return data
        if status in ("failed", "error"):
            raise RuntimeError(f"Job {job_id} failed with error: {data.get('error', 'unknown')}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout_seconds} seconds.")


@pytest.fixture
def job_poller() -> Callable[..., Dict[str, Any]]:
    """Fixture returning the job polling helper."""
    return poll_job_until_completion


# ============================================================================
# R2-R5 Synthetic Image Fixtures & Lab Chroma Utilities
# ============================================================================

SSAO_MULTISCALE_ROIS: Dict[str, Tuple[slice, slice]] = {
    "wall": (slice(50, 150), slice(330, 440)),
    "seam_r3": (slice(60, 180), slice(80, 82)),
    "joint_r11": (slice(283, 287), slice(60, 200)),
    "corner_r35": (slice(320, 420), slice(331, 339)),
}

BLOOM_ROIS: Dict[str, Tuple[slice, slice]] = {
    "lamp": (slice(75, 105), slice(113, 143)),
    "halo": (slice(54, 67), slice(108, 148)),
    "dark": (slice(205, 245), slice(205, 245)),
}

OAK_CABINET_ROIS: Dict[str, Tuple[slice, slice]] = {
    "view_a_oak": (slice(160, 360), slice(150, 235)),
    "view_b_oak": (slice(160, 360), slice(175, 255)),
    "wall": (slice(120, 220), slice(20, 100)),
}

SPOT_FIREPLACE_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "ceiling_spots": (0.02, 0.16, 0.28, 0.72),
    "spot_wall_wash": (0.16, 0.52, 0.28, 0.72),
    "firebox": (0.64, 0.86, 0.36, 0.54),
    "hearth_spill": (0.86, 0.99, 0.30, 0.60),
    "neutral_wall": (0.28, 0.68, 0.03, 0.22),
    "tv_sign": (0.35, 0.58, 0.62, 0.92),
}

LED_CHANNEL_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "led_strip": (0.33, 0.35, 0.15, 0.85),
    "led_spill": (0.35, 0.41, 0.15, 0.85),
    "control_wall": (0.62, 0.86, 0.15, 0.45),
}

TEXT_TV_ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "tv_screen": (0.22, 0.58, 0.14, 0.54),
    "framed_sign": (0.24, 0.56, 0.60, 0.90),
    "control_wall": (0.66, 0.92, 0.08, 0.40),
}


@pytest.fixture
def synthetic_1080p_interior() -> bytes:
    """Returns a 1920x1080 interior architectural JPEG for <100ms preview latency benchmarks."""
    return generate_image_bytes(1920, 1080, format="JPEG", style="interior")


@pytest.fixture
def synthetic_ssao_multiscale_png() -> bytes:
    """Returns a 512x512 lossless PNG with uniform wall, 2px seam, 10px joint, and 30px recess."""
    arr = np.full((512, 512, 3), 190, dtype=np.uint8)
    # 2px dark seam (detected by r=3 fine scale)
    arr[40:200, 80:82] = (110, 110, 110)
    # 10px medium joint (center requires r=11 medium scale)
    arr[280:290, 40:220] = (110, 110, 110)
    # 30px recessed corner/alcove (center requires r=35 depth shading scale)
    arr[300:440, 320:350] = (110, 110, 110)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_bloom_png() -> bytes:
    """Returns a 256x256 lossless PNG with bright lamp, mid-gray wall, and dark corner."""
    arr = np.full((256, 256, 3), 120, dtype=np.uint8)
    arr[70:110, 108:148] = (252, 250, 245)
    arr[190:250, 190:250] = (25, 25, 25)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_oak_cabinet_views() -> Tuple[bytes, bytes]:
    """Returns (view_a_png, view_b_png) with identical warm oak RGB (185,140,95) + neutral gray wall."""
    # View A: Closed cabinet doors
    a = np.full((512, 512, 3), 190, dtype=np.uint8)
    a[0:100, :] = (225, 225, 230)
    a[410:512, :] = (155, 150, 145)
    a[130:390, 130:370] = (185, 140, 95)
    a[130:390, 249:251] = (95, 70, 45)
    a[220:290, 242:246] = (210, 180, 105)
    a[220:290, 254:258] = (210, 180, 105)
    a[135:385, 132:136] = (255, 238, 200)

    # View B: Shifted view with left door ajar and darker interior niche
    b = np.full((512, 512, 3), 190, dtype=np.uint8)
    b[0:100, :] = (225, 225, 230)
    b[410:512, :] = (155, 150, 145)
    b[130:390, 165:270] = (185, 140, 95)
    b[130:390, 270:345] = (90, 70, 55)
    b[130:390, 345:405] = (185, 140, 95)
    b[220:290, 258:262] = (210, 180, 105)
    b[135:385, 167:171] = (255, 238, 200)

    buf_a = io.BytesIO()
    buf_b = io.BytesIO()
    Image.fromarray(a, mode="RGB").save(buf_a, format="PNG")
    Image.fromarray(b, mode="RGB").save(buf_b, format="PNG")
    return buf_a.getvalue(), buf_b.getvalue()


@pytest.fixture
def synthetic_spot_fireplace_scene_png() -> bytes:
    """Returns a deterministic 768x512 architectural PNG with ceiling spots, unlit fireplace, neutral wall, and TV text."""
    w, h = 768, 512
    bgr = np.full((h, w, 3), (185, 185, 185), dtype=np.uint8)
    # Ceiling band
    bgr[0:82, :] = (212, 212, 216)
    # 3 recessed ceiling spots in (0.02..0.16, 0.28..0.72)
    for cx in (260, 384, 508):
        cv2.ellipse(bgr, (cx, 44), (20, 10), 0, 0, 360, (235, 250, 255), -1)
        cv2.ellipse(bgr, (cx, 44), (22, 12), 0, 0, 360, (130, 130, 135), 2)
    # Wood/slatted panel under spots
    panel_x0, panel_x1 = 215, 553
    bgr[82:328, panel_x0:panel_x1] = (120, 145, 175)
    for x in range(panel_x0 + 8, panel_x1, 16):
        cv2.line(bgr, (x, 82), (x, 328), (88, 108, 132), 2)
    # Unlit dark fireplace cutout in (0.64..0.86, 0.36..0.54)
    cv2.rectangle(bgr, (265, 328), (425, 442), (78, 78, 82), -1)
    cv2.rectangle(bgr, (276, 338), (414, 434), (25, 25, 28), -1)
    # Hearth floor in (0.86..0.99, 0.30..0.60)
    bgr[442:512, :] = (165, 165, 168)
    # Crisp TV/sign box in (0.35..0.58, 0.62..0.92)
    cv2.rectangle(bgr, (476, 179), (706, 297), (24, 24, 26), -1)
    cv2.rectangle(bgr, (486, 189), (696, 287), (42, 46, 52), -1)
    cv2.putText(bgr, "ELLEGANZA", (502, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (242, 242, 244), 2, cv2.LINE_AA)
    cv2.putText(bgr, "STUDIO 4K", (516, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (215, 215, 220), 2, cv2.LINE_AA)

    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    return enc.tobytes()


@pytest.fixture
def synthetic_led_channel_png() -> bytes:
    """Returns a deterministic 512x512 PNG with neutral cabinetry and a continuous 5px Promob yellow LED marker strip."""
    w, h = 512, 512
    bgr = np.full((h, w, 3), (186, 186, 188), dtype=np.uint8)
    # Upper cabinet box
    bgr[40:168, 70:442] = (145, 160, 175)
    cv2.rectangle(bgr, (70, 40), (442, 168), (95, 105, 115), 2)
    # Saturated yellow Promob LED marker line: RGB(255, 238, 0) -> BGR(0, 238, 255)
    bgr[169:175, 80:432] = (0, 238, 255)
    # Lower counter
    bgr[310:490, 70:442] = (160, 162, 165)
    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    return enc.tobytes()


@pytest.fixture
def synthetic_text_tv_scene_png() -> bytes:
    """Returns a deterministic 768x512 PNG with crisp TV logo text and framed sign typography."""
    w, h = 768, 512
    bgr = np.full((h, w, 3), (188, 188, 190), dtype=np.uint8)
    bgr[0:76, :] = (218, 218, 222)
    bgr[410:512, :] = (145, 130, 118)
    # TV screen in (0.22..0.58, 0.14..0.54)
    cv2.rectangle(bgr, (108, 113), (415, 297), (20, 20, 22), -1)
    cv2.rectangle(bgr, (118, 123), (405, 287), (40, 44, 50), -1)
    cv2.putText(bgr, "PROMOB", (152, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (245, 245, 248), 3, cv2.LINE_AA)
    cv2.putText(bgr, "CABINETRY", (148, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (215, 220, 228), 2, cv2.LINE_AA)
    # Framed sign in (0.24..0.56, 0.60..0.90)
    cv2.rectangle(bgr, (461, 123), (691, 287), (60, 56, 52), 4)
    cv2.rectangle(bgr, (469, 131), (683, 279), (242, 242, 240), -1)
    cv2.putText(bgr, "WOLF", (512, 194), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (28, 28, 30), 3, cv2.LINE_AA)
    cv2.putText(bgr, "2026", (526, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (45, 45, 48), 2, cv2.LINE_AA)

    ok, enc = cv2.imencode(".png", bgr)
    assert ok
    return enc.tobytes()


@pytest.fixture(scope="session")
def promob_sample_image(promob_reference_image) -> bytes:
    """Loads frontend/public/samples/promob_sample.jpg with fallback to promob_reference_image."""
    sample_path = REPO_ROOT / "frontend" / "public" / "samples" / "promob_sample.jpg"
    if sample_path.is_file():
        data = sample_path.read_bytes()
        if len(data) > 1000:
            return data
    return promob_reference_image


def compute_lab_chroma_stats(
    img_bytes: Union[bytes, np.ndarray],
    mask_slice: Optional[Tuple[slice, slice]] = None,
    ref_bytes: Optional[Union[bytes, np.ndarray, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Computes mean CIE L*, a*, b* and optional delta_ab using cv2.COLOR_BGR2Lab."""
    if isinstance(img_bytes, bytes):
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            pil_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            bgr = cv2.cvtColor(np.asarray(pil_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    else:
        bgr = img_bytes

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    region = lab[mask_slice] if mask_slice is not None else lab
    mean_l = float(np.mean(region[:, :, 0]))
    mean_a = float(np.mean(region[:, :, 1]))
    mean_b = float(np.mean(region[:, :, 2]))

    delta_ab = 0.0
    if ref_bytes is not None:
        if isinstance(ref_bytes, dict):
            ref_a = float(ref_bytes["a"])
            ref_b = float(ref_bytes["b"])
        else:
            ref_stats = compute_lab_chroma_stats(ref_bytes, mask_slice=mask_slice, ref_bytes=None)
            ref_a = ref_stats["a"]
            ref_b = ref_stats["b"]
        delta_ab = float(math.hypot(mean_a - ref_a, mean_b - ref_b))

    return {"L": mean_l, "a": mean_a, "b": mean_b, "delta_ab": delta_ab}


# ============================================================================
# Dual nvidia-smi + PyTorch CUDA VRAM Monitor
# ============================================================================

class VramMonitor:
    """Polls nvidia-smi every 0.10s and captures in-process torch.cuda peak memory."""

    def __init__(self, poll_interval: float = 0.10):
        self.poll_interval = poll_interval
        self.baseline_nvidia_mb: float = 0.0
        self.peak_nvidia_mb: float = 0.0
        self.min_free_nvidia_mb: float = 0.0
        self.peak_torch_mb: float = 0.0
        self.nvidia_available: bool = False
        self.samples: List[Tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _query_nvidia_smi() -> Optional[Tuple[float, float]]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return None
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) < 2:
                return None
            return float(parts[0]), float(parts[1])
        except Exception:
            return None

    def _record_sample(self) -> None:
        sample = self._query_nvidia_smi()
        if sample is None:
            return
        used_mb, free_mb = sample
        if not self.nvidia_available and self.baseline_nvidia_mb == 0.0:
            self.baseline_nvidia_mb = used_mb
        self.nvidia_available = True
        self.samples.append((used_mb, free_mb))
        if used_mb > self.peak_nvidia_mb:
            self.peak_nvidia_mb = used_mb
        if self.min_free_nvidia_mb == 0.0 or free_mb < self.min_free_nvidia_mb:
            self.min_free_nvidia_mb = free_mb

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            self._record_sample()

    def start(self) -> "VramMonitor":
        self._stop_event.clear()
        self.samples.clear()
        self.baseline_nvidia_mb = 0.0
        self.peak_nvidia_mb = 0.0
        self.min_free_nvidia_mb = 0.0
        self.peak_torch_mb = 0.0
        self.nvidia_available = False

        initial = self._query_nvidia_smi()
        if initial is not None:
            self.nvidia_available = True
            self.baseline_nvidia_mb = initial[0]
            self.peak_nvidia_mb = initial[0]
            self.min_free_nvidia_mb = initial[1]
            self.samples.append(initial)

        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(0)
            except Exception:
                pass

        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="vram_monitor")
        self._thread.start()
        return self

    def stop(self) -> "VramMonitor":
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._record_sample()
        if HAS_TORCH and torch.cuda.is_available():
            try:
                self.peak_torch_mb = float(torch.cuda.max_memory_allocated(0)) / (1024.0 * 1024.0)
            except Exception:
                self.peak_torch_mb = 0.0
        return self

    @property
    def delta_nvidia_mb(self) -> float:
        return max(0.0, self.peak_nvidia_mb - self.baseline_nvidia_mb)

    def assert_within_budget(
        self,
        max_nvidia_mb: float = 5632.0,
        max_delta_nvidia_mb: float = 4600.0,
        max_torch_mb: float = 4600.0,
    ) -> None:
        """Asserts dual-process GPU VRAM bounds across nvidia-smi and in-process PyTorch CUDA."""
        assert self.peak_torch_mb > 0.0 or self.nvidia_available, (
            "At least one VRAM monitor source (PyTorch CUDA or nvidia-smi) must be active"
        )
        if self.peak_torch_mb > 0.0:
            assert self.peak_torch_mb < max_torch_mb, (
                f"PyTorch peak VRAM {self.peak_torch_mb:.1f} MB exceeded {max_torch_mb:.1f} MB"
            )
        if self.nvidia_available:
            assert self.peak_nvidia_mb < max_nvidia_mb, (
                f"nvidia-smi peak VRAM {self.peak_nvidia_mb:.1f} MB exceeded {max_nvidia_mb:.1f} MB"
            )
            assert self.delta_nvidia_mb < max_delta_nvidia_mb, (
                f"nvidia-smi delta VRAM {self.delta_nvidia_mb:.1f} MB exceeded {max_delta_nvidia_mb:.1f} MB "
                f"(baseline={self.baseline_nvidia_mb:.1f}, peak={self.peak_nvidia_mb:.1f})"
            )

    @staticmethod
    def assert_health_released(
        pre_alloc_mb: float,
        post_alloc_mb: float,
        tolerance_mb: float = 512.0,
    ) -> None:
        assert post_alloc_mb <= pre_alloc_mb + tolerance_mb, (
            f"Post-job VRAM not released cleanly: pre={pre_alloc_mb:.1f} MB, post={post_alloc_mb:.1f} MB"
        )

    def __enter__(self) -> "VramMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


@pytest.fixture
def vram_monitor() -> Callable[..., VramMonitor]:
    """Fixture returning the VramMonitor class for context-managed GPU memory polling."""
    return VramMonitor


# ============================================================================
# Headless Browser CDP + Ephemeral Uvicorn Fixtures (Layer 2 Frontend E2E)
# ============================================================================

def find_chromium_binary() -> Optional[str]:
    """Probes BROWSER_PATH and standard Windows/POSIX Chrome/Edge locations at runtime."""
    env_browser = os.environ.get("BROWSER_PATH", "").strip()
    if env_browser and os.path.isfile(env_browser):
        return env_browser

    for cmd in ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(cmd)
        if found:
            return found

    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _kill_process_tree(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        for _ in range(2):
            if proc.poll() is not None:
                return
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=5.0,
                    check=False,
                )
            else:
                import signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                proc.wait(timeout=3.0)
                return
            except subprocess.TimeoutExpired:
                time.sleep(0.2)
    except Exception:
        pass


def _parse_multipart_fields(post_data: str) -> Dict[str, str]:
    """Extracts non-file text fields from a multipart/form-data payload."""
    fields: Dict[str, str] = {}
    if not post_data:
        return fields
    pattern = re.compile(
        r'Content-Disposition:\s*form-data;\s*name="([^"]+)"[ \t]*\r?\n\r?\n([^\r\n]*)',
        re.IGNORECASE,
    )
    for match in pattern.finditer(post_data):
        name, value = match.group(1), match.group(2)
        if name != "file":
            fields[name] = value
    return fields


class CdpBrowser:
    """Lightweight synchronous CDP driver over Chrome/Edge WebSocket (zero mocks)."""

    def __init__(self, binary_path: str, base_url: str):
        self.binary_path = binary_path
        self.base_url = base_url.rstrip("/")
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="promob_cdp_", ignore_cleanup_errors=True)
        self._proc: Optional[subprocess.Popen] = None
        self._ws_stack = contextlib.ExitStack()
        self._browser_ws = None
        self._page_ws = None
        self._msg_id = 0
        self._observe_api = False
        self.captured_api_requests: List[Dict[str, Any]] = []
        self.network_responses: List[Dict[str, Any]] = []
        self._pending_requests: Dict[str, Dict[str, Any]] = {}
        self._start()

    def _start(self) -> None:
        cmd = [
            self.binary_path,
            "--headless=new",
            "--disable-gpu",
            "--disable-gpu-shader-disk-cache",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--remote-debugging-port=0",
            f"--user-data-dir={self._tmp_dir.name}",
            "--window-size=1440,960",
            "about:blank",
        ]
        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.PIPE,
            "text": True,
            "errors": "replace",
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(cmd, **popen_kwargs)

        devtools_url = None
        stderr_lines: List[str] = []

        def _read_stderr():
            assert self._proc is not None and self._proc.stderr is not None
            for line in self._proc.stderr:
                stderr_lines.append(line)

        reader = threading.Thread(target=_read_stderr, daemon=True)
        reader.start()

        deadline = time.time() + 15.0
        active_port_file = Path(self._tmp_dir.name) / "DevToolsActivePort"
        while time.time() < deadline:
            for line in list(stderr_lines):
                m = re.search(r"DevTools listening on (ws://\S+)", line)
                if m:
                    devtools_url = m.group(1)
                    break
            if devtools_url:
                break
            if active_port_file.is_file():
                try:
                    lines = active_port_file.read_text(encoding="utf-8").strip().splitlines()
                    if len(lines) >= 2:
                        devtools_url = f"ws://127.0.0.1:{lines[0].strip()}{lines[1].strip()}"
                        break
                except OSError:
                    pass
            if self._proc.poll() is not None:
                rc = self._proc.returncode
                self.close()
                raise RuntimeError(
                    f"Chromium exited early ({rc}): {''.join(stderr_lines)}"
                )
            time.sleep(0.1)

        if not devtools_url:
            self.close()
            raise RuntimeError(f"Failed to acquire DevTools URL: {''.join(stderr_lines)}")

        import httpx
        from websockets.sync.client import connect

        port_match = re.search(r"ws://127\.0\.0\.1:(\d+)", devtools_url)
        if not port_match:
            self.close()
            raise RuntimeError(f"Unexpected DevTools URL format: {devtools_url}")
        debug_port = port_match.group(1)

        page_ws_url = None
        for _ in range(30):
            try:
                targets = httpx.get(f"http://127.0.0.1:{debug_port}/json/list", timeout=2.0).json()
                for target in targets:
                    if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                        page_ws_url = target["webSocketDebuggerUrl"]
                        break
                if page_ws_url:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        if not page_ws_url:
            self.close()
            raise RuntimeError("No page target exposed by Chromium DevTools.")

        self._browser_ws = self._ws_stack.enter_context(connect(devtools_url, max_size=16 * 1024 * 1024))
        self._page_ws = self._ws_stack.enter_context(connect(page_ws_url, max_size=32 * 1024 * 1024))

        self._send("Page.enable")
        self._send("Runtime.enable")
        self._send("DOM.enable")
        self._send("Network.enable", {"maxPostDataSize": 10 * 1024 * 1024})

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _handle_event(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "Fetch.requestPaused":
            req_id = params["requestId"]
            req = params.get("request", {})
            url = req.get("url", "")
            req_method = req.get("method", "")
            post_data = req.get("postData", "")
            if not post_data and req.get("postDataEntries"):
                import base64
                chunks = []
                for entry in req["postDataEntries"]:
                    b = entry.get("bytes")
                    if b:
                        chunks.append(base64.b64decode(b).decode("latin1", errors="replace"))
                post_data = "".join(chunks)

            if self._observe_api and "/api/" in url:
                self.captured_api_requests.append(
                    {
                        "url": url,
                        "method": req_method,
                        "headers": req.get("headers", {}),
                        "postData": post_data,
                        "form_fields": _parse_multipart_fields(post_data),
                        "networkId": params.get("networkId"),
                        "timestamp": time.time(),
                    }
                )

            # NEVER mock or fulfill /api/* requests: always continue to the real backend.
            cont_id = self._next_id()
            self._page_ws.send(
                json.dumps(
                    {
                        "id": cont_id,
                        "method": "Fetch.continueRequest",
                        "params": {"requestId": req_id},
                    }
                )
            )
        elif method == "Network.requestWillBeSent":
            req_id = params.get("requestId", "")
            req = params.get("request", {})
            url = req.get("url", "")
            if "/api/" in url:
                self._pending_requests[req_id] = {
                    "url": url,
                    "method": req.get("method", ""),
                    "postData": req.get("postData", ""),
                }
        elif method == "Network.responseReceived":
            req_id = params.get("requestId", "")
            resp = params.get("response", {})
            url = resp.get("url", "")
            if "/api/" in url:
                self.network_responses.append(
                    {
                        "requestId": req_id,
                        "url": url,
                        "status": resp.get("status"),
                        "mimeType": resp.get("mimeType"),
                        "timestamp": time.time(),
                    }
                )

    def _send(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 15.0) -> Dict[str, Any]:
        assert self._page_ws is not None
        cmd_id = self._next_id()
        payload = {"id": cmd_id, "method": method, "params": params or {}}
        self._page_ws.send(json.dumps(payload))

        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            raw = self._page_ws.recv(timeout=remaining)
            msg = json.loads(raw)
            if msg.get("id") == cmd_id:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method} failed: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:
                self._handle_event(msg)
        raise TimeoutError(f"Timed out waiting for CDP response to {method}")

    def pump_events(self, duration_seconds: float = 0.25) -> None:
        if self._page_ws is None:
            return
        deadline = time.time() + duration_seconds
        while time.time() < deadline:
            try:
                raw = self._page_ws.recv(timeout=0.05)
                msg = json.loads(raw)
                if "method" in msg:
                    self._handle_event(msg)
            except TimeoutError:
                pass
            except Exception:
                break

    def observe_api_requests(self, clear: bool = True) -> None:
        if clear:
            self.captured_api_requests.clear()
            self.network_responses.clear()
        if not self._observe_api:
            self._send(
                "Fetch.enable",
                {
                    "patterns": [
                        {"urlPattern": "*://127.0.0.1:*/api/*", "requestStage": "Request"},
                        {"urlPattern": "*://localhost:*/api/*", "requestStage": "Request"},
                    ]
                },
            )
            self._observe_api = True

    def clear_observed_requests(self) -> None:
        self.captured_api_requests.clear()
        self.network_responses.clear()

    def goto(self, url: str, timeout: float = 15.0) -> None:
        self._send("Page.navigate", {"url": url})
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.evaluate("document.readyState")
            if state == "complete":
                return
            self.pump_events(0.1)
        raise TimeoutError(f"Page {url} did not reach readyState=complete within {timeout}s")

    def evaluate(self, expression: str) -> Any:
        result = self._send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        if "exceptionDetails" in result:
            raise RuntimeError(f"JS evaluation error: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def wait_for_text(self, text: str, timeout: float = 15.0) -> None:
        js = f"(document.body && document.body.innerText || '').includes({json.dumps(text)})"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.evaluate(js):
                return
            self.pump_events(0.15)
        body = self.evaluate("document.body ? document.body.innerText : ''")
        raise TimeoutError(f"Text {text!r} not found within {timeout}s. Page text:\n{body[:800]}")

    def wait_for_condition(self, js_expr: str, timeout: float = 15.0) -> Any:
        deadline = time.time() + timeout
        while time.time() < deadline:
            val = self.evaluate(js_expr)
            if val:
                return val
            self.pump_events(0.15)
        raise TimeoutError(f"Condition {js_expr!r} remained falsy after {timeout}s")

    def click_text(self, text: str) -> None:
        js = f"""
        (() => {{
          const target = {json.dumps(text)};
          const candidates = Array.from(document.querySelectorAll('button, [role="button"], a'));
          const visible = candidates.filter(el => {{
            const style = window.getComputedStyle(el);
            const r = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden' && r.width > 0 && r.height > 0;
          }});
          let match = visible.find(el => (el.innerText || '').trim() === target);
          if (!match) {{
            match = visible.find(el => (el.innerText || '').includes(target));
          }}
          if (!match) return null;
          match.scrollIntoView({{block: 'center', inline: 'center'}});
          const rect = match.getBoundingClientRect();
          return {{x: rect.left + rect.width / 2, y: rect.top + rect.height / 2, disabled: !!match.disabled}};
        }})()
        """
        coords = self.evaluate(js)
        if not coords:
            raise AssertionError(f"Clickable element containing {text!r} not found")
        if coords.get("disabled"):
            raise AssertionError(f"Clickable element {text!r} is disabled")
        x, y = float(coords["x"]), float(coords["y"])
        self._send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        self._send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
        self._send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})
        self.pump_events(0.15)

    def set_radix_slider(
        self,
        label_substring: str,
        key: Optional[str] = None,
        repeat: int = 1,
        fraction: Optional[float] = None,
    ) -> Optional[str]:
        """Drives a Radix [role='slider'] associated with label_substring via keyboard or pointer."""
        js_focus = f"""
        (() => {{
          const label = {json.dumps(label_substring)};
          const rows = Array.from(document.querySelectorAll('aside div'));
          const row = rows.find(div => {{
            const txt = div.innerText || '';
            return txt.includes(label) && div.querySelector('[role="slider"]') && div.querySelectorAll('[role="slider"]').length === 1;
          }});
          if (!row) return null;
          const slider = row.querySelector('[role="slider"]');
          const track = (slider.parentElement && slider.parentElement.closest('[data-orientation]')) || row.querySelector('[data-orientation]') || slider.parentElement;
          slider.scrollIntoView({{block: 'center'}});
          slider.focus();
          const tRect = track.getBoundingClientRect();
          return {{
            x: tRect.left,
            y: tRect.top + tRect.height / 2,
            width: tRect.width,
            valuenow: slider.getAttribute('aria-valuenow')
          }};
        }})()
        """
        info = self.evaluate(js_focus)
        if not info:
            raise AssertionError(f"Radix slider for {label_substring!r} not found")

        if fraction is not None:
            frac = max(0.0, min(1.0, float(fraction)))
            px = float(info["x"]) + float(info["width"]) * frac
            py = float(info["y"])
            self._send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": px, "y": py})
            self._send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": px, "y": py, "button": "left", "clickCount": 1})
            self._send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": px, "y": py, "button": "left", "clickCount": 1})
            self.pump_events(0.2)

        if key is not None:
            vk_map = {
                "Home": 36,
                "End": 35,
                "ArrowLeft": 37,
                "ArrowUp": 38,
                "ArrowRight": 39,
                "ArrowDown": 40,
                "PageUp": 33,
                "PageDown": 34,
            }
            vk = vk_map.get(key, 0)
            for _ in range(max(1, repeat)):
                self._send(
                    "Input.dispatchKeyEvent",
                    {"type": "rawKeyDown", "key": key, "code": key, "windowsVirtualKeyCode": vk},
                )
                self._send(
                    "Input.dispatchKeyEvent",
                    {"type": "keyUp", "key": key, "code": key, "windowsVirtualKeyCode": vk},
                )
            self.pump_events(0.2)

        return self.evaluate(
            f"""
            (() => {{
              const label = {json.dumps(label_substring)};
              const rows = Array.from(document.querySelectorAll('aside div'));
              const row = rows.find(div => (div.innerText || '').includes(label) && div.querySelectorAll('[role="slider"]').length === 1);
              return row ? row.querySelector('[role="slider"]').getAttribute('aria-valuenow') : null;
            }})()
            """
        )

    def get_computed_style(self, text_or_selector: str, by_text: bool = False) -> Optional[Dict[str, Any]]:
        js = f"""
        (() => {{
          const q = {json.dumps(text_or_selector)};
          const byText = {json.dumps(by_text)};
          let el = null;
          if (!byText) {{
            el = document.querySelector(q);
          }} else {{
            const nodes = Array.from(document.querySelectorAll('aside *'));
            el = nodes.filter(n => n.children.length === 0 && (n.textContent || '').includes(q)).pop() || null;
          }}
          if (!el) return null;
          const cs = window.getComputedStyle(el);
          const rect = el.getBoundingClientRect();
          const aside = document.querySelector('aside');
          const asideRect = aside ? aside.getBoundingClientRect() : null;
          return {{
            whiteSpace: cs.whiteSpace,
            overflowX: cs.overflowX,
            textOverflow: cs.textOverflow,
            scrollWidth: el.scrollWidth,
            clientWidth: el.clientWidth,
            rectRight: rect.right,
            asideRight: asideRect ? asideRect.right : null,
            textContent: (el.textContent || '').trim()
          }};
        }})()
        """
        return self.evaluate(js)

    def wait_for_api_request(
        self,
        url_substring: str,
        min_index: int = 0,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for item in self.captured_api_requests[min_index:]:
                if url_substring in item["url"]:
                    return item
            self.pump_events(0.1)
        raise TimeoutError(f"No {url_substring!r} API request observed after index {min_index}")

    def wait_for_api_response(
        self,
        url_substring: str,
        min_index: int = 0,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for item in self.network_responses[min_index:]:
                if url_substring in item["url"]:
                    return item
            self.pump_events(0.1)
        raise TimeoutError(f"No {url_substring!r} API response observed after index {min_index}")

    def set_file_input(self, selector: str, file_path: str) -> None:
        """Uploads a local file into an <input type='file'> element via CDP DOM.setFileInputFiles."""
        doc = self._send("DOM.getDocument", {"depth": 1})
        root_node_id = doc["root"]["nodeId"]
        node = self._send("DOM.querySelector", {"nodeId": root_node_id, "selector": selector})
        node_id = int(node.get("nodeId", 0))
        if node_id <= 0:
            raise AssertionError(f"File input element matching {selector!r} not found")
        abs_path = str(Path(file_path).resolve())
        self._send("DOM.setFileInputFiles", {"nodeId": node_id, "files": [abs_path]})
        self.pump_events(0.25)

    def close(self) -> None:
        try:
            if self._browser_ws is not None:
                self._browser_ws.send(json.dumps({"id": 999999, "method": "Browser.close"}))
        except Exception:
            pass
        try:
            self._ws_stack.close()
        except Exception:
            pass
        self._page_ws = None
        self._browser_ws = None
        if self._proc is not None:
            proc = self._proc
            if proc.poll() is None:
                try:
                    proc.wait(timeout=2.5)
                except subprocess.TimeoutExpired:
                    pass
            if proc.poll() is None:
                _kill_process_tree(proc)
            elif os.name == "nt":
                # Reap any lingering Chromium child processes (e.g. Dawn/GPU cache writers)
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=5.0,
                        check=False,
                    )
                except Exception:
                    pass
            if proc.stderr is not None:
                try:
                    proc.stderr.close()
                except Exception:
                    pass
            self._proc = None
        for attempt in range(20):
            try:
                if os.path.isdir(self._tmp_dir.name):
                    shutil.rmtree(self._tmp_dir.name, ignore_errors=False)
                self._tmp_dir.cleanup()
                break
            except PermissionError:
                if attempt < 19:
                    time.sleep(0.15)
        else:
            shutil.rmtree(self._tmp_dir.name, ignore_errors=True)
            self._tmp_dir.cleanup()


@pytest.fixture(scope="session")
def frontend_built() -> Path:
    """Ensures frontend production bundle (frontend/dist) is built once before ephemeral_server."""
    if os.environ.get("PROMOB_FRONTEND_PREBUILT") == "1":
        dist = REPO_ROOT / "frontend" / "dist"
        assert (dist / "index.html").is_file(), "Build the frontend before using PROMOB_FRONTEND_PREBUILT=1"
        return dist
    cmd: Union[str, List[str]] = (
        "npm --prefix frontend run build"
        if os.name == "nt"
        else ["npm", "--prefix", "frontend", "run", "build"]
    )
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180.0,
        shell=(os.name == "nt"),
        check=False,
    )
    assert proc.returncode == 0, (
        f"Frontend build failed (exit {proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    dist_index = REPO_ROOT / "frontend" / "dist" / "index.html"
    assert dist_index.is_file(), f"Expected built index.html at {dist_index}"
    return REPO_ROOT / "frontend" / "dist"


@pytest.fixture(scope="session")
def ephemeral_server(frontend_built: Path):
    """Launches a real ephemeral uvicorn server serving frontend/dist + /api/* on a free port."""
    import httpx

    port = _get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "backend")

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "errors": "replace",
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        **popen_kwargs,
    )

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    def _drain_stream(stream, target_list: List[str]) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                target_list.append(line)
                if len(target_list) > 200:
                    del target_list[:100]
        except Exception:
            pass

    threading.Thread(target=_drain_stream, args=(proc.stdout, stdout_lines), daemon=True).start()
    threading.Thread(target=_drain_stream, args=(proc.stderr, stderr_lines), daemon=True).start()

    try:
        ready = False
        deadline = time.time() + 45.0
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"uvicorn exited early ({proc.returncode}):\n"
                    f"{''.join(stdout_lines)}\n{''.join(stderr_lines)}"
                )
            try:
                resp = httpx.get(f"{base_url}/api/health", timeout=1.0)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.2)

        if not ready:
            tail_out = "".join(stdout_lines[-20:])
            tail_err = "".join(stderr_lines[-20:])
            raise RuntimeError(
                f"Ephemeral uvicorn server did not become ready at {base_url}\n"
                f"STDOUT tail:\n{tail_out}\nSTDERR tail:\n{tail_err}"
            )

        yield base_url
    finally:
        _kill_process_tree(proc)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


@pytest.fixture
def cdp_browser(ephemeral_server: str):
    """Runtime-probed headless Chrome/Edge CDP browser connected to ephemeral_server."""
    binary = find_chromium_binary()
    if not binary:
        pytest.skip("No Chrome/Edge binary found for browser E2E")

    browser = CdpBrowser(binary_path=binary, base_url=ephemeral_server)
    try:
        yield browser
    finally:
        browser.close()


_SITECUSTOMIZE_SOCKET_AUDIT = r'''
import ipaddress
import json
import os
import socket
import time

_LOG_PATH = os.environ.get("PROMOB_SOCKET_AUDIT_LOG", "")


def _decode_host(host) -> str:
    if isinstance(host, (bytes, bytearray)):
        return host.decode("ascii", errors="ignore")
    return str(host)


def _is_loopback_host(host: str) -> bool:
    h = (host or "").strip().lower()
    if h in ("", "localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _record(kind: str, host: str, port, blocked: bool) -> None:
    if not _LOG_PATH:
        return
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": time.time(),
                "kind": kind,
                "host": str(host),
                "port": port,
                "blocked": blocked,
            }) + "\n")
    except Exception:
        pass


_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_conn = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo


def _guarded_connect(self, address):
    if getattr(self, "family", None) == getattr(socket, "AF_UNIX", -1):
        return _orig_connect(self, address)
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
        host_str = _decode_host(host)
        if not _is_loopback_host(host_str):
            _record("socket.connect", host_str, port, True)
            raise PermissionError(f"Offline socket audit blocked outbound TCP connect to {host_str}:{port}")
        _record("socket.connect", host_str, port, False)
    return _orig_connect(self, address)


def _guarded_connect_ex(self, address):
    if getattr(self, "family", None) == getattr(socket, "AF_UNIX", -1):
        return _orig_connect_ex(self, address)
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
        host_str = _decode_host(host)
        if not _is_loopback_host(host_str):
            _record("socket.connect_ex", host_str, port, True)
            raise PermissionError(f"Offline socket audit blocked outbound TCP connect_ex to {host_str}:{port}")
        _record("socket.connect_ex", host_str, port, False)
    return _orig_connect_ex(self, address)


def _guarded_create_connection(address, *args, **kwargs):
    host, port = address[0], address[1]
    host_str = _decode_host(host)
    if not _is_loopback_host(host_str):
        _record("socket.create_connection", host_str, port, True)
        raise PermissionError(f"Offline socket audit blocked outbound create_connection to {host_str}:{port}")
    _record("socket.create_connection", host_str, port, False)
    return _orig_create_conn(address, *args, **kwargs)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    if host is not None:
        host_str = _decode_host(host)
        if not _is_loopback_host(host_str):
            _record("socket.getaddrinfo", host_str, port, True)
            raise PermissionError(f"Offline socket audit blocked DNS lookup for {host_str}:{port}")
        _record("socket.getaddrinfo", host_str, port, False)
    return _orig_getaddrinfo(host, port, *args, **kwargs)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.create_connection = _guarded_create_connection
socket.getaddrinfo = _guarded_getaddrinfo
'''


class OfflineAuditedServer:
    """Manages an offline ephemeral uvicorn server with a socket-audit hook blocking non-loopback TCP."""

    def __init__(self, extra_env: Optional[Dict[str, str]] = None):
        import httpx

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="promob_offline_")
        hook_dir = Path(self._tmp_dir.name)
        (hook_dir / "sitecustomize.py").write_text(_SITECUSTOMIZE_SOCKET_AUDIT, encoding="utf-8")
        self.log_path = hook_dir / "socket_audit.jsonl"

        self.port = _get_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["PROMOB_SOCKET_AUDIT_LOG"] = str(self.log_path)
        if extra_env:
            env.update(extra_env)
        env["PYTHONPATH"] = f"{hook_dir}{os.pathsep}{REPO_ROOT / 'backend'}"

        popen_kwargs: Dict[str, Any] = {
            "cwd": str(REPO_ROOT),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "errors": "replace",
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            **popen_kwargs,
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def _drain(stream, target: List[str]) -> None:
            if stream is None:
                return
            try:
                for line in stream:
                    target.append(line)
                    if len(target) > 200:
                        del target[:100]
            except Exception:
                pass

        threading.Thread(target=_drain, args=(self._proc.stdout, stdout_lines), daemon=True).start()
        threading.Thread(target=_drain, args=(self._proc.stderr, stderr_lines), daemon=True).start()

        deadline = time.time() + 45.0
        ready = False
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self.close()
                raise RuntimeError(
                    f"Offline uvicorn exited early ({self._proc.returncode}):\n"
                    f"{''.join(stdout_lines)}\n{''.join(stderr_lines)}"
                )
            try:
                resp = httpx.get(f"{self.base_url}/api/health", timeout=1.0)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.2)

        if not ready:
            tail_out = "".join(stdout_lines[-20:])
            tail_err = "".join(stderr_lines[-20:])
            self.close()
            raise RuntimeError(
                f"Offline audited uvicorn did not become ready at {self.base_url}\n"
                f"STDOUT tail:\n{tail_out}\nSTDERR tail:\n{tail_err}"
            )

    def get_audit_events(self) -> List[Dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        events: List[Dict[str, Any]] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events

    def get_outbound_attempts(self) -> List[Dict[str, Any]]:
        return [ev for ev in self.get_audit_events() if ev.get("blocked")]

    def close(self) -> None:
        _kill_process_tree(self._proc)
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        for attempt in range(5):
            try:
                self._tmp_dir.cleanup()
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.1)


@pytest.fixture
def offline_audited_server():
    """Starts a fresh offline, socket-audited uvicorn server for zero-external-network verification."""
    srv = OfflineAuditedServer()
    try:
        yield srv
    finally:
        srv.close()


@pytest.fixture
def offline_audited_server_factory():
    """Factory creating custom-env offline socket-audited uvicorn servers (e.g., invalid HF_HOME)."""
    created: List[OfflineAuditedServer] = []

    def _factory(extra_env: Optional[Dict[str, str]] = None) -> OfflineAuditedServer:
        srv = OfflineAuditedServer(extra_env=extra_env)
        created.append(srv)
        return srv

    try:
        yield _factory
    finally:
        for srv in created:
            srv.close()
