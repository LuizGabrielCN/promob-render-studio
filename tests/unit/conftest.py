"""
Pytest fixtures for unit testing the Promob Render Enhancer pipeline.
Generates deterministic synthetic image fixtures without relying on external assets.
"""
import sys
import os
import pytest
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

@pytest.fixture(autouse=True)
def disable_real_object_detection(monkeypatch):
    """Keep mocked unit pipelines offline without disabling detection in integration/E2E tests."""
    monkeypatch.setenv("PROMOB_OBJECT_DETECTION", "0")

@pytest.fixture
def synthetic_flat_bgr():
    """Returns a 128x128 uniform neutral gray image in BGR format."""
    return np.full((128, 128, 3), 128, dtype=np.uint8)


@pytest.fixture
def synthetic_gradient_bgr():
    """Returns a 256x256 horizontal grayscale gradient spanning [0, 255]."""
    grad_1d = np.linspace(0, 255, 256, dtype=np.uint8)
    grad_2d = np.tile(grad_1d, (256, 1))
    return np.stack([grad_2d, grad_2d, grad_2d], axis=-1)


@pytest.fixture
def synthetic_crevice_bgr():
    """
    Returns a 128x128 light gray image with a 2-pixel wide dark vertical crevice
    in the center (ideal for testing Morphological Ambient Occlusion).
    """
    img = np.full((128, 128, 3), 200, dtype=np.uint8)
    img[:, 63:65, :] = 40  # Deep narrow crease
    return img


@pytest.fixture
def synthetic_checkerboard_bgr():
    """Returns a 128x128 sharp checkerboard pattern with headroom for sharpness/micro-contrast testing."""
    block_size = 16
    grid = (np.indices((128, 128)) // block_size).sum(axis=0) % 2
    pattern = (60 + grid * 135).astype(np.uint8)
    return np.stack([pattern, pattern, pattern], axis=-1)


@pytest.fixture
def non_power_of_two_bgr():
    """Returns an irregularly sized image (237x319x3) to test tiling boundary handling."""
    np.random.seed(1337)
    return np.random.randint(20, 230, (237, 319, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_u8():
    """Returns an arbitrary uint8 BGR image for photometric verification."""
    np.random.seed(42)
    return np.random.randint(20, 230, (120, 160, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_f32(sample_bgr_u8):
    """Returns a float32 normalized [0.0, 1.0] BGR image."""
    return sample_bgr_u8.astype(np.float32) / 255.0


@pytest.fixture
def sample_bgra_u8():
    """Returns an arbitrary 4-channel BGRA uint8 image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
