"""
Unit tests for Stage 5 Tiling, 2D Hanning Feathering & Canvas Reconstruction.
"""
import os
import tempfile
import hashlib
import numpy as np
import pytest

from app.pipeline.super_res import (
    upscale_tiled,
    create_hanning_window,
    generate_hanning_window,
    decompose_tiles,
    compute_tile_coordinates,
    DEFAULT_TILE_SIZE,
    DEFAULT_PADDING,
    MODEL_SHA256,
    MODEL_EXPECTED_SIZE,
    verify_file_sha256,
)


def test_hanning_window_symmetric():
    """2D Hanning window must be symmetric and peak at the center."""
    w2d = create_hanning_window(128, 128)
    assert w2d.shape == (128, 128)
    assert np.allclose(w2d, w2d.T)
    center_val = w2d[64, 64]
    edge_val = w2d[0, 0]
    assert center_val > edge_val


def test_hanning_window_properties():
    """Verifies 2D Hanning window shape, symmetry, and positive floor."""
    th, tw = 64, 64
    scale = 4
    win = generate_hanning_window(th, tw, scale=scale)
    assert win.shape == (th * scale, tw * scale, 1)
    assert np.all(win >= 0.02), "Window floor was violated"
    assert np.all(win <= 1.0), "Window maximum exceeded 1.0"
    assert np.allclose(win, win[::-1, :, :], atol=1e-5)
    assert np.allclose(win, win[:, ::-1, :], atol=1e-5)


def test_decompose_tiles_covers_entire_canvas(non_power_of_two_bgr):
    """Tile decomposition must cover 100% of the canvas without omitting pixels."""
    h, w = non_power_of_two_bgr.shape[:2]
    tile_size, overlap = 128, 32
    tiles = decompose_tiles(h, w, tile_size=tile_size, overlap=overlap)

    covered = np.zeros((h, w), dtype=bool)
    for (y, x, th, tw) in tiles:
        covered[y:y+th, x:x+tw] = True
    assert covered.all(), "Tiling grid failed to cover the entire image canvas!"


def test_tile_coordinate_coverage():
    """Verifies that tile decomposition covers 100% of image pixels without gaps."""
    test_dims = [(100, 100), (256, 256), (723, 900), (1080, 1920), (2160, 3840)]
    for h, w in test_dims:
        tiles = compute_tile_coordinates(h, w, tile_size=256, padding=32)
        coverage = np.zeros((h, w), dtype=np.int32)
        for y0, y1, x0, x1 in tiles:
            assert 0 <= y0 < y1 <= h
            assert 0 <= x0 < x1 <= w
            coverage[y0:y1, x0:x1] += 1
        assert np.all(coverage >= 1), f"Coverage gap found for image dimensions {w}x{h}"


def test_tile_coordinates_invalid_step():
    """Verifies ValueError when padding >= tile_size / 2."""
    with pytest.raises(ValueError):
        compute_tile_coordinates(500, 500, tile_size=128, padding=64)


def test_uniform_canvas_reconstruction():
    """Verifies that accumulating and normalizing a constant image yields zero reconstruction error."""
    h, w = 300, 400
    scale = 4
    tiles = compute_tile_coordinates(h, w, tile_size=256, padding=32)
    canvas = np.zeros((h * scale, w * scale, 1), dtype=np.float32)
    weights = np.zeros((h * scale, w * scale, 1), dtype=np.float32)

    val = 0.75
    for y0, y1, x0, x1 in tiles:
        th, tw = y1 - y0, x1 - x0
        tile_val = np.full((th * scale, tw * scale, 1), val, dtype=np.float32)
        w_tile = generate_hanning_window(th, tw, scale=scale)
        canvas[y0*scale:y1*scale, x0*scale:x1*scale, :] += tile_val * w_tile
        weights[y0*scale:y1*scale, x0*scale:x1*scale, :] += w_tile

    reconstructed = canvas / (weights + 1e-7)
    max_err = np.max(np.abs(reconstructed - val))
    assert max_err < 1e-5, f"Accumulation normalization error too large: {max_err}"


def test_identity_tiling_reconstruction_seamless(non_power_of_two_bgr):
    """
    Reconstructing an image via identity tile processing with Hanning feathering
    must yield machine-precision zero-seam equivalence (MAE < 1e-4).
    """
    h, w, c = non_power_of_two_bgr.shape
    f_img = non_power_of_two_bgr.astype(np.float32) / 255.0

    recon, tile_count = upscale_tiled(non_power_of_two_bgr, scale=1, tile_size=128, overlap=32)

    recon_f = recon.astype(np.float32) / 255.0
    mae = np.mean(np.abs(f_img - recon_f))
    assert mae < 1e-4, f"Seam artifacts detected! MAE={mae}"
    assert tile_count > 1


def test_model_hash_verification():
    """Verifies SHA-256 integrity checker logic."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test content for hash verification")
        tmp_name = f.name
    try:
        expected = hashlib.sha256(b"test content for hash verification").hexdigest()
        assert verify_file_sha256(tmp_name, expected) is True
        assert verify_file_sha256(tmp_name, "invalid_hash_string") is False
        assert verify_file_sha256("non_existent_file.tmp", expected) is False
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
