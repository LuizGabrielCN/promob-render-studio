"""
Unit tests for HardwareGuardrails dynamic VRAM scaling & safe memory management.
"""
from unittest.mock import patch
import pytest
from app.core.hardware import HardwareGuardrails


@patch.object(HardwareGuardrails, "get_vram_status")
def test_high_vram_selects_cuda_512(mock_status):
    """VRAM >= 3000MB must select CUDA with 512x512 tile size and 32px overlap."""
    mock_status.return_value = {
        "cuda_available": True,
        "free_mb": 4200.0,
        "total_mb": 6144.0,
        "device_name": "NVIDIA GeForce RTX 2060"
    }
    device, tile_size, overlap = HardwareGuardrails.determine_execution_strategy(512)
    assert device == "cuda"
    assert tile_size == 512
    assert overlap == 32


@patch.object(HardwareGuardrails, "get_vram_status")
def test_medium_vram_throttles_to_cuda_256(mock_status):
    """VRAM between 1500MB and 3000MB must throttle tile size to 256x256."""
    mock_status.return_value = {
        "cuda_available": True,
        "free_mb": 2100.0,
        "total_mb": 6144.0,
        "device_name": "NVIDIA GeForce RTX 2060"
    }
    device, tile_size, overlap = HardwareGuardrails.determine_execution_strategy(512)
    assert device == "cuda"
    assert tile_size == 256
    assert overlap == 16


@patch.object(HardwareGuardrails, "get_vram_status")
def test_low_vram_falls_back_to_cpu(mock_status):
    """VRAM < 1500MB must fall back to CPU execution."""
    mock_status.return_value = {
        "cuda_available": True,
        "free_mb": 900.0,
        "total_mb": 6144.0,
        "device_name": "NVIDIA GeForce RTX 2060"
    }
    device, tile_size, overlap = HardwareGuardrails.determine_execution_strategy(512)
    assert device == "cpu"
    assert tile_size == 256
    assert overlap == 16


@patch.object(HardwareGuardrails, "get_vram_status")
def test_cuda_unavailable_falls_back_to_cpu(mock_status):
    """Missing CUDA must cleanly fall back to CPU."""
    mock_status.return_value = {
        "cuda_available": False,
        "free_mb": 0.0,
        "total_mb": 0.0,
        "device_name": "CPU"
    }
    device, tile_size, overlap = HardwareGuardrails.determine_execution_strategy(512)
    assert device == "cpu"
    assert tile_size == 256
    assert overlap == 16


def test_release_memory_safe():
    """release_memory() must execute cleanly without raising exceptions on any platform."""
    try:
        HardwareGuardrails.release_memory()
    except Exception as e:
        pytest.fail(f"release_memory raised unexpected exception: {e}")


def test_get_lock():
    """get_lock() must return a valid threading.Lock instance."""
    lock = HardwareGuardrails.get_lock()
    assert lock is not None
    assert hasattr(lock, "acquire")
    assert hasattr(lock, "release")
