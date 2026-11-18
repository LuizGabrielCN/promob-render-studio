"""
Unit tests for R4 EnhancementConfig contract compatibility & validation,
and HardwareGuardrails VRAM cache invalidation + generation-guard concurrency safety.
"""
import threading
import time
from unittest.mock import patch
import pytest
from pydantic import ValidationError
from app.core.interfaces import EnhancementConfig
from app.core.hardware import HardwareGuardrails


def test_enhancement_config_defaults_backward_compatible():
    """All 13 legacy defaults and 4 new defaults must remain backward compatible."""
    cfg = EnhancementConfig()
    # 13 legacy fields
    assert cfg.temperature == 0.0
    assert cfg.tint == 0.0
    assert cfg.exposure == 0.0
    assert cfg.shadow_lift == 0.20
    assert cfg.highlight_recovery == 0.12
    assert cfg.clahe_clip_limit == 2.0
    assert cfg.clahe_grid_size == 8
    assert cfg.ambient_occlusion == 0.15
    assert cfg.sharpness == 0.40
    assert cfg.denoise_strength == 0.10
    assert cfg.upscale_factor == 4
    assert cfg.tile_size == 256
    assert cfg.preserve_geometry is True

    # 4 new R4 fields
    assert cfg.render_mode == "classic"
    assert cfg.generative_strength == 0.48
    assert cfg.scene_profile == "auto"
    assert cfg.specular_bloom == 0.25


def test_enhancement_config_valid_render_modes():
    """Both 'classic' and 'generative' render modes must be accepted."""
    assert EnhancementConfig(render_mode="classic").render_mode == "classic"
    assert EnhancementConfig(render_mode="generative").render_mode == "generative"


@pytest.mark.parametrize("bad_mode", ["invalid", "", "CLASSIC", "Generative", 123])
def test_enhancement_config_invalid_render_mode_raises(bad_mode):
    """Unsupported render_mode values must fail Pydantic validation."""
    with pytest.raises(ValidationError):
        EnhancementConfig(render_mode=bad_mode)


@pytest.mark.parametrize("profile", ["auto", "banheiro", "cozinha", "quarto_sala"])
def test_enhancement_config_valid_scene_profiles(profile):
    """All four documented architectural scene profiles must be accepted."""
    assert EnhancementConfig(scene_profile=profile).scene_profile == profile


@pytest.mark.parametrize("bad_profile", ["sala", "", "AUTO", "banheiro_suite", 99])
def test_enhancement_config_invalid_scene_profile_raises(bad_profile):
    """Unsupported scene_profile values must fail Pydantic validation."""
    with pytest.raises(ValidationError):
        EnhancementConfig(scene_profile=bad_profile)


def test_enhancement_config_generative_strength_and_bloom_bounds():
    """generative_strength and specular_bloom must enforce [0.0, 1.0] inclusive bounds."""
    cfg_min = EnhancementConfig(generative_strength=0.0, specular_bloom=0.0)
    cfg_max = EnhancementConfig(generative_strength=1.0, specular_bloom=1.0)
    assert cfg_min.generative_strength == 0.0
    assert cfg_min.specular_bloom == 0.0
    assert cfg_max.generative_strength == 1.0
    assert cfg_max.specular_bloom == 1.0

    for bad_val in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            EnhancementConfig(generative_strength=bad_val)
        with pytest.raises(ValidationError):
            EnhancementConfig(specular_bloom=bad_val)


def test_enhancement_config_partial_legacy_kwargs():
    """Legacy callers passing partial subsets must receive valid defaults for new fields."""
    cfg = EnhancementConfig(temperature=0.2, upscale_factor=1)
    assert cfg.temperature == 0.2
    assert cfg.upscale_factor == 1
    assert cfg.render_mode == "classic"
    assert cfg.generative_strength == 0.48
    assert cfg.scene_profile == "auto"
    assert cfg.specular_bloom == 0.25


@pytest.fixture
def reset_vram_cache():
    """Isolates HardwareGuardrails class-level cache state around each test."""
    with HardwareGuardrails._vram_cache_lock:
        HardwareGuardrails._vram_cache = None
        HardwareGuardrails._vram_cache_ts = 0.0
        HardwareGuardrails._vram_refreshing = False
    yield
    with HardwareGuardrails._vram_cache_lock:
        HardwareGuardrails._vram_cache = None
        HardwareGuardrails._vram_cache_ts = 0.0
        HardwareGuardrails._vram_refreshing = False


def test_release_memory_invalidates_vram_cache(reset_vram_cache):
    """release_memory() must clear _vram_cache so immediate callers observe fresh VRAM."""
    readings = [
        {"cuda_available": True, "free_mb": 1000.0, "total_mb": 6144.0, "device_name": "RTX 2060"},
        {"cuda_available": True, "free_mb": 4200.0, "total_mb": 6144.0, "device_name": "RTX 2060"},
    ]
    with patch.object(HardwareGuardrails, "_query_vram_status", side_effect=readings) as mock_q:
        first = HardwareGuardrails.get_vram_status(max_age_seconds=2.0)
        cached = HardwareGuardrails.get_vram_status(max_age_seconds=2.0)
        assert first["free_mb"] == 1000.0
        assert cached["free_mb"] == 1000.0
        assert mock_q.call_count == 1

        HardwareGuardrails.release_memory()
        assert HardwareGuardrails._vram_cache is None
        assert HardwareGuardrails._vram_cache_ts == 0.0

        fresh = HardwareGuardrails.get_vram_status(max_age_seconds=2.0)
        assert fresh["free_mb"] == 4200.0
        assert mock_q.call_count == 2


def test_determine_execution_strategy_sees_fresh_vram_after_release(reset_vram_cache):
    """determine_execution_strategy() must promote from CPU to CUDA 512 immediately after release_memory()."""
    readings = [
        {"cuda_available": True, "free_mb": 900.0, "total_mb": 6144.0, "device_name": "RTX 2060"},
        {"cuda_available": True, "free_mb": 4200.0, "total_mb": 6144.0, "device_name": "RTX 2060"},
    ]
    with patch.object(HardwareGuardrails, "_query_vram_status", side_effect=readings):
        assert HardwareGuardrails.determine_execution_strategy(512) == ("cpu", 256, 16)
        HardwareGuardrails.release_memory()
        assert HardwareGuardrails.determine_execution_strategy(512) == ("cuda", 512, 32)


def test_release_memory_prevents_stale_async_refresh_overwrite(reset_vram_cache):
    """
    An in-flight _refresh_vram_cache_async thread started before release_memory()
    must not overwrite _vram_cache with stale pre-release VRAM after release_memory() runs.
    """
    with HardwareGuardrails._vram_cache_lock:
        HardwareGuardrails._vram_cache = {
            "cuda_available": True,
            "free_mb": 1100.0,
            "total_mb": 6144.0,
            "device_name": "RTX 2060",
        }
        HardwareGuardrails._vram_cache_ts = time.monotonic() - 10.0

    query_started = threading.Event()
    allow_query_finish = threading.Event()

    def slow_stale_query():
        query_started.set()
        allow_query_finish.wait(timeout=2.0)
        return {
            "cuda_available": True,
            "free_mb": 1100.0,
            "total_mb": 6144.0,
            "device_name": "RTX 2060",
        }

    with patch.object(HardwareGuardrails, "_query_vram_status", side_effect=slow_stale_query):
        # Triggers background _refresh_vram_cache_async
        stale_returned = HardwareGuardrails.get_vram_status(max_age_seconds=2.0)
        assert stale_returned["free_mb"] == 1100.0
        assert query_started.wait(timeout=2.0)

        # Invalidate cache while background query is mid-flight
        HardwareGuardrails.release_memory()
        assert HardwareGuardrails._vram_cache is None

        # Let stale background query finish and wait for thread to exit
        allow_query_finish.set()
        deadline = time.monotonic() + 2.0
        while HardwareGuardrails._vram_refreshing and time.monotonic() < deadline:
            time.sleep(0.01)

    assert HardwareGuardrails._vram_refreshing is False
    assert HardwareGuardrails._vram_cache is None


def test_vram_refresh_thread_start_failure_resets_flag(reset_vram_cache):
    """If starting the background VRAM refresh thread raises, _vram_refreshing must reset to False."""
    with HardwareGuardrails._vram_cache_lock:
        HardwareGuardrails._vram_cache = {
            "cuda_available": True,
            "free_mb": 2000.0,
            "total_mb": 6144.0,
            "device_name": "RTX 2060",
        }
        HardwareGuardrails._vram_cache_ts = time.monotonic() - 10.0

    with patch("app.core.hardware.threading.Thread.start", side_effect=RuntimeError("cannot start thread")):
        status = HardwareGuardrails.get_vram_status(max_age_seconds=2.0)

    assert status["free_mb"] == 2000.0
    assert HardwareGuardrails._vram_refreshing is False
