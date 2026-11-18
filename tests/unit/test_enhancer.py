"""
Unit tests for ClassicEnhancer implementation.
Verifies defensive input validation, preview latency (<100ms), and alpha channel preservation.
"""
from unittest.mock import patch
import time
import pytest
import numpy as np

from app.core.interfaces import EnhancementConfig, EnhancementResult
from app.pipeline.enhancer import ClassicEnhancer


def test_process_preview_invalid_inputs_raise_value_error():
    """Verifies that invalid or malformed inputs cleanly raise ValueError in process_preview."""
    enhancer = ClassicEnhancer()
    config = EnhancementConfig()

    invalid_inputs = [
        None,
        "string_path_not_allowed.jpg",
        [1, 2, 3],
        np.zeros(100, dtype=np.uint8),              # 1D array
        np.zeros((100, 100), dtype=np.uint8),        # 2D array
        np.zeros((0, 0, 3), dtype=np.uint8),         # 0x0x3 empty array
        np.zeros((10, 10, 1), dtype=np.uint8),       # Single-channel array
        np.zeros((10, 10, 5), dtype=np.uint8),       # 5-channel array
    ]

    for item in invalid_inputs:
        with pytest.raises(ValueError, match="Input image cannot be None or empty|Expected 3-channel"):
            enhancer.process_preview(item, config)


def test_process_invalid_inputs_raise_value_error():
    """Verifies that invalid or malformed inputs cleanly raise ValueError in full process."""
    enhancer = ClassicEnhancer()
    config = EnhancementConfig()

    invalid_inputs = [
        None,
        "invalid_input",
        [255, 255, 255],
        np.zeros(50, dtype=np.uint8),
        np.zeros((50, 50), dtype=np.uint8),
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((20, 20, 2), dtype=np.uint8),
    ]

    for item in invalid_inputs:
        with pytest.raises(ValueError, match="Input image cannot be None or empty|Expected 3-channel"):
            enhancer.process(item, config)


def test_process_preview_latency_under_100ms():
    """Verifies that process_preview on a 1920x1080 image executes in <100ms."""
    enhancer = ClassicEnhancer()
    config = EnhancementConfig(
        temperature=0.2,
        tint=0.1,
        exposure=0.1,
        shadow_lift=0.25,
        highlight_recovery=0.15,
        clahe_clip_limit=2.0,
        ambient_occlusion=0.2,
        sharpness=0.4,
        denoise_strength=0.1,
    )

    # 1920x1080 typical Promob render resolution
    np.random.seed(42)
    sample_fhd = np.random.randint(30, 220, (1080, 1920, 3), dtype=np.uint8)

    # Warmup invocation to initialize OpenCV internal dispatch
    _ = enhancer.process_preview(sample_fhd, config)

    # Benchmark run
    t0 = time.perf_counter()
    out = enhancer.process_preview(sample_fhd, config)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.100, f"Preview latency {elapsed * 1000:.2f}ms exceeded 100ms budget!"
    assert out.shape[0] <= 960 and out.shape[1] <= 960, f"Proxy was not correctly bounded: {out.shape}"
    assert out.dtype == np.uint8


def test_process_preview_preserves_bgra():
    """Verifies that process_preview preserves alpha transparency channel in 4-channel BGRA."""
    enhancer = ClassicEnhancer()
    config = EnhancementConfig(
        temperature=0.1,
        shadow_lift=0.2,
        clahe_clip_limit=1.5,
        ambient_occlusion=0.15,
        sharpness=0.3,
    )

    # Small image (no downscale proxy) to check exact pixel alpha preservation
    np.random.seed(123)
    bgra_small = np.random.randint(0, 256, (128, 128, 4), dtype=np.uint8)
    out_small = enhancer.process_preview(bgra_small, config)

    assert out_small.shape == (128, 128, 4)
    np.testing.assert_array_equal(out_small[:, :, 3], bgra_small[:, :, 3])

    # Downscaled proxy image to check 4-channel integrity
    bgra_fhd = np.random.randint(0, 256, (1080, 1920, 4), dtype=np.uint8)
    out_fhd = enhancer.process_preview(bgra_fhd, config)

    assert out_fhd.ndim == 3
    assert out_fhd.shape[2] == 4
    assert out_fhd.shape[1] == 960


def test_process_native_scale_1(synthetic_checkerboard_bgr):
    """Verifies that process with upscale_factor=1 executes photometric pipeline and returns EnhancementResult."""
    enhancer = ClassicEnhancer()
    config = EnhancementConfig(
        upscale_factor=1,
        shadow_lift=0.20,
        highlight_recovery=0.10,
        ambient_occlusion=0.10,
        sharpness=0.30,
    )

    callbacks_received = []

    def progress_cb(pct: float, stage: str):
        callbacks_received.append((pct, stage))

    out_bgr, result = enhancer.process(
        synthetic_checkerboard_bgr,
        config=config,
        progress_callback=progress_cb,
    )

    assert isinstance(result, EnhancementResult)
    assert out_bgr.shape == synthetic_checkerboard_bgr.shape
    assert result.output_width == synthetic_checkerboard_bgr.shape[1]
    assert result.output_height == synthetic_checkerboard_bgr.shape[0]
    assert result.processing_time_seconds > 0
    assert len(callbacks_received) > 0


def test_enhancer_capabilities():
    """Verifies that get_capabilities returns expected metadata dictionary."""
    enhancer = ClassicEnhancer()
    caps = enhancer.get_capabilities()

    assert caps["engine"] == "ClassicEnhancer"
    assert caps["version"] == "1.0.0"
    assert caps["preserves_geometry"] is True
    assert caps["deterministic"] is True
    assert "photometric_stages" in caps
    assert "pbr_soft_gi_bloom_specular" in caps["photometric_stages"]
    assert caps["render_modes"] == ["classic", "generative"]
    assert caps["scene_profiles"] == ["auto", "banheiro", "cozinha", "quarto_sala"]
    assert "generative_engine" in caps
    assert "hardware" in caps


def test_enhancer_no_25_percent_dilution_when_preserve_geometry_true():
    """preserve_geometry=True must not attenuate photometric/PBR effects with 0.75/0.25 blend."""
    enhancer = ClassicEnhancer()
    rng = np.random.default_rng(7)
    img = rng.integers(30, 220, (128, 128, 3), dtype=np.uint8)

    cfg_true = EnhancementConfig(
        upscale_factor=1,
        preserve_geometry=True,
        ambient_occlusion=0.60,
        specular_bloom=0.50,
    )
    cfg_false = EnhancementConfig(
        upscale_factor=1,
        preserve_geometry=False,
        ambient_occlusion=0.60,
        specular_bloom=0.50,
    )

    prev_true = enhancer.process_preview(img, cfg_true)
    prev_false = enhancer.process_preview(img, cfg_false)
    np.testing.assert_array_equal(prev_true, prev_false)

    out_true, _ = enhancer.process(img, cfg_true)
    out_false, _ = enhancer.process(img, cfg_false)
    np.testing.assert_array_equal(out_true, out_false)


def test_enhancer_process_generative_mode_invokes_generator_and_releases_vram():
    """render_mode='generative' invokes LocalControlNetGenerator and passes output to pipeline."""
    enhancer = ClassicEnhancer()
    img = np.full((128, 128, 3), 120, dtype=np.uint8)
    expected = np.full((128, 128, 3), 175, dtype=np.uint8)
    cfg = EnhancementConfig(
        render_mode="generative",
        generative_strength=0.48,
        scene_profile="banheiro",
        specular_bloom=0.25,
        upscale_factor=1,
        photo_finish=0.0,  # passthrough check; the photo finish has its own tests below
    )

    with patch("app.pipeline.enhancer.get_controlnet_generator") as mock_get_gen, \
         patch("app.pipeline.enhancer.HardwareGuardrails.release_memory") as mock_release:
        mock_gen = mock_get_gen.return_value
        mock_gen._pipeline_device = "cuda"
        mock_gen.generate.return_value = expected

        out, result = enhancer.process(img, cfg)

    mock_gen.generate.assert_called_once()
    assert mock_release.call_count >= 2
    assert result.metadata["render_mode"] == "generative"
    assert result.metadata["scene_profile"] == "banheiro"
    np.testing.assert_array_equal(out, expected)


def test_enhancer_generative_mode_anchors_clean_reference_and_forwards_preserve_geometry():
    """Generative mode must pass un-darkened s1 as reference_bgr (avoiding CLAHE -30 niche drop) and forward preserve_geometry."""
    enhancer = ClassicEnhancer()
    # Bright light oak interior simulation
    img = np.full((128, 128, 3), (180, 200, 215), dtype=np.uint8)
    img[20:108, 20:108] = (188, 208, 222)
    cfg = EnhancementConfig(
        render_mode="generative",
        generative_strength=0.48,
        scene_profile="auto",
        clahe_clip_limit=2.0,
        ambient_occlusion=0.25,
        specular_bloom=0.47,
        upscale_factor=1,
        preserve_geometry=True,
        photo_finish=0.0,  # passthrough check; the photo finish has its own tests below
    )

    with patch("app.pipeline.enhancer.get_controlnet_generator") as mock_get_gen, \
         patch("app.pipeline.enhancer.HardwareGuardrails.release_memory"):
        mock_gen = mock_get_gen.return_value
        mock_gen._pipeline_device = "cuda"
        mock_gen.generate.side_effect = lambda init_img, **kwargs: kwargs["reference_bgr"].copy()

        out, _ = enhancer.process(img, cfg)

    call_kwargs = mock_gen.generate.call_args.kwargs
    assert call_kwargs["preserve_geometry"] is True
    assert "reference_bgr" in call_kwargs
    ref_mean = float(call_kwargs["reference_bgr"].mean())
    init_mean = float(mock_gen.generate.call_args.args[0].mean())
    # Neither reference nor init should suffer CLAHE's ~30-level bright-panel darkening
    assert ref_mean > 195.0
    assert init_mean > 185.0
    np.testing.assert_array_equal(out, call_kwargs["reference_bgr"])


def test_enhancer_generative_forwards_new_flags_and_applies_photo_finish():
    """Passos 3/4: latent_object_lock / freeu / prompt_style reach generate(); the photo finish (default 0.5)
    changes the generated frame while keeping its mean brightness."""
    enhancer = ClassicEnhancer()
    rng = np.random.default_rng(0)
    gen_out = rng.integers(60, 200, size=(96, 128, 3)).astype(np.uint8)
    cfg = EnhancementConfig(render_mode="generative", upscale_factor=1, freeu=True, latent_object_lock=False,
                            prompt_style="legacy")
    defaults = EnhancementConfig()
    assert defaults.photo_finish == 0.5
    assert defaults.freeu is False and defaults.latent_object_lock is True and defaults.prompt_style == "legacy"

    with patch("app.pipeline.enhancer.get_controlnet_generator") as mock_get_gen, \
         patch("app.pipeline.enhancer.HardwareGuardrails.release_memory"):
        mock_gen = mock_get_gen.return_value
        mock_gen._pipeline_device = "cuda"
        mock_gen.last_run_info = {"latent_lock_steps": 0, "object_mask_source": "heuristic"}
        mock_gen.generate.return_value = gen_out
        out, result = enhancer.process(np.full((96, 128, 3), 120, np.uint8), cfg)

    kw = mock_gen.generate.call_args.kwargs
    assert kw["freeu"] is True and kw["latent_object_lock"] is False and kw["prompt_style"] == "legacy"
    assert result.metadata["photo_finish"] == 0.5
    assert result.metadata["object_mask_source"] == "heuristic"
    assert not np.array_equal(out, gen_out)
    assert abs(float(out.mean()) - float(gen_out.mean())) < 3.0


def test_enhancer_grain_is_applied_after_super_resolution():
    """Real-ESRGAN wipes grain out, so grain is added to the UPSCALED frame (tone stage goes before)."""
    enhancer = ClassicEnhancer()
    flat = np.full((64, 64, 3), 128, np.uint8)
    cfg = EnhancementConfig(render_mode="generative", upscale_factor=2, photo_finish=1.0)
    upscaler_inputs = []

    def fake_upscale(img, scale, **kwargs):
        upscaler_inputs.append(img.copy())
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1), 1

    with patch("app.pipeline.enhancer.get_controlnet_generator") as mock_get_gen, \
         patch("app.pipeline.enhancer.HardwareGuardrails.release_memory"), \
         patch("app.pipeline.enhancer.upscale_tiled", side_effect=fake_upscale):
        mock_gen = mock_get_gen.return_value
        mock_gen._pipeline_device = "cuda"
        mock_gen.last_run_info = {}
        mock_gen.generate.return_value = flat
        out, _ = enhancer.process(flat, cfg)

    assert out.shape == (128, 128, 3)
    up_in = upscaler_inputs[0].astype(np.float32)
    # Upscaler input: no per-pixel grain (only the smooth vignette); output: independent per-pixel grain
    # (a replicated 2x2 block would give a zero residual)
    assert float(np.abs(np.diff(up_in[:, :, 1], axis=1)).mean()) < 0.6
    o = out[:, :, 1].astype(np.float32)
    assert float((o[0::2, 0::2] != o[0::2, 1::2]).mean()) > 0.10


def test_enhancer_preserves_native_lettering_after_neural_upscale():
    import cv2

    native = np.full((64, 64, 3), 30, dtype=np.uint8)
    native[26:30, 24:40:3] = 200
    text_mask = np.zeros(native.shape[:2], dtype=np.float32)
    text_mask[22:34, 20:44] = 1.0
    neural_output = np.full((256, 256, 3), 125, dtype=np.uint8)
    config = EnhancementConfig(render_mode="generative", upscale_factor=4, photo_finish=0.0, led_strips=False)

    def generate(*args, text_logo_mask_callback, **kwargs):
        text_logo_mask_callback(text_mask)
        return native

    with patch("app.pipeline.enhancer.get_controlnet_generator") as get_generator, \
         patch("app.pipeline.enhancer.HardwareGuardrails.release_memory"), \
         patch("app.pipeline.enhancer.upscale_tiled", return_value=(neural_output.copy(), 1)):
        get_generator.return_value.generate.side_effect = generate
        output, _ = ClassicEnhancer().process(native, config)

    interpolated = cv2.resize(native, (256, 256), interpolation=cv2.INTER_CUBIC)
    np.testing.assert_array_equal(output[100:120, 92:164], interpolated[100:120, 92:164])
    np.testing.assert_array_equal(output[:80], neural_output[:80])


def test_enhancer_unpacks_three_tuple_led_markers_in_preview_and_process():
    import cv2
    enhancer = ClassicEnhancer()
    img = np.full((240, 360, 3), 60, dtype=np.uint8)
    cv2.line(img, (40, 120), (300, 120), (0, 220, 255), 3)

    prev = enhancer.process_preview(img, EnhancementConfig(led_strips=True))
    assert prev.shape == img.shape

    out, res = enhancer.process(img, EnhancementConfig(led_strips=True, upscale_factor=1))
    assert out.shape == img.shape
    assert res.metadata["led_strips_detected"] == 1


def test_enhancer_releases_super_res_before_generator_and_on_upscale_failure():
    enhancer = ClassicEnhancer()
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    cfg = EnhancementConfig(render_mode="generative", generative_strength=0.5, upscale_factor=2, photo_finish=0.0)
    events = []

    with patch("app.pipeline.enhancer.release_super_res_engine", side_effect=lambda: events.append("release_sr")), \
         patch("app.pipeline.enhancer.get_controlnet_generator") as mock_get_gen, \
         patch("app.pipeline.enhancer.HardwareGuardrails.release_memory", side_effect=lambda: events.append("release_mem")), \
         patch("app.pipeline.enhancer.upscale_tiled", side_effect=RuntimeError("simulated dml failure")):
        mock_gen = mock_get_gen.return_value
        mock_gen._pipeline_device = "cuda"
        mock_gen.last_run_info = {}

        def fake_gen(*args, **kwargs):
            events.append("generate")
            return img.copy()

        mock_gen.generate.side_effect = fake_gen
        mock_get_gen.side_effect = lambda: (events.append("get_gen"), mock_gen)[1]

        with pytest.raises(RuntimeError, match="simulated dml failure"):
            enhancer.process(img, cfg)

    assert events[0] == "release_sr"
    assert events.index("release_sr") < events.index("get_gen") < events.index("generate")
    # finally block must release both super_res and CUDA memory even after upscale_tiled raises
    assert events[-2:] == ["release_sr", "release_mem"]
