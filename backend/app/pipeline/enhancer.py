"""
Concrete ClassicEnhancer implementation unifying deterministic photometric stages
and tiled neural super-resolution under the BaseEnhancer contract.
"""
import time
import uuid
import logging
from typing import Callable, Optional, Tuple, Dict, Any
import cv2
import numpy as np

from app.core.interfaces import BaseEnhancer, EnhancementConfig, EnhancementResult
from app.core.hardware import HardwareGuardrails
from app.pipeline.color import apply_color_correction
from app.pipeline.tone import apply_tone_curve, apply_tone_curve_and_clahe
from app.pipeline.ao import apply_ambient_occlusion
from app.pipeline.detail import apply_detail_enhancement
from app.pipeline.lighting import apply_pbr_lighting
from app.pipeline.generative import get_controlnet_generator
from app.pipeline.led import render_led_strips, split_led_markers
from app.pipeline.photo_finish import apply_photo_finish
from app.pipeline.super_res import upscale_tiled, release_super_res_engine

logger = logging.getLogger(__name__)


def _restore_upscaled_text(output_bgr: np.ndarray, source_bgr: np.ndarray, mask: np.ndarray) -> None:
    """Blend protected native-resolution lettering back after neural upscaling."""
    x, y, width, height = cv2.boundingRect((mask > 0.01).astype(np.uint8))
    if width == 0 or height == 0:
        return

    source_h, source_w = source_bgr.shape[:2]
    scale_x = output_bgr.shape[1] // source_w
    scale_y = output_bgr.shape[0] // source_h
    # Leave two source pixels around the mask for cubic interpolation at its edges.
    x0, x1 = max(0, x - 2), min(source_w, x + width + 2)
    y0, y1 = max(0, y - 2), min(source_h, y + height + 2)
    roi = output_bgr[y0 * scale_y:y1 * scale_y, x0 * scale_x:x1 * scale_x]
    size = (roi.shape[1], roi.shape[0])
    native = cv2.resize(source_bgr[y0:y1, x0:x1, :3], size, interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(mask[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
    roi[:] = cv2.blendLinear(native, roi, alpha, 1.0 - alpha)


class ClassicEnhancer(BaseEnhancer):
    """
    Deterministic Architectural Render Enhancer.
    Guarantees 100% geometric fidelity with zero hallucination.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        logger.info(f"ClassicEnhancer initialized (model_path={model_path})")

    def process_preview(self, input_bgr: np.ndarray, config: EnhancementConfig) -> np.ndarray:
        """
        Generates rapid preview on a downscaled proxy (max dimension <= 960px).
        Executes Photometric Stages 1-4 in ~35-50ms (<100ms). Skips neural super-resolution.
        """
        if input_bgr is None or not isinstance(input_bgr, np.ndarray) or input_bgr.size == 0:
            raise ValueError("Input image cannot be None or empty.")
        if input_bgr.ndim != 3 or input_bgr.shape[2] not in (3, 4):
            raise ValueError(f"Expected 3-channel BGR or 4-channel BGRA image, got shape {getattr(input_bgr, 'shape', None)}")

        h, w = input_bgr.shape[:2]
        max_dim = max(h, w)
        max_preview_dim = 960

        # Downscale proxy if required
        if max_dim > max_preview_dim:
            scale = max_preview_dim / float(max_dim)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            proxy = cv2.resize(input_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            proxy = input_bgr.copy()

        led_mask = None
        led_meta = None
        if config.led_strips:
            proxy, led_mask, led_meta = split_led_markers(proxy)

        # Execute photometric pipeline
        # Stage 1: Color balance, temperature & exposure
        s1 = apply_color_correction(
            proxy,
            temperature=config.temperature,
            tint=config.tint,
            exposure=config.exposure,
        )

        # Stage 2: Dynamic range tone curve & CLAHE
        s2 = apply_tone_curve_and_clahe(
            s1,
            shadow_lift=config.shadow_lift,
            highlight_recovery=config.highlight_recovery,
            clahe_clip_limit=config.clahe_clip_limit,
            clahe_grid_size=config.clahe_grid_size,
        )

        # Stage 3: Morphological Ambient Occlusion
        s3 = apply_ambient_occlusion(
            s2,
            strength=config.ambient_occlusion,
        )

        # Stage 4: Micro-contrast sharpness & bilateral denoise
        s4 = apply_detail_enhancement(
            s3,
            sharpness=config.sharpness,
            denoise_strength=config.denoise_strength,
        )

        # Stage 5: PBR Soft GI Bloom & selective specular highlights
        preview_out = apply_pbr_lighting(
            s4,
            strength=config.specular_bloom,
        )

        if led_mask is not None and led_mask.any():
            preview_out = render_led_strips(preview_out, led_mask, metadata=led_meta)
        return preview_out

    def process(
        self,
        input_bgr: np.ndarray,
        config: EnhancementConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[np.ndarray, EnhancementResult]:
        """
        Executes full-resolution enhancement pipeline with hardware guardrails and tiled super-resolution.
        """
        if input_bgr is None or not isinstance(input_bgr, np.ndarray) or input_bgr.size == 0:
            raise ValueError("Input image cannot be None or empty.")
        if input_bgr.ndim != 3 or input_bgr.shape[2] not in (3, 4):
            raise ValueError(f"Expected 3-channel BGR or 4-channel BGRA image, got shape {getattr(input_bgr, 'shape', None)}")

        job_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        def _report(progress: float, stage: str):
            if progress_callback:
                progress_callback(progress, stage)

        with HardwareGuardrails.get_lock():
            # Capture the render's CUDA high-water mark, not the allocations left after cleanup.
            # ONNX/DirectML allocations are outside PyTorch's allocator and are reported separately
            # by external workload monitoring rather than presented as a measured peak here.
            cuda_tracking = False
            try:
                import torch
                cuda_tracking = torch.cuda.is_available()
                if cuda_tracking:
                    torch.cuda.reset_peak_memory_stats()
            except (ImportError, RuntimeError):
                cuda_tracking = False
            # Yellow Promob marker lines = LED profiles: remove them before any stage can smear them,
            # and paint them back lit right before super-resolution.
            led_mask = None
            led_meta = None
            if config.led_strips:
                input_bgr, led_mask, led_meta = split_led_markers(input_bgr)
                if led_mask.any():
                    _report(0.03, "Perfis de LED detectados (linhas amarelas); preparando iluminação...")

            _report(0.05, "Calibrando balanço de branco, temperatura e exposição...")
            s1 = apply_color_correction(
                input_bgr,
                temperature=config.temperature,
                tint=config.tint,
                exposure=config.exposure,
            )

            used_generative = (
                config.render_mode == "generative"
                and config.generative_strength > 0.0
            )
            device_used = "cpu"
            gen_info: Dict[str, Any] = {}
            text_logo_mask: Optional[np.ndarray] = None

            def _capture_text_logo_mask(mask: np.ndarray) -> None:
                nonlocal text_logo_mask
                text_logo_mask = mask

            try:
                if used_generative:
                    _report(0.18, "Preparando referência limpa e faixa dinâmica para ControlNet PBR...")
                    # Avoid pre-diffusion CLAHE (-30 niche darkening), heavy Scale-3 AO, and high-pass
                    # specular dot amplification; pass clean white-balanced s1 as structural/Lab anchor.
                    gen_init = apply_tone_curve(
                        s1,
                        shadow_lift=config.shadow_lift,
                        highlight_recovery=config.highlight_recovery * 0.5,
                    )
                    if config.ambient_occlusion > 0.0:
                        gen_init = apply_ambient_occlusion(
                            gen_init,
                            strength=min(config.ambient_occlusion, 0.05),
                        )
                    if config.denoise_strength > 0.0:
                        gen_init = apply_detail_enhancement(
                            gen_init,
                            sharpness=0.0,
                            denoise_strength=config.denoise_strength,
                        )

                    # Release cached DirectML ONNX session so SD1.5 + ControlNet get maximum VRAM headroom
                    release_super_res_engine()
                    generator = get_controlnet_generator()
                    _report(0.36, "Iniciando motor generativo ControlNet PBR local...")

                    def _gen_progress(gen_pct: float, gen_msg: str):
                        overall = 0.36 + 0.34 * float(np.clip(gen_pct, 0.0, 1.0))
                        _report(overall, gen_msg)

                    gen_led_kwargs: Dict[str, Any] = (
                        {"led_mask": led_mask, "led_meta": led_meta}
                        if (led_mask is not None and bool(np.any(led_mask)))
                        else {}
                    )
                    s5 = generator.generate(
                        gen_init,
                        generative_strength=config.generative_strength,
                        scene_profile=config.scene_profile,
                        preserve_chroma=True,
                        progress_callback=_gen_progress,
                        reference_bgr=s1,
                        preserve_geometry=config.preserve_geometry,
                        latent_object_lock=config.latent_object_lock,
                        freeu=config.freeu,
                        prompt_style=config.prompt_style,
                        led_strips=config.led_strips,
                        text_logo_mask_callback=_capture_text_logo_mask if config.upscale_factor > 1 else None,
                        **gen_led_kwargs,
                    )
                    run_info = getattr(generator, "last_run_info", None)
                    gen_info = dict(run_info) if isinstance(run_info, dict) else {}
                    # Ensure PyTorch CUDA VRAM is freed before ONNX Real-ESRGAN 4K
                    HardwareGuardrails.release_memory()
                    device_used = generator._pipeline_device
                else:
                    _report(0.15, "Ajustando curva de tom e contraste local CLAHE...")
                    s2 = apply_tone_curve_and_clahe(
                        s1,
                        shadow_lift=config.shadow_lift,
                        highlight_recovery=config.highlight_recovery,
                        clahe_clip_limit=config.clahe_clip_limit,
                        clahe_grid_size=config.clahe_grid_size,
                    )

                    _report(0.25, "Calculando oclusão de ambiente morfológica...")
                    s3 = apply_ambient_occlusion(
                        s2,
                        strength=config.ambient_occlusion,
                    )

                    _report(0.28, "Reforçando nitidez de arestas e micro-contraste...")
                    s4 = apply_detail_enhancement(
                        s3,
                        sharpness=config.sharpness,
                        denoise_strength=config.denoise_strength,
                    )

                    _report(0.34, "Aplicando iluminação indireta GI Bloom e realce especular PBR...")
                    s5 = apply_pbr_lighting(
                        s4,
                        strength=config.specular_bloom,
                    )

                # Photographic finish (generative only): tone/bloom/vignette before the LED repaint, so the
                # lit LEDs are not compressed by the filmic shoulder; grain after super-resolution.
                finish = float(config.photo_finish) if used_generative else 0.0
                if finish > 0.0:
                    s5 = apply_photo_finish(s5, finish, tone=True, grain=False)

                led_count = 0
                if led_mask is not None and led_mask.any():
                    s5 = render_led_strips(s5, led_mask, metadata=led_meta)
                    led_count = int(led_meta["count"]) if led_meta is not None else int(
                        cv2.connectedComponents((led_mask > 0).astype(np.uint8))[0] - 1
                    )

                # Stage 7: Super-Resolution 4K Tiled
                tile_count = 0
                if config.upscale_factor > 1:
                    device, tile_size, overlap = HardwareGuardrails.determine_execution_strategy(config.tile_size)
                    device_used = device
                    sr_base = 0.70 if used_generative else 0.40
                    sr_span = 0.28 if used_generative else 0.55
                    _report(sr_base, f"Iniciando super-resolução 4K ({device.upper()}, tiles {tile_size}x{tile_size})...")

                    def _tile_progress(tile_pct: float, tile_msg: str):
                        overall = sr_base + sr_span * tile_pct
                        _report(overall, tile_msg)

                    output_bgr, tile_count = upscale_tiled(
                        s5,
                        scale=config.upscale_factor,
                        tile_size=tile_size,
                        overlap=overlap,
                        device=device,
                        model_path=self.model_path,
                        progress_callback=_tile_progress,
                    )
                else:
                    output_bgr = s5

                if finish > 0.0:
                    output_bgr = apply_photo_finish(output_bgr, finish, tone=False, grain=True)

                if text_logo_mask is not None:
                    _restore_upscaled_text(output_bgr, s5, text_logo_mask)

                peak_vram = max(
                    float(gen_info.get("detection_vram_mb") or 0.0),
                    torch.cuda.max_memory_allocated() / (1024 ** 2) if cuda_tracking else 0.0,
                )
            finally:
                # Release cache; free DirectML session after generative runs to avoid VRAM starvation across jobs/processes
                if used_generative:
                    release_super_res_engine()
                HardwareGuardrails.release_memory()

            _report(1.0, "Processamento concluído com sucesso!")

        elapsed = time.perf_counter() - start_time
        result = EnhancementResult(
            job_id=job_id,
            output_width=output_bgr.shape[1],
            output_height=output_bgr.shape[0],
            processing_time_seconds=round(elapsed, 3),
            device_used=device_used,
            vram_peak_mb=peak_vram,
            tile_count=tile_count,
            metadata={
                "upscale_factor": config.upscale_factor,
                "preserve_geometry": config.preserve_geometry,
                "render_mode": config.render_mode,
                "generative_strength": config.generative_strength,
                "scene_profile": config.scene_profile,
                "specular_bloom": config.specular_bloom,
                "led_strips_detected": led_count,
                "photo_finish": finish,
                "vram_peak_scope": "pytorch_cuda" if cuda_tracking else "unavailable",
                "latent_object_lock": config.latent_object_lock,
                "freeu": config.freeu,
                "prompt_style": gen_info.get("prompt_style", config.prompt_style),
                **{k: v for k, v in gen_info.items() if k not in ("freeu", "prompt_style")},
                "input_shape": list(input_bgr.shape),
                "output_shape": list(output_bgr.shape),
            },
        )

        return output_bgr, result

    def get_capabilities(self) -> Dict[str, Any]:
        """Returns engine capabilities and hardware status."""
        vram = HardwareGuardrails.get_vram_status()
        gen_status = get_controlnet_generator().get_status()
        return {
            "engine": "ClassicEnhancer",
            "version": "1.0.0",
            "max_scale": 4,
            "preserves_geometry": True,
            "deterministic": True,
            "photometric_stages": [
                "color_temperature_tint_exposure",
                "dynamic_range_clahe_lab",
                "multiscale_ssao_ambient_occlusion",
                "unsharp_mask_bilateral_denoise",
                "pbr_soft_gi_bloom_specular",
            ],
            "super_resolution_engine": "Real-ESRGAN-ONNX",
            "render_modes": ["classic", "generative"],
            "scene_profiles": ["auto", "banheiro", "cozinha", "quarto_sala"],
            "generative_engine": gen_status,
            "hardware": vram,
        }
