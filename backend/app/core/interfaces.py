"""
Core Interfaces and Data Contracts for Promob Render Enhancer.
Defines abstract contracts for enhancement engines and Pydantic validation schemas.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class EnhancementConfig(BaseModel):
    """
    Configuration parameters for architectural render enhancement.
    All parameters are bounded and include physical default calibrations.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 1. Color Balance & Temperature
    temperature: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
        description="Color temperature shift. -1.0 (Cold/Blue) to +1.0 (Warm/Amber)."
    )
    tint: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
        description="Chromatic tint shift. -1.0 (Green compensation) to +1.0 (Magenta warm wood)."
    )
    exposure: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        allow_inf_nan=False,
        description="Global exposure compensation in EV stops (-2.0 to +2.0)."
    )

    # 2. Tone Curve & Dynamic Range
    shadow_lift: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Shadow illumination recovery strength (0.0 to 1.0, default 0.20 for +23% lift)."
    )
    highlight_recovery: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Highlight compression roll-off (0.0 to 1.0, default 0.12 for -6% glare dampening)."
    )
    clahe_clip_limit: float = Field(
        default=2.0,
        ge=0.0,
        le=5.0,
        allow_inf_nan=False,
        description="Local contrast CLAHE clip limit strictly in CIE L* channel (0.0 = disabled, 2.0 = standard)."
    )
    clahe_grid_size: int = Field(
        default=8,
        ge=4,
        le=16,
        description="Grid division size for CLAHE adaptive histogram tiles."
    )

    # 3. Ambient Occlusion & Micro-Depth
    ambient_occlusion: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Morphological ambient occlusion strength for contact crevices and seams."
    )

    # 4. Detail, Nitidez & Denoise
    sharpness: float = Field(
        default=0.40,
        ge=0.0,
        le=2.0,
        allow_inf_nan=False,
        description="Unsharp mask edge micro-contrast multiplier (0.0 = disabled, 0.40 = 3.6x sharpness boost)."
    )
    denoise_strength: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Edge-preserving bilateral denoise strength for low-quality JPEG artifacts."
    )

    # 5. Super-Resolution & Tiling
    upscale_factor: int = Field(
        default=4,
        ge=1,
        le=4,
        description="Upscale factor (1 = native resolution, 4 = 4K neural super-resolution)."
    )
    tile_size: int = Field(
        default=256,
        ge=128,
        le=512,
        description="Base tile dimension for VRAM safety."
    )
    preserve_geometry: bool = Field(
        default=True,
        description="Strict lock guaranteeing 100% geometric fidelity without generative drift."
    )

    # 6. Photometric PBR Lighting & Local Generative Engine
    render_mode: str = Field(
        default="classic",
        pattern="^(classic|generative)$",
        description="Rendering engine mode: 'classic' (deterministic PBR) or 'generative' (local ControlNet + PBR)."
    )
    generative_strength: float = Field(
        default=0.48,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="ControlNet generative realism blend / denoising strength (0.0 to 1.0, default 0.48)."
    )
    scene_profile: str = Field(
        default="auto",
        pattern="^(auto|banheiro|cozinha|quarto_sala|externa)$",
        description="Architectural scene prompt profile ('auto', 'banheiro', 'cozinha', 'quarto_sala', 'externa')."
    )
    specular_bloom: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Soft Global Illumination bloom and selective PBR specular highlight strength (0.0 to 1.0)."
    )
    led_strips: bool = Field(
        default=True,
        description="Turn thin saturated-yellow Promob marker lines into lit LED profiles (warm core + glow)."
    )
    photo_finish: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Photographic finish in the generative mode: filmic shoulder, soft bloom, vignette and "
                    "luminance grain (grain applied after super-resolution). 0.0 = off."
    )
    latent_object_lock: bool = Field(
        default=True,
        description="Freeze loose objects in latent space during diffusion (soft-inpainting per step)."
    )
    freeu: bool = Field(
        default=False,
        description="FreeU on the SD 1.5 UNet (experimental; off until it wins the A/B on the samples)."
    )
    prompt_style: str = Field(
        default="legacy",
        pattern="^(photo|legacy)$",
        description="Prompt wording: 'legacy' (default; won the v3 A/B on colour fidelity) or 'photo' (photographic wording, A/B)."
    )


class EnhancementResult(BaseModel):
    """
    Standard execution result containing metadata and performance metrics.
    """
    job_id: str = Field(description="Unique UUID identifier for the enhancement run.")
    output_width: int = Field(description="Final output width in pixels.")
    output_height: int = Field(description="Final output height in pixels.")
    processing_time_seconds: float = Field(description="Total execution latency in seconds.")
    device_used: str = Field(description="Hardware execution backend (e.g. 'cuda:0', 'directml', 'cpu').")
    vram_peak_mb: float = Field(description="Peak VRAM allocated during inference in MB.")
    tile_count: int = Field(default=0, description="Number of tiles processed for super-resolution.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary execution metrics.")


class BaseEnhancer(ABC):
    """
    Abstract Base Class defining the contract for architectural render enhancers.
    Decouples deterministic pipelines from potential future diffusion models.
    """

    @abstractmethod
    def process_preview(self, input_bgr: np.ndarray, config: EnhancementConfig) -> np.ndarray:
        """
        Fast synchronous execution on downscaled proxy (<1080px) for real-time UI preview (<100ms).
        Bypasses neural super-resolution.

        Args:
            input_bgr: Source image in BGR uint8 format (H, W, 3).
            config: Parametric enhancement configuration.

        Returns:
            Enhanced preview image in BGR uint8 format.
        """
        pass

    @abstractmethod
    def process(
        self,
        input_bgr: np.ndarray,
        config: EnhancementConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[np.ndarray, EnhancementResult]:
        """
        Full high-resolution enhancement pipeline with tiled super-resolution.

        Args:
            input_bgr: Source image in BGR uint8 format (H, W, 3).
            config: Parametric enhancement configuration.
            progress_callback: Optional callback reporting (progress_fraction: float 0.0-1.0, stage_name: str).

        Returns:
            Tuple of (output_bgr: np.ndarray, result: EnhancementResult).
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Returns model metadata, supported resolutions, hardware status, and backend features.
        """
        pass
