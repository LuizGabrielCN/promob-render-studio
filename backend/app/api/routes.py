"""
FastAPI REST Routes for Promob Render Enhancer.
Provides endpoints for hardware monitoring, real-time preview, asynchronous 4K enhancement,
job lifecycle polling, and image asset downloading.
"""
import io
import re
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.core.interfaces import BaseEnhancer, EnhancementConfig
from app.core.hardware import HardwareGuardrails
from app.core.storage import StorageManager, get_default_storage
from app.pipeline.enhancer import ClassicEnhancer

logger = logging.getLogger(__name__)

router = APIRouter()

# Global background thread pool for heavy processing
_executor: Optional[ThreadPoolExecutor] = None
_default_enhancer: Optional[ClassicEnhancer] = None


def get_executor() -> ThreadPoolExecutor:
    """Returns or lazily creates thread pool executor for background tasks."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="promob_worker")
    return _executor


def shutdown_executor() -> None:
    """Release the worker pool so a subsequent app lifespan gets a usable executor."""
    global _executor
    executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=False)


def get_default_enhancer() -> ClassicEnhancer:
    """Returns or lazily creates default ClassicEnhancer instance."""
    global _default_enhancer
    if _default_enhancer is None:
        _default_enhancer = ClassicEnhancer()
    return _default_enhancer


def get_enhancer(request: Request) -> BaseEnhancer:
    """Dependency retrieving active enhancer from app.state or fallback singleton."""
    return getattr(request.app.state, "enhancer", None) or get_default_enhancer()


def get_storage(request: Request) -> StorageManager:
    """Dependency retrieving active storage manager from app.state or fallback singleton."""
    return getattr(request.app.state, "storage", None) or get_default_storage()


class EnhancementFormParams:
    """
    Dependency extracting and validating enhancement parameters from multipart form data.
    Automatically enforces physical bounds and Pydantic schema validation.
    """
    def __init__(
        self,
        temperature: float = Form(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False, description="Temperature shift [-1.0, 1.0]"),
        tint: float = Form(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False, description="Tint shift [-1.0, 1.0]"),
        exposure: float = Form(default=0.0, ge=-2.0, le=2.0, allow_inf_nan=False, description="Exposure EV [-2.0, 2.0]"),
        shadow_lift: float = Form(default=0.20, ge=0.0, le=1.0, allow_inf_nan=False, description="Shadow lift [0.0, 1.0]"),
        highlight_recovery: float = Form(default=0.12, ge=0.0, le=1.0, allow_inf_nan=False, description="Highlight recovery [0.0, 1.0]"),
        clahe_clip_limit: float = Form(default=2.0, ge=0.0, le=5.0, allow_inf_nan=False, description="CLAHE clip limit [0.0, 5.0]"),
        clahe_grid_size: int = Form(default=8, ge=4, le=16, description="CLAHE grid division [4, 16]"),
        ambient_occlusion: float = Form(default=0.15, ge=0.0, le=1.0, allow_inf_nan=False, description="Morphological AO [0.0, 1.0]"),
        sharpness: float = Form(default=0.40, ge=0.0, le=2.0, allow_inf_nan=False, description="Unsharp mask sharpness [0.0, 2.0]"),
        denoise_strength: float = Form(default=0.10, ge=0.0, le=1.0, allow_inf_nan=False, description="Bilateral denoise [0.0, 1.0]"),
        upscale_factor: int = Form(default=4, ge=1, le=4, description="Upscale factor [1, 4]"),
        tile_size: int = Form(default=256, ge=128, le=512, description="Tile dimension [128, 512]"),
        preserve_geometry: bool = Form(default=True, description="Strict geometric lock"),
        render_mode: str = Form(default="classic", pattern="^(classic|generative)$", description="Rendering engine mode ('classic' or 'generative')"),
        generative_strength: float = Form(default=0.48, ge=0.0, le=1.0, allow_inf_nan=False, description="ControlNet generative realism strength [0.0, 1.0]"),
        scene_profile: str = Form(default="auto", pattern="^(auto|banheiro|cozinha|quarto_sala|externa)$", description="Architectural scene profile"),
        specular_bloom: float = Form(default=0.25, ge=0.0, le=1.0, allow_inf_nan=False, description="Soft GI bloom & specular highlight [0.0, 1.0]"),
        led_strips: bool = Form(default=True, description="Yellow marker lines -> lit LED profiles"),
        photo_finish: float = Form(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False, description="Photographic finish (generative) [0.0, 1.0]"),
        latent_object_lock: bool = Form(default=True, description="Freeze objects in latent space during diffusion"),
        freeu: bool = Form(default=False, description="FreeU (experimental A/B)"),
        prompt_style: str = Form(default="legacy", pattern="^(photo|legacy)$", description="Prompt wording (A/B)"),
    ):
        self.config = EnhancementConfig(
            temperature=temperature,
            tint=tint,
            exposure=exposure,
            shadow_lift=shadow_lift,
            highlight_recovery=highlight_recovery,
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid_size=clahe_grid_size,
            ambient_occlusion=ambient_occlusion,
            sharpness=sharpness,
            denoise_strength=denoise_strength,
            upscale_factor=upscale_factor,
            tile_size=tile_size,
            preserve_geometry=preserve_geometry,
            render_mode=render_mode,
            generative_strength=generative_strength,
            scene_profile=scene_profile,
            specular_bloom=specular_bloom,
            led_strips=led_strips,
            photo_finish=photo_finish,
            latent_object_lock=latent_object_lock,
            freeu=freeu,
            prompt_style=prompt_style,
        )


async def get_validated_params(
    request: Request,
    params: EnhancementFormParams = Depends(),
) -> EnhancementFormParams:
    """Rejects explicit empty string values for enum Form fields that Starlette/FastAPI otherwise maps to defaults."""
    form = await request.form()
    for field_name in ("render_mode", "scene_profile", "prompt_style"):
        if field_name in form and form.get(field_name) == "":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Valor vazio não permitido para '{field_name}'.",
            )
    return params


def decode_upload_bytes(contents: bytes, *, for_preview: bool = False) -> np.ndarray:
    """
    Validates and decodes uploaded image bytes into a 3-channel uint8 BGR ndarray.
    Rejects 0-byte, corrupt, unsupported bit depths, and invalid channels with HTTP 400.

    For ``for_preview=True``, skips redundant PIL full-stream decompression on PNGs and
    uses half-resolution JPEG decoding (``IMREAD_REDUCED_COLOR_2``) when dimensions exceed
    1920px (with standard RGB/L layout and default EXIF orientation) since ``process_preview``
    immediately downscales to <= 960px.
    """
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo de upload está vazio (0 bytes).",
        )

    decode_flag = cv2.IMREAD_UNCHANGED
    if for_preview:
        # Inspect JPEG header cheaply (~0.05ms) to enable 1/2 DCT decode on >1920px JPEGs
        # without altering orientation or channels. Skip PIL.verify() full decompression.
        if contents[:3] == b"\xff\xd8\xff":
            try:
                with Image.open(io.BytesIO(contents)) as pil_img:
                    if (
                        pil_img.format == "JPEG"
                        and max(pil_img.size) > 1920
                        and pil_img.mode in ("RGB", "L")
                        and pil_img.getexif().get(274, 1) == 1
                    ):
                        decode_flag = cv2.IMREAD_REDUCED_COLOR_2
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Arquivo inválido, corrompido ou formato não suportado.",
                )
    else:
        # 1. Structural integrity check via PIL for full-resolution /enhance
        try:
            with Image.open(io.BytesIO(contents)) as pil_img:
                pil_img.verify()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Arquivo inválido, corrompido ou formato não suportado.",
            )

    # 2. OpenCV decoding preserving alpha transparency & bit depth (or reduced 1/2 JPEG for preview)
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, decode_flag)

    if img is None or img.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não foi possível decodificar os pixels da imagem.",
        )

    # 3. Normalizar ou validar profundidade de bits antes de qualquer conversão
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Profundidade de bits não suportada: {img.dtype}. Use imagens de 8 ou 16 bits.",
        )

    # 4. Converter canais se necessário (agora garantidamente uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Número inválido de canais: {getattr(img, 'shape', 'desconhecido')}. Esperado 1, 3 ou 4 canais.",
        )

    return img


async def validate_and_decode_upload(file: UploadFile) -> np.ndarray:
    """
    Reads, validates, and decodes an uploaded image file off the event loop.
    Rejects 0-byte, corrupt, non-image, and malformed files with HTTP 400.
    """
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Falha ao ler arquivo de upload: {e}",
        )

    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo de upload está vazio (0 bytes).",
        )

    return await run_in_threadpool(decode_upload_bytes, contents, for_preview=False)


# ============================================================================
# Route 1: Health Check & GPU Telemetry (GET /health)
# ============================================================================
@router.get("/health", summary="Hardware & Service Health Check")
async def health_check() -> Dict[str, Any]:
    """
    Returns service health status and real-time GPU/VRAM telemetry.
    Latency benchmark: < 200ms.
    """
    vram_status = HardwareGuardrails.get_vram_status()
    device_name = vram_status.get("device_name", "CPU")
    free_mb = vram_status.get("free_mb", 0.0)

    return {
        "status": "ok",
        "gpu": device_name,
        "vram_free_mb": max(0.0, float(free_mb)),
        "cuda_available": vram_status.get("cuda_available", False),
        "total_vram_mb": vram_status.get("total_mb", 0.0),
        "allocated_vram_mb": vram_status.get("allocated_mb", 0.0),
    }


# ============================================================================
# Route 2: Fast Interactive Preview (POST /preview)
# ============================================================================
def _process_and_encode_preview(
    contents: bytes,
    config: EnhancementConfig,
    enhancer: BaseEnhancer,
) -> bytes:
    """CPU-bound preview pipeline executed in worker threadpool."""
    img_bgr = decode_upload_bytes(contents, for_preview=True)

    try:
        preview_out = enhancer.process_preview(img_bgr, config)
    except Exception as e:
        logger.exception(f"Erro no processamento de prévia: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no processamento de prévia: {e}",
        )

    # Encode to JPEG quality 92 for fast (<2ms) transmission and low latency
    success, encoded = cv2.imencode(".jpg", preview_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha ao codificar imagem de prévia.",
        )

    return encoded.tobytes()


@router.post("/preview", summary="Generate Low-Latency Preview (<100ms)")
async def generate_preview(
    file: UploadFile = File(..., description="Imagem original do Promob (JPEG/PNG/WEBP)"),
    params: EnhancementFormParams = Depends(get_validated_params),
    enhancer: BaseEnhancer = Depends(get_enhancer),
) -> Response:
    """
    Generates a rapid (<100ms) downscaled preview proxy with photometric adjustments applied.
    Bypasses heavy neural super-resolution for real-time slider responsiveness.
    """
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Falha ao ler arquivo de upload: {e}",
        )

    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo de upload está vazio (0 bytes).",
        )

    jpeg_bytes = await run_in_threadpool(
        _process_and_encode_preview,
        contents,
        params.config,
        enhancer,
    )
    return Response(content=jpeg_bytes, media_type="image/jpeg")


# ============================================================================
# Route 3: Asynchronous 4K Enhancement Submission (POST /enhance)
# ============================================================================
def _run_enhancement_task(
    job_id: str,
    img_bgr: np.ndarray,
    config: EnhancementConfig,
    storage: StorageManager,
    enhancer: BaseEnhancer,
):
    """Worker function executed in background thread pool."""
    try:
        def _progress_cb(pct: float, stage: str):
            storage.update_job(
                job_id,
                progress=round(pct * 100, 1),
                stage=stage,
            )

        storage.update_job(
            job_id,
            status="processing",
            progress=5.0,
            stage="Iniciando processamento fotométrico e restauração...",
        )

        output_bgr, result = enhancer.process(
            img_bgr,
            config=config,
            progress_callback=_progress_cb,
        )

        # Persist final enhanced image to disk cache
        file_path = storage.save_result(job_id, output_bgr)

        storage.update_job(
            job_id,
            status="completed",
            progress=100.0,
            stage="Processamento concluído com sucesso!",
            width=result.output_width,
            height=result.output_height,
            output_width=result.output_width,
            output_height=result.output_height,
            result_url=f"/api/download/{job_id}",
            file_path=file_path,
            processing_time_seconds=result.processing_time_seconds,
        )
        logger.info(f"Job {job_id} successfully completed in {result.processing_time_seconds}s.")

    except Exception as exc:
        logger.exception(f"Job {job_id} failed with exception: {exc}")
        storage.update_job(
            job_id,
            status="failed",
            progress=100.0,
            stage="Erro durante o processamento",
            error=str(exc),
        )


@router.post("/enhance", summary="Submit Asynchronous 4K Enhancement Job", status_code=status.HTTP_200_OK)
async def submit_enhancement(
    file: UploadFile = File(..., description="Imagem original do Promob"),
    params: EnhancementFormParams = Depends(get_validated_params),
    enhancer: BaseEnhancer = Depends(get_enhancer),
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    """
    Submits a full 4K super-resolution enhancement job to background worker queue.
    Returns immediately with job_id and initial 'processing' status.
    """
    # Synchronously validate input file before queuing
    img_bgr = await validate_and_decode_upload(file)

    job_id = str(uuid.uuid4())
    job = storage.create_job(job_id)

    # Dispatch to background thread pool without blocking event loop
    executor = get_executor()
    executor.submit(
        _run_enhancement_task,
        job_id,
        img_bgr,
        params.config,
        storage,
        enhancer,
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
    }


# ============================================================================
# Route 4: Job Polling & Progress (GET /jobs/{job_id})
# ============================================================================
@router.get("/jobs/{job_id}", summary="Get Enhancement Job Status & Progress")
async def get_job_status(
    job_id: str,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    """
    Retrieves the real-time execution status and progress (0-100) of an enhancement job.
    Returns HTTP 404 if the job ID does not exist, or 400 if malformed.
    """
    if not StorageManager.is_valid_job_id(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identificador de job inválido ou malformado.",
        )

    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job com ID '{job_id}' não foi encontrado.",
        )

    return job.to_dict()


# ============================================================================
# Route 5: Download Enhanced 4K Asset (GET /download/{job_id})
# ============================================================================
@router.get("/download/{job_id}", summary="Download Enhanced 4K Image Asset")
async def download_enhanced_image(
    job_id: str,
    storage: StorageManager = Depends(get_storage),
) -> FileResponse:
    """
    Streams the finished 4K enhanced image as a PNG download with Content-Disposition headers.
    Returns HTTP 404 if the job or file does not exist, or HTTP 400 if not completed.
    """
    if not StorageManager.is_valid_job_id(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Identificador de job inválido.",
        )

    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job com ID '{job_id}' não foi encontrado.",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"O processamento do job ainda está com status '{job.status}'. Aguarde a conclusão.",
        )

    file_path = storage.get_result_path(job_id)
    if not file_path or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Arquivo final aprimorado não encontrado no cache de armazenamento.",
        )

    filename = f"promob_enhanced_{job_id}.png"
    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


# ============================================================================
# Route 6: Presets Configuration (GET /presets)
# ============================================================================
@router.get("/presets", summary="List Architectural Presets")
async def list_presets() -> Dict[str, Any]:
    """Returns curated architectural enhancement presets for the UI."""
    return {
        "realismo_ia": {
            "name": "Realismo IA (Estilo ChatGPT)",
            "description": "Re-renderização PBR local com ControlNet: materiais e iluminação realistas mantendo as cores e os objetos do projeto; linhas amarelas viram perfis de LED acesos",
            "config": {
                "render_mode": "generative",
                "generative_strength": 0.48,
                "scene_profile": "auto",
                "specular_bloom": 0.25,
                "photo_finish": 0.5,
                "temperature": 0.04,
                "tint": 0.0,
                "exposure": 0.05,
                "shadow_lift": 0.23,
                "highlight_recovery": 0.10,
                "clahe_clip_limit": 2.0,
                "clahe_grid_size": 8,
                "ambient_occlusion": 0.15,
                "sharpness": 0.40,
                "denoise_strength": 0.10,
                "upscale_factor": 4,
                "tile_size": 256,
                "preserve_geometry": True,
            },
        },
        "natural": {
            "name": "Natural",
            "description": "Calibração arquitetônica equilibrada e tons neutros",
            "config": {
                "render_mode": "classic",
                "specular_bloom": 0.18,
                "temperature": 0.15,
                "shadow_lift": 0.20,
                "highlight_recovery": 0.10,
                "clahe_clip_limit": 1.5,
                "sharpness": 0.30,
                "ambient_occlusion": 0.15,
                "denoise_strength": 0.10,
            }
        },
        "marcenaria": {
            "name": "Marcenaria",
            "description": "Realce de veios de madeira, texturas de MDF e profundidade de frestas",
            "config": {
                "render_mode": "classic",
                "specular_bloom": 0.25,
                "temperature": 0.20,
                "shadow_lift": 0.25,
                "highlight_recovery": 0.10,
                "clahe_clip_limit": 2.5,
                "sharpness": 0.60,
                "ambient_occlusion": 0.25,
                "denoise_strength": 0.08,
            }
        },
        "luz_quente": {
            "name": "Luz Quente",
            "description": "Simulação de iluminação interior acolhedora a 3000K",
            "config": {
                "render_mode": "classic",
                "specular_bloom": 0.30,
                "temperature": 0.50,
                "tint": 0.10,
                "exposure": 0.10,
                "shadow_lift": 0.30,
                "highlight_recovery": 0.15,
                "clahe_clip_limit": 1.8,
                "sharpness": 0.35,
                "ambient_occlusion": 0.15,
            }
        },
        "ultra_detalhe": {
            "name": "Ultra Detalhe",
            "description": "Nitidez máxima e micro-contraste adaptativo para grandes ampliações",
            "config": {
                "render_mode": "classic",
                "specular_bloom": 0.20,
                "temperature": 0.15,
                "shadow_lift": 0.22,
                "highlight_recovery": 0.12,
                "clahe_clip_limit": 3.0,
                "sharpness": 0.80,
                "ambient_occlusion": 0.20,
                "denoise_strength": 0.15,
            }
        }
    }


# ============================================================================
# Route 7: Engine Capabilities (GET /capabilities)
# ============================================================================
@router.get("/capabilities", summary="Get Engine Capabilities")
def get_engine_capabilities(
    enhancer: BaseEnhancer = Depends(get_enhancer),
) -> Dict[str, Any]:
    """Returns engine capabilities, photometric pipeline stages, and hardware metadata."""
    return enhancer.get_capabilities()
