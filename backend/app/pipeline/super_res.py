"""
Neural Super-Resolution & Tiled Reconstruction Engine for Promob Render Enhancer.
Wraps Real-ESRGAN x4plus ONNX with 2D Hanning cosine taper feathering and 6GB VRAM guardrails.
"""
from typing import Callable, Optional, Dict, Any, Tuple, List
import gc
import os
import sys
import time
import urllib.request
import hashlib
import logging
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Model specifications
MODEL_FILENAME = "realesrgan_x4plus.onnx"
MODEL_SHA256 = "ec90f9f71386729be0d7ffb324ac4c61a55f3539c8e7ff42ab57984952005ce9"
MODEL_EXPECTED_SIZE = 67051650
HF_REPO_ID = "othytenk/realesrgan-x4plus"


def model_download_url() -> str:
    """URL direta do peso no Hugging Face. Respeita HF_ENDPOINT (espelhos e testes de falha)."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{HF_REPO_ID}/resolve/main/{MODEL_FILENAME}"


HF_DIRECT_URL = model_download_url()

DEFAULT_TILE_SIZE = 256
DEFAULT_PADDING = 32
DEFAULT_UPSCALE_FACTOR = 4


def verify_file_sha256(filepath: str, expected_hash: str) -> bool:
    """Verifies SHA-256 hash of a local file in chunks."""
    if not os.path.exists(filepath):
        return False
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest().lower() == expected_hash.lower()
    except Exception as e:
        logger.warning(f"Error computing SHA-256 for {filepath}: {e}")
        return False


def _candidate_model_paths(model_dir: Optional[str] = None) -> List[str]:
    """Ordered list of places where an existing Real-ESRGAN ONNX file is accepted."""
    candidates = []
    if model_dir:
        candidates.append(os.path.join(model_dir, MODEL_FILENAME))

    # Diretório de modelos do aplicativo instalado (fora da pasta do programa)
    env_dir = os.environ.get("PROMOB_MODELS_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, MODEL_FILENAME))

    # Project models/ directory relative to this module
    candidates.append(os.path.join(_project_models_dir(), MODEL_FILENAME))

    # User cache directory
    candidates.append(os.path.join(_user_cache_dir(), MODEL_FILENAME))

    # Hugging Face default cache
    hf_cache_dir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        "models--othytenk--realesrgan-x4plus",
        "snapshots",
    )
    if os.path.exists(hf_cache_dir):
        for root, _, files in os.walk(hf_cache_dir):
            if MODEL_FILENAME in files:
                candidates.append(os.path.join(root, MODEL_FILENAME))
    return candidates


def _project_models_dir() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(module_dir, "..", "..", "..", "models"))


def _user_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "promob_enhancer", "models")


def find_existing_model(model_dir: Optional[str] = None) -> Optional[str]:
    """Returns the path of an already-downloaded model with the expected size, or None."""
    for candidate in _candidate_model_paths(model_dir):
        if os.path.exists(candidate) and os.path.getsize(candidate) == MODEL_EXPECTED_SIZE:
            return candidate
    return None


def default_download_dir() -> str:
    """Where a missing model is downloaded: PROMOB_MODELS_DIR, project models/ or the user cache."""
    env_dir = os.environ.get("PROMOB_MODELS_DIR")
    if env_dir:
        return env_dir
    project_models_dir = _project_models_dir()
    if os.access(os.path.dirname(project_models_dir), os.W_OK):
        return project_models_dir
    return _user_cache_dir()


def download_model(
    target_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    timeout: float = 30.0,
) -> str:
    """
    Downloads the Real-ESRGAN ONNX file to ``target_dir`` and verifies its SHA-256.
    ``progress_callback(bytes_done, bytes_total)`` is called while streaming.
    Raises RuntimeError on network errors or hash mismatch, leaving no partial file behind.
    """
    target_dir = target_dir or default_download_dir()
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, MODEL_FILENAME)
    temp_path = target_path + ".tmp"
    url = model_download_url()

    logger.info(f"Downloading Real-ESRGAN x4plus ONNX model from {url} to {target_path}...")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "promob-render-studio"})
        with urllib.request.urlopen(request, timeout=timeout) as response, open(temp_path, "wb") as out:
            total = int(response.headers.get("Content-Length") or MODEL_EXPECTED_SIZE)
            done = 0
            if progress_callback:
                progress_callback(0, total)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress_callback:
                    progress_callback(done, total)

        if not verify_file_sha256(temp_path, MODEL_SHA256):
            raise RuntimeError("o arquivo baixado não confere com o SHA-256 esperado")

        os.replace(temp_path, target_path)
        logger.info(f"Model downloaded and verified successfully: {target_path}")
        return target_path
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        reason = getattr(e, "reason", None) or e
        raise RuntimeError(
            "Não foi possível baixar o modelo Real-ESRGAN. Verifique a conexão com a internet e "
            f"tente novamente. (Detalhe técnico: {reason})"
        ) from e


def resolve_model_path(
    model_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Locates or automatically downloads the Real-ESRGAN x4plus ONNX model.
    Checks:
      1. Explicit model_dir and PROMOB_MODELS_DIR.
      2. Local project models directory.
      3. User app cache directory.
      4. HuggingFace Hub local cache.
    If not found, downloads automatically from HuggingFace.
    """
    existing = find_existing_model(model_dir)
    if existing:
        logger.info(f"Located existing Real-ESRGAN model at: {existing}")
        return existing

    if progress_callback:
        progress_callback(0.0, "Iniciando download do modelo de super-resolução...")

    def _bytes_progress(done: int, total: int) -> None:
        if progress_callback and total > 0:
            fraction = min(1.0, done / total)
            progress_callback(fraction * 0.1, f"Baixando modelo 4K ({fraction*100:.1f}%)...")

    return download_model(default_download_dir(), progress_callback=_bytes_progress)


def create_hanning_window(h: int, w: int) -> np.ndarray:
    """
    Creates a 2D separable Hanning cosine taper window of shape (h, w),
    floored at 0.02 to avoid severe peripheral attenuation.
    """
    wx = np.hanning(w).astype(np.float32) if w > 4 else np.ones(w, dtype=np.float32)
    wy = np.hanning(h).astype(np.float32) if h > 4 else np.ones(h, dtype=np.float32)
    w2d = wy[:, np.newaxis] * wx[np.newaxis, :]
    return np.maximum(w2d, 0.02)


def generate_hanning_window(th: int, tw: int, scale: int = DEFAULT_UPSCALE_FACTOR) -> np.ndarray:
    """
    Generates a 2D separable Hanning cosine feathering window of shape (th*scale, tw*scale, 1):
      W(y, x) = Wy(y) * Wx(x)
    Floored at 0.02 to avoid severe peripheral attenuation before accumulation.
    """
    w2d = create_hanning_window(th * scale, tw * scale)
    return w2d[:, :, np.newaxis]


def compute_tile_coordinates(
    height: int,
    width: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    padding: int = DEFAULT_PADDING,
) -> List[Tuple[int, int, int, int]]:
    """
    Decomposes an image into overlapping tile coordinates (y0, y1, x0, x1).
    Guarantees complete coverage and bounded step sizes.
    """
    step = tile_size - 2 * padding
    if step <= 0:
        raise ValueError(f"Tile size ({tile_size}) must be > 2 * padding ({2*padding})")

    y_starts = []
    y = 0
    while y < height:
        y_starts.append(y)
        if y + tile_size >= height:
            break
        y += step

    x_starts = []
    x = 0
    while x < width:
        x_starts.append(x)
        if x + tile_size >= width:
            break
        x += step

    tiles = []
    for y_idx in y_starts:
        for x_idx in x_starts:
            y0 = y_idx
            y1 = min(height, y_idx + tile_size)
            x0 = x_idx
            x1 = min(width, x_idx + tile_size)
            tiles.append((y0, y1, x0, x1))

    return tiles


def decompose_tiles(
    h: int,
    w: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_PADDING,
) -> List[Tuple[int, int, int, int]]:
    """
    Decomposes (h, w) into (y, x, th, tw) tiles with overlap.
    """
    step = max(1, tile_size - overlap)
    tiles = []
    y = 0
    while y < h:
        th = min(tile_size, h - y)
        x = 0
        while x < w:
            tw = min(tile_size, w - x)
            tiles.append((y, x, th, tw))
            if x + tw >= w:
                break
            x += step
        if y + th >= h:
            break
        y += step
    return tiles


class SuperResolutionEngine:
    """
    Neural super-resolution engine wrapping Real-ESRGAN x4plus ONNX.
    Manages session providers, memory limits, and seamless 2D tiled upscaling.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        tile_size: int = DEFAULT_TILE_SIZE,
        padding: int = DEFAULT_PADDING,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.tile_size = tile_size
        self.padding = padding
        self.scale = DEFAULT_UPSCALE_FACTOR
        self.model_path = model_path or resolve_model_path(progress_callback=progress_callback)
        self.session, self.device_used = self._init_session(self.model_path, device)

        # Verify input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _init_session(self, model_path: str, requested_device: str) -> Tuple[ort.InferenceSession, str]:
        """
        Initializes ONNX Runtime session with provider fallback hierarchy:
          DirectMLExecutionProvider -> CUDAExecutionProvider -> CPUExecutionProvider
        """
        available = ort.get_available_providers()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = True
        opts.intra_op_num_threads = min(os.cpu_count() or 4, 8)

        providers_to_try = []
        dml_provider = next((p for p in ("DmlExecutionProvider", "DirectMLExecutionProvider") if p in available), None)

        if requested_device == "cuda" and "CUDAExecutionProvider" in available:
            providers_to_try.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 3 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            }))

        if dml_provider and (requested_device in ("directml", "dml", "auto") or (requested_device == "cuda" and "CUDAExecutionProvider" not in available)):
            providers_to_try.append((dml_provider, {"device_id": 0}))

        if requested_device != "cuda" and "CUDAExecutionProvider" in available:
            providers_to_try.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 3 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            }))

        providers_to_try.append(("CPUExecutionProvider", {}))

        for prov_name, prov_opts in providers_to_try:
            try:
                session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=[prov_name],
                    provider_options=[prov_opts] if prov_opts else None,
                )
                logger.info(f"SuperResolutionEngine initialized with provider: {prov_name}")
                return session, prov_name
            except Exception as e:
                logger.warning(f"Failed to initialize provider {prov_name}: {e}. Trying fallback...")

        session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        return session, "CPUExecutionProvider"

    def upscale(
        self,
        input_bgr: np.ndarray,
        tile_size: Optional[int] = None,
        padding: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Performs 4x tiled super-resolution on an input OpenCV BGR image.
        Returns:
          (upscaled_bgr, metadata_dict)
        """
        t0 = time.time()
        tile_size = tile_size or self.tile_size
        padding = padding or self.padding

        h, w = input_bgr.shape[:2]
        out_h, out_w = h * self.scale, w * self.scale

        if input_bgr.ndim == 3 and input_bgr.shape[2] == 3:
            rgb_img = input_bgr[:, :, ::-1].astype(np.float32) / 255.0
        else:
            raise ValueError("Input image must be a 3-channel BGR image.")

        tiles = compute_tile_coordinates(h, w, tile_size, padding)
        total_tiles = len(tiles)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_canvas = np.zeros((out_h, out_w, 1), dtype=np.float32)

        for idx, (y0, y1, x0, x1) in enumerate(tiles):
            if progress_callback:
                progress = idx / total_tiles
                progress_callback(progress, f"Renderizando bloco 4K {idx+1}/{total_tiles}...")

            tile = rgb_img[y0:y1, x0:x1, :]
            th, tw, _ = tile.shape

            tensor_tile = np.ascontiguousarray(
                np.transpose(tile, (2, 0, 1))[np.newaxis, :, :, :],
                dtype=np.float32,
            )

            try:
                out_tensor = self.session.run(None, {self.input_name: tensor_tile})[0]
            except Exception as run_exc:
                if self.device_used != "CPUExecutionProvider":
                    logger.warning(
                        f"Provider {self.device_used} failed at tile {idx+1}/{total_tiles} ({run_exc!r}); "
                        "releasing GPU session and falling back to CPUExecutionProvider."
                    )
                    self.session = None
                    gc.collect()
                    self.session, self.device_used = self._init_session(self.model_path, "cpu")
                    out_tensor = self.session.run(None, {self.input_name: tensor_tile})[0]
                else:
                    raise

            out_tile = np.squeeze(out_tensor, axis=0)
            out_tile = np.transpose(out_tile, (1, 2, 0))
            out_tile = np.clip(out_tile, 0.0, 1.0)

            w_tile = generate_hanning_window(th, tw, self.scale)

            out_y0, out_y1 = y0 * self.scale, y1 * self.scale
            out_x0, out_x1 = x0 * self.scale, x1 * self.scale

            canvas[out_y0:out_y1, out_x0:out_x1, :] += out_tile * w_tile
            weight_canvas[out_y0:out_y1, out_x0:out_x1, :] += w_tile

        epsilon = 1e-7
        normalized = canvas / (weight_canvas + epsilon)
        normalized = np.clip(normalized, 0.0, 1.0)

        out_bgr = (normalized[:, :, ::-1] * 255.0 + 0.5).astype(np.uint8)
        elapsed = time.time() - t0

        if progress_callback:
            progress_callback(1.0, "Super-resolução 4K concluída com sucesso!")

        metadata = {
            "input_width": w,
            "input_height": h,
            "output_width": out_w,
            "output_height": out_h,
            "scale_factor": self.scale,
            "tile_count": total_tiles,
            "tile_size": tile_size,
            "padding": padding,
            "device_used": self.device_used,
            "processing_time_seconds": round(elapsed, 3),
        }

        return out_bgr, metadata


_CACHED_ENGINE: Optional[SuperResolutionEngine] = None


def release_super_res_engine() -> None:
    """Releases cached ONNX DirectML/CUDA SuperResolutionEngine session to free GPU VRAM."""
    global _CACHED_ENGINE
    if _CACHED_ENGINE is not None:
        _CACHED_ENGINE.session = None
        _CACHED_ENGINE = None
        gc.collect()


def get_super_res_engine(
    model_path: Optional[str] = None,
    device: str = "auto",
    tile_size: int = DEFAULT_TILE_SIZE,
    padding: int = DEFAULT_PADDING,
) -> SuperResolutionEngine:
    """Returns a singleton cached SuperResolutionEngine instance matching requested device."""
    global _CACHED_ENGINE
    need_cpu = (device == "cpu")
    need_gpu = (device in ("cuda", "directml", "dml"))
    device_mismatch = False
    if _CACHED_ENGINE is not None:
        is_cpu_cached = (_CACHED_ENGINE.device_used == "CPUExecutionProvider")
        if (need_cpu and not is_cpu_cached) or (need_gpu and is_cpu_cached):
            device_mismatch = True

    if _CACHED_ENGINE is None or (model_path and _CACHED_ENGINE.model_path != model_path) or device_mismatch:
        if _CACHED_ENGINE is not None:
            release_super_res_engine()
        _CACHED_ENGINE = SuperResolutionEngine(
            model_path=model_path,
            device=device,
            tile_size=tile_size,
            padding=padding,
        )
    return _CACHED_ENGINE


def upscale_tiled(
    img: np.ndarray,
    scale: int = DEFAULT_UPSCALE_FACTOR,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_PADDING,
    device: str = "auto",
    model_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Performs tiled upscaling with 2D Hanning feathering.
    If scale == 1: executes an identity feathered canvas reconstruction for testing zero seam artifacts.
    If scale > 1: executes Real-ESRGAN super-resolution.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    h, w = img.shape[:2]

    if scale == 1:
        # Seamless feathered identity reconstruction
        tiles = decompose_tiles(h, w, tile_size=tile_size, overlap=overlap)
        tile_count = len(tiles)

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        weight_canvas = np.zeros((h, w, 1), dtype=np.float32)
        f_img = (img[:, :, :3]).astype(np.float32) / 255.0

        for (y, x, th, tw) in tiles:
            tile = f_img[y:y+th, x:x+tw, :]
            w_tile = create_hanning_window(th, tw)[:, :, np.newaxis]
            canvas[y:y+th, x:x+tw, :] += tile * w_tile
            weight_canvas[y:y+th, x:x+tw, :] += w_tile

        normalized = canvas / (weight_canvas + 1e-7)
        out_bgr = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)
        return out_bgr, tile_count

    # Neural 4x super-resolution
    pad = overlap if overlap > 0 and overlap * 2 < tile_size else DEFAULT_PADDING
    engine = get_super_res_engine(
        model_path=model_path,
        device=device,
        tile_size=tile_size,
        padding=pad,
    )
    out_bgr, meta = engine.upscale(
        img[:, :, :3],
        tile_size=tile_size,
        padding=pad,
        progress_callback=progress_callback,
    )
    return out_bgr, meta["tile_count"]
