"""
Hardware Guardrails and Memory Safety Manager.
Enforces strict 6GB VRAM budget for NVIDIA GeForce RTX 2060 with dynamic tile scaling
and graceful fallback to CPU.
"""
import gc
import logging
import subprocess
import threading
import time
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class HardwareGuardrails:
    """
    Monitors GPU VRAM availability and determines safe tile sizes to prevent CUDA OOM errors.
    """
    VRAM_HIGH_THRESHOLD_MB = 3000.0    # >= 3000MB free -> 512x512 tiles (overlap 32px)
    VRAM_MEDIUM_THRESHOLD_MB = 1500.0  # 1500-3000MB free -> 256x256 tiles (overlap 16px)
    VRAM_CEILING_LIMIT_MB = 5500.0     # Hard ceiling for 6GB GPU budget

    _lock = threading.Lock()
    _vram_cache_lock = threading.Lock()
    _vram_cache: Optional[Dict[str, Any]] = None
    _vram_cache_ts: float = 0.0
    _vram_refreshing: bool = False
    _vram_cache_generation: int = 0

    @classmethod
    def _query_vram_status(cls) -> Dict[str, Any]:
        """
        Introspects GPU VRAM availability synchronously.
        Tries PyTorch CUDA first, falls back to nvidia-smi CSV query, then CPU fallback.
        """
        # Strategy 1: PyTorch CUDA if available
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                allocated_bytes = torch.cuda.memory_allocated(0)
                reserved_bytes = torch.cuda.memory_reserved(0)
                return {
                    "cuda_available": True,
                    "device_name": torch.cuda.get_device_name(0),
                    "free_mb": round(free_bytes / (1024 ** 2), 1),
                    "total_mb": round(total_bytes / (1024 ** 2), 1),
                    "allocated_mb": round(allocated_bytes / (1024 ** 2), 1),
                    "reserved_mb": round(reserved_bytes / (1024 ** 2), 1),
                    "source": "torch_cuda",
                }
        except (ImportError, Exception) as e:
            logger.debug(f"PyTorch CUDA query unavailable: {e}")

        # Strategy 2: Direct nvidia-smi query
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used,name", "--format=csv,noheader,nounits"]
            output = subprocess.check_output(cmd, text=True, timeout=2.0).strip()
            if output:
                first_line = output.splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                total_mb = float(parts[0])
                free_mb = float(parts[1])
                used_mb = float(parts[2])
                name = parts[3]
                return {
                    "cuda_available": True,
                    "device_name": name,
                    "free_mb": free_mb,
                    "total_mb": total_mb,
                    "allocated_mb": used_mb,
                    "reserved_mb": used_mb,
                    "source": "nvidia-smi",
                }
        except (subprocess.SubprocessError, FileNotFoundError, Exception) as e:
            logger.debug(f"nvidia-smi query unavailable: {e}")

        # Strategy 3: Graceful CPU Fallback
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "free_mb": 0.0,
            "total_mb": 0.0,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "source": "cpu_fallback",
        }

    @classmethod
    def _refresh_vram_cache_async(cls) -> None:
        with cls._vram_cache_lock:
            start_gen = cls._vram_cache_generation
        try:
            fresh = cls._query_vram_status()
            with cls._vram_cache_lock:
                if cls._vram_cache_generation == start_gen:
                    cls._vram_cache = fresh
                    cls._vram_cache_ts = time.monotonic()
        finally:
            with cls._vram_cache_lock:
                cls._vram_refreshing = False

    @classmethod
    def get_vram_status(cls, max_age_seconds: float = 2.0) -> Dict[str, Any]:
        """
        Returns GPU VRAM availability with non-blocking stale-while-revalidate caching
        once initialized so health checks remain sub-10ms on Windows/nvidia-smi.
        """
        if max_age_seconds > 0:
            with cls._vram_cache_lock:
                if cls._vram_cache is not None:
                    age = time.monotonic() - cls._vram_cache_ts
                    if age > max_age_seconds and not cls._vram_refreshing:
                        cls._vram_refreshing = True
                        try:
                            threading.Thread(
                                target=cls._refresh_vram_cache_async,
                                name="vram_refresh",
                                daemon=True,
                            ).start()
                        except Exception as e:
                            cls._vram_refreshing = False
                            logger.debug(f"Failed to start VRAM refresh thread: {e}")
                    return dict(cls._vram_cache)

        with cls._vram_cache_lock:
            start_gen = cls._vram_cache_generation
        fresh = cls._query_vram_status()
        with cls._vram_cache_lock:
            if cls._vram_cache_generation == start_gen:
                cls._vram_cache = fresh
                cls._vram_cache_ts = time.monotonic()
        return dict(fresh)

    @classmethod
    def determine_execution_strategy(cls, requested_tile_size: int = 512) -> Tuple[str, int, int]:
        """
        Determines the safe execution device, tile dimension, and overlap margin based on real-time VRAM.

        Returns:
            Tuple[device: str, tile_size: int, tile_overlap: int]
            - device: "cuda" or "cpu"
            - tile_size: 512 or 256
            - tile_overlap: 32 or 16
        """
        status = cls.get_vram_status()
        if not status["cuda_available"]:
            logger.info("GPU unavailable; falling back to CPU execution.")
            return "cpu", 256, 16

        free_mb = status["free_mb"]
        if free_mb >= cls.VRAM_HIGH_THRESHOLD_MB:
            effective_tile = min(requested_tile_size, 512)
            logger.info(f"High VRAM headroom ({free_mb} MB). Using GPU with tile size {effective_tile}x{effective_tile}.")
            return "cuda", effective_tile, 32
        elif free_mb >= cls.VRAM_MEDIUM_THRESHOLD_MB:
            logger.warning(f"Moderate VRAM headroom ({free_mb} MB). Throttling tile size to 256x256.")
            return "cuda", 256, 16
        else:
            logger.warning(f"Low VRAM headroom ({free_mb} MB < {cls.VRAM_MEDIUM_THRESHOLD_MB} MB). Falling back to CPU.")
            return "cpu", 256, 16

    @classmethod
    def release_memory(cls):
        """
        Force garbage collection, clear PyTorch CUDA cached memory allocations,
        and invalidate the cached VRAM reading so post-release callers see fresh headroom.
        Safe to call even if PyTorch or CUDA is not installed.
        """
        try:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass
        finally:
            with cls._vram_cache_lock:
                cls._vram_cache = None
                cls._vram_cache_ts = 0.0
                cls._vram_cache_generation += 1

    @classmethod
    def get_lock(cls) -> threading.Lock:
        """Returns the concurrency lock to serialize heavy GPU jobs."""
        return cls._lock
