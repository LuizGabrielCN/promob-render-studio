"""
Storage and Job Lifecycle Manager for Promob Render Enhancer.
Handles temporary upload buffers, completed high-resolution renders,
job status tracking, and TTL-based disk/memory cache cleanup.
"""
import os
import re
import time
import uuid
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default TTL: 1 hour (3600 seconds)
DEFAULT_CACHE_TTL_SECONDS = 3600.0


@dataclass
class JobInfo:
    """Represents the real-time execution state and metadata of an enhancement job."""
    job_id: str
    status: str = "processing"  # "queued", "processing", "completed", "failed"
    progress: float = 0.0       # 0.0 to 100.0
    stage: str = "Iniciando processamento..."
    width: Optional[int] = None
    height: Optional[int] = None
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    file_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes job state into JSON-compatible dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": round(self.progress, 1),
            "stage": self.stage,
            "width": self.width,
            "height": self.height,
            "output_width": self.output_width or self.width,
            "output_height": self.output_height or self.height,
            "result_url": self.result_url or f"/api/download/{self.job_id}",
            "processing_time": self.processing_time_seconds,
            "processing_time_seconds": self.processing_time_seconds,
            "error": self.error,
        }


class StorageManager:
    """
    Manages in-memory job status metadata and persistent disk cache for enhanced images.
    Thread-safe and guarded against directory traversal attacks.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        if cache_dir is None:
            env_dir = os.environ.get("PROMOB_CACHE_DIR")
            if env_dir:
                self.cache_dir = Path(env_dir).resolve()
            else:
                self.cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
        else:
            self.cache_dir = Path(cache_dir).resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobInfo] = {}
        self._lock = threading.Lock()
        logger.info(f"StorageManager initialized. Cache directory: {self.cache_dir}")

    @staticmethod
    def is_valid_job_id(job_id: str) -> bool:
        """Validates job ID format against path traversal and dangerous characters."""
        if not job_id or not isinstance(job_id, str):
            return False
        if len(job_id) > 128:
            return False
        # Disallow path navigation characters
        if ".." in job_id or "/" in job_id or "\\" in job_id:
            return False
        # Disallow script tags or SQL injection characters
        if "<" in job_id or ">" in job_id or ";" in job_id:
            return False
        return bool(re.match(r"^[a-zA-Z0-9_\-]+$", job_id))

    def create_job(self, job_id: Optional[str] = None) -> JobInfo:
        """Registers a new enhancement job in initial processing state."""
        with self._lock:
            jid = job_id or str(uuid.uuid4())
            job = JobInfo(
                job_id=jid,
                status="processing",
                progress=0.0,
                stage="Iniciando processamento...",
                result_url=f"/api/download/{jid}",
            )
            self._jobs[jid] = job
            return job

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Retrieves job state by ID, returning None if not found or malformed."""
        if not self.is_valid_job_id(job_id):
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        result_url: Optional[str] = None,
        error: Optional[str] = None,
        processing_time_seconds: Optional[float] = None,
        file_path: Optional[str] = None,
    ) -> Optional[JobInfo]:
        """Updates job execution status in a thread-safe manner."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0.0, min(100.0, float(progress)))
            if stage is not None:
                job.stage = stage
            if width is not None:
                job.width = width
                job.output_width = width
            if height is not None:
                job.height = height
                job.output_height = height
            if output_width is not None:
                job.output_width = output_width
                if job.width is None:
                    job.width = output_width
            if output_height is not None:
                job.output_height = output_height
                if job.height is None:
                    job.height = output_height
            if result_url is not None:
                job.result_url = result_url
            if error is not None:
                job.error = error
            if processing_time_seconds is not None:
                job.processing_time_seconds = processing_time_seconds
            if file_path is not None:
                job.file_path = file_path

            job.updated_at = time.time()
            return job

    def save_result(self, job_id: str, image_bgr: np.ndarray) -> str:
        """
        Encodes and writes final 4K enhanced image to cache storage as lossless PNG.
        Guarantees path safety and updates job file reference.
        """
        if not self.is_valid_job_id(job_id):
            raise ValueError(f"Invalid job ID: {job_id}")
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise ValueError("Cannot save empty or invalid image buffer.")

        target_file = (self.cache_dir / f"enhanced_{job_id}.png").resolve()
        # Verify target is strictly within cache_dir
        if not str(target_file).startswith(str(self.cache_dir)):
            raise PermissionError("Path traversal detected.")

        success = cv2.imwrite(str(target_file), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            raise IOError(f"Failed to write image file to {target_file}")

        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.file_path = str(target_file)
                job.updated_at = time.time()

        return str(target_file)

    def get_result_path(self, job_id: str) -> Optional[Path]:
        """
        Returns verified file Path for completed job render, protecting against path traversal.
        """
        if not self.is_valid_job_id(job_id):
            return None

        # Check in-memory record first
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.file_path and os.path.isfile(job.file_path):
                return Path(job.file_path)

        # Fallback to direct cache file existence check
        candidate = (self.cache_dir / f"enhanced_{job_id}.png").resolve()
        try:
            if candidate.is_relative_to(self.cache_dir) and candidate.is_file():
                return candidate
        except (ValueError, AttributeError):
            if str(candidate).startswith(str(self.cache_dir)) and candidate.is_file():
                return candidate

        return None

    def cleanup_old_jobs(
        self,
        ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_age_hours: Optional[float] = None,
    ) -> int:
        """
        Purges expired job records and orphaned cached files older than ttl_seconds (or max_age_hours).
        Returns count of removed items.
        """
        if max_age_hours is not None:
            ttl_seconds = float(max_age_hours) * 3600.0

        now = time.time()
        removed_count = 0

        with self._lock:
            expired_ids = [
                jid for jid, j in self._jobs.items()
                if (now - j.updated_at) > ttl_seconds
            ]
            for jid in expired_ids:
                job = self._jobs.pop(jid, None)
                if job and job.file_path and os.path.isfile(job.file_path):
                    try:
                        os.remove(job.file_path)
                        removed_count += 1
                    except OSError as e:
                        logger.warning(f"Could not remove cached file {job.file_path}: {e}")

        # Scan cache directory for orphaned image files older than TTL
        try:
            for item in self.cache_dir.iterdir():
                if item.is_file() and item.name.startswith("enhanced_"):
                    try:
                        mtime = item.stat().st_mtime
                        if (now - mtime) > ttl_seconds:
                            item.unlink()
                            removed_count += 1
                    except OSError:
                        pass
        except OSError as e:
            logger.warning(f"Error scanning cache dir for cleanup: {e}")

        return removed_count


# Global singleton instance
_storage_instance: Optional[StorageManager] = None
_storage_lock = threading.Lock()


def get_default_storage() -> StorageManager:
    """Returns or creates the global StorageManager singleton."""
    global _storage_instance
    with _storage_lock:
        if _storage_instance is None:
            _storage_instance = StorageManager()
        return _storage_instance
