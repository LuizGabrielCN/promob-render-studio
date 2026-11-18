"""
FastAPI Server Entry Point for Promob Render Enhancer (ArchRender Studio).
Configures CORS middleware, API route registration, static client mounting,
and lifecycle management for neural models and hardware memory safety.
"""
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.api.routes import router as api_router, get_default_enhancer, shutdown_executor
from app.core.hardware import HardwareGuardrails
from app.core.storage import get_default_storage

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("promob_enhancer.server")


async def _periodic_cleanup_task(storage, interval_seconds: float = 1800.0, max_age_hours: float = 2.0):
    """
    Periodically purges expired job cache files to prevent disk accumulation.
    Runs in the background during application lifecycle.
    """
    logger.info(
        f"Background cache cleanup worker started (interval: {interval_seconds:.0f}s, "
        f"retention: {max_age_hours:.1f}h)."
    )
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                purged = storage.cleanup_old_jobs(max_age_hours=max_age_hours)
                if purged > 0:
                    logger.info(f"Periodic cache cleanup purged {purged} expired job files.")
            except Exception as e:
                logger.warning(f"Error during periodic cache cleanup: {e}")
    except asyncio.CancelledError:
        logger.info("Background cache cleanup worker cancelled cleanly.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes storage directories, inspects GPU/VRAM guardrails, and resolves model weights on startup;
    starts periodic cache cleanup; cleans up GPU caches and thread pools on shutdown.
    """
    logger.info("=" * 60)
    logger.info("Initializing Promob Render Enhancer Backend (Milestone 2 & 4)...")
    logger.info("=" * 60)

    # 1. Initialize persistent storage
    storage = get_default_storage()
    app.state.storage = storage
    logger.info(f"Persistent cache directory initialized: {storage.cache_dir}")

    # 2. Inspect Hardware Guardrails (RTX 2060 6GB target)
    vram_info = HardwareGuardrails.get_vram_status()
    logger.info(
        f"Hardware status: {vram_info.get('device_name')} "
        f"(CUDA Available: {vram_info.get('cuda_available')}, "
        f"Free VRAM: {vram_info.get('free_mb')} MB / {vram_info.get('total_mb')} MB)"
    )

    # 3. Initialize the enhancer. The upscaler resolves its model on demand.
    enhancer = get_default_enhancer()
    app.state.enhancer = enhancer
    logger.info("ClassicEnhancer engine loaded and registered.")

    # 4. Start background periodic cache cleanup worker (every 30 min, 2h retention)
    cleanup_task = asyncio.create_task(
        _periodic_cleanup_task(storage, interval_seconds=1800.0, max_age_hours=2.0)
    )

    yield

    # Shutdown logic
    logger.info("Shutting down Promob Render Enhancer server...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    HardwareGuardrails.release_memory()
    shutdown_executor()
    logger.info("Shutdown completed cleanly.")


def create_app() -> FastAPI:
    """Factory function creating and configuring the FastAPI application instance."""
    application = FastAPI(
        title="Promob Render Enhancer API",
        description="Professional architectural render enhancement engine for Promob (ArchRender Studio)",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS Configuration for local frontend development (Vite @ 5173, etc.)
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API Routes
    application.include_router(api_router, prefix="/api")

    # Static file mounting for production build (Vite dist)
    frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        logger.info(f"Mounting production frontend build from {frontend_dist}")
        if (frontend_dist / "assets").is_dir():
            application.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
        if (frontend_dist / "samples").is_dir():
            application.mount("/samples", StaticFiles(directory=str(frontend_dist / "samples")), name="samples")

    @application.get("/favicon.svg", include_in_schema=False)
    def favicon():
        fav_path = frontend_dist / "favicon.svg"
        if fav_path.is_file():
            return FileResponse(fav_path)
        return Response(status_code=404)

    @application.get("/", include_in_schema=False)
    def index_root(request: Request):
        accept = request.headers.get("accept", "")
        if "text/html" in accept and frontend_dist.is_dir() and (frontend_dist / "index.html").is_file():
            return FileResponse(frontend_dist / "index.html")

        return JSONResponse({
            "name": "Promob Render Enhancer API",
            "version": "0.1.0",
            "status": "online",
            "docs": "/docs",
            "health": "/api/health",
            "frontend": "Run 'npm run dev' inside /frontend or build to /frontend/dist for unified serving.",
        })

    return application


app = create_app()
