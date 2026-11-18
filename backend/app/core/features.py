"""
Ativação sob demanda dos recursos pesados: ampliação 4× e Realismo IA.

O modo Clássico funciona só com as dependências do núcleo. Os recursos abaixo são
ativados quando a pessoa escolhe usá-los na interface:

* ``upscale_4x``: baixa o peso Real-ESRGAN (ONNX, ~64 MB) para o diretório de modelos.
* ``realismo_ia``: verifica se há GPU NVIDIA com driver compatível, instala as
  bibliotecas de IA (PyTorch CUDA etc., só no aplicativo instalado no Windows),
  confirma que o PyTorch enxerga a GPU e, só então, baixa os pesos do Hugging Face
  para o cache padrão (o mesmo usado por ``backend/scripts/download_models.py``).

Nada aqui envia imagens para a internet; a rede só é usada para baixar pesos e pacotes.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.core.model_manifest import REQUIRED

logger = logging.getLogger(__name__)

FEATURE_UPSCALE = "upscale_4x"
FEATURE_REALISMO = "realismo_ia"
FEATURES = (FEATURE_UPSCALE, FEATURE_REALISMO)

# Requisitos mínimos para PyTorch 2.6 + CUDA 12.4 no Windows.
MIN_DRIVER_VERSION: Tuple[int, ...] = (527, 41)
MIN_VRAM_MB = 4096
RECOMMENDED_VRAM_MB = 6144

MB = 1024 * 1024
# Estimativas exibidas antes do download (o progresso usa os tamanhos reais quando disponíveis).
UPSCALE_DOWNLOAD_BYTES = 67_051_650
AI_PACKAGES_DOWNLOAD_BYTES = 2_700 * MB
AI_PACKAGES_DISK_BYTES = 5_600 * MB
REALISMO_MODELS_BYTES = 8_000 * MB

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class FeatureUnavailable(RuntimeError):
    """O recurso não pode ser usado neste computador (ex.: sem GPU NVIDIA compatível)."""


@dataclass
class InstallTask:
    feature: str
    state: str = "idle"  # idle | running | failed | done
    progress: float = 0.0  # 0-100
    stage: str = ""
    error: Optional[str] = None
    unavailable: bool = False
    bytes_done: int = 0
    bytes_total: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["progress"] = round(self.progress, 1)
        return data


# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------

def is_bundled() -> bool:
    """True quando rodando a partir do instalador do Windows (Python privado + uv incluído)."""
    return os.environ.get("PROMOB_BUNDLED") == "1"


def data_dir() -> Path:
    env = os.environ.get("PROMOB_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "promob_enhancer"


def _python_for_subprocess() -> str:
    """python.exe ao lado do pythonw.exe (o atalho usa pythonw, sem janela de console)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        candidate = exe.with_name("python.exe")
        if candidate.is_file():
            return str(candidate)
    return str(exe)


def _ai_packages_installed() -> bool:
    importlib.invalidate_caches()
    from app.pipeline.generative import _check_diffusion_deps

    return _check_diffusion_deps()


def _bundled_installer_paths() -> Optional[Tuple[str, str]]:
    uv = os.environ.get("PROMOB_UV")
    req = os.environ.get("PROMOB_AI_REQUIREMENTS")
    if uv and req and Path(uv).is_file() and Path(req).is_file():
        return uv, req
    return None


def _parse_version(text: str) -> Tuple[int, ...]:
    parts = []
    for piece in text.strip().split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def query_nvidia_gpu(timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Consulta o driver NVIDIA via nvidia-smi sem depender do PyTorch."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=timeout,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    parts = [p.strip() for p in output.splitlines()[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        vram_mb = float(parts[2])
    except ValueError:
        vram_mb = 0.0
    return {"name": parts[0], "driver_version": parts[1], "vram_mb": vram_mb}


def cuda_preflight(gpu: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Verifica, antes de qualquer download, se o Realismo IA tem chance de rodar.
    Retorna {"ok": bool, "reason": str|None, "warning": str|None, "gpu": {...}|None}.
    """
    if os.environ.get("PROMOB_SIMULATE_NO_CUDA") == "1":
        gpu = None
    elif gpu is None:
        gpu = query_nvidia_gpu()
    if gpu is None:
        return {
            "ok": False,
            "gpu": None,
            "warning": None,
            "reason": (
                "O Realismo IA precisa de uma placa de vídeo NVIDIA com driver instalado (CUDA). "
                "Não encontramos uma neste computador. O modo Clássico continua disponível e "
                "funciona normalmente no processador."
            ),
        }
    driver = _parse_version(gpu.get("driver_version", ""))
    if driver and driver < MIN_DRIVER_VERSION:
        need = ".".join(str(p) for p in MIN_DRIVER_VERSION)
        return {
            "ok": False,
            "gpu": gpu,
            "warning": None,
            "reason": (
                f"A placa {gpu['name']} usa o driver {gpu['driver_version']}. O Realismo IA "
                f"precisa do driver NVIDIA {need} ou mais recente. Atualize o driver pelo site "
                "da NVIDIA e abra o app de novo. O modo Clássico continua disponível."
            ),
        }
    vram = float(gpu.get("vram_mb") or 0.0)
    if 0 < vram < MIN_VRAM_MB:
        return {
            "ok": False,
            "gpu": gpu,
            "warning": None,
            "reason": (
                f"A placa {gpu['name']} tem {vram / 1024:.1f} GB de memória de vídeo. O Realismo IA "
                f"precisa de pelo menos {MIN_VRAM_MB // 1024} GB. O modo Clássico continua disponível."
            ),
        }
    warning = None
    if 0 < vram < RECOMMENDED_VRAM_MB:
        warning = (
            f"A placa tem {vram / 1024:.1f} GB de memória de vídeo. O Realismo IA foi testado com "
            "6 GB; com menos memória o processamento pode falhar ou ficar lento."
        )
    return {"ok": True, "gpu": gpu, "reason": None, "warning": warning}


def _missing_realismo_files() -> List[Tuple[str, str]]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return [(repo, f) for repo, files in REQUIRED.items() for f in files]
    missing = []
    for repo, files in REQUIRED.items():
        for filename in files:
            if not isinstance(try_to_load_from_cache(repo_id=repo, filename=filename), str):
                missing.append((repo, filename))
    return missing


def _torch_cuda_in_subprocess(timeout: float = 180.0) -> Dict[str, Any]:
    """Importa o PyTorch num processo separado e confirma se ele enxerga a GPU."""
    code = (
        "import json, torch\n"
        "ok = torch.cuda.is_available()\n"
        "print(json.dumps({'cuda': ok, 'torch': torch.__version__, 'cuda_version': torch.version.cuda,"
        " 'device': torch.cuda.get_device_name(0) if ok else None}))\n"
    )
    proc = subprocess.run(
        [_python_for_subprocess(), "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return {"cuda": False, "error": " ".join(tail) or f"código {proc.returncode}"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"cuda": False, "error": proc.stdout.strip()[-300:]}


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _hf_repo_cache_dir(repo_id: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))


def _format_gb(n: int) -> str:
    return f"{n / (1024 ** 3):.1f} GB".replace(".", ",")


def _format_mb(n: int) -> str:
    return f"{n / MB:.0f} MB"


def _ensure_free_space(path: Path, needed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < needed * 1.1:
        raise RuntimeError(
            f"Espaço insuficiente em disco: são necessários cerca de {_format_gb(needed)} e há "
            f"{_format_gb(free)} livres em {path.anchor or path}. Libere espaço e tente novamente."
        )


# ---------------------------------------------------------------------------
# Gerenciador
# ---------------------------------------------------------------------------

class FeatureManager:
    """Mantém o estado das ativações e executa uma por vez em segundo plano."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[str, InstallTask] = {f: InstallTask(feature=f) for f in FEATURES}
        self._thread: Optional[threading.Thread] = None
        self._preflight_cache: Optional[Dict[str, Any]] = None
        self._torch_cuda_cache: Optional[bool] = None

    # -- status ---------------------------------------------------------------
    def _preflight(self, refresh: bool = False) -> Dict[str, Any]:
        if self._preflight_cache is None or refresh:
            self._preflight_cache = cuda_preflight()
        return self._preflight_cache

    def _torch_cuda_ok(self) -> bool:
        if self._torch_cuda_cache is None:
            try:
                import torch

                self._torch_cuda_cache = bool(torch.cuda.is_available())
            except Exception:
                self._torch_cuda_cache = False
        return self._torch_cuda_cache

    def status(self) -> Dict[str, Any]:
        from app.pipeline.super_res import find_existing_model

        with self._lock:
            tasks = {name: task.to_dict() for name, task in self._tasks.items()}
        upscale_ready = find_existing_model() is not None

        preflight = self._preflight()
        packages = _ai_packages_installed()
        missing = _missing_realismo_files()
        can_install_packages = _bundled_installer_paths() is not None
        cuda_ok = packages and self._torch_cuda_ok()
        realismo_ready = packages and cuda_ok and not missing

        reason = None
        if not realismo_ready:
            if not preflight["ok"]:
                reason = preflight["reason"]
            elif packages and not cuda_ok:
                reason = (
                    "As bibliotecas de IA estão instaladas, mas o PyTorch não conseguiu usar a GPU "
                    "(CUDA). Atualize o driver NVIDIA e reinicie o app. O modo Clássico continua disponível."
                )
            elif not packages and not can_install_packages:
                reason = (
                    "As bibliotecas de IA não estão instaladas neste ambiente. Na instalação por "
                    "código-fonte, execute `uv sync --project backend --locked`."
                )

        packages_bytes = 0 if packages else AI_PACKAGES_DOWNLOAD_BYTES
        models_bytes = REALISMO_MODELS_BYTES if missing else 0
        if missing and len(missing) < sum(len(v) for v in REQUIRED.values()):
            # Parte do cache já existe; a estimativa exata aparece durante o download.
            models_bytes = REALISMO_MODELS_BYTES // 2

        return {
            "bundled": is_bundled(),
            "features": {
                FEATURE_UPSCALE: {
                    "ready": upscale_ready,
                    "available": True,
                    "reason": None,
                    "warning": None,
                    "download_bytes": 0 if upscale_ready else UPSCALE_DOWNLOAD_BYTES,
                    "disk_bytes": UPSCALE_DOWNLOAD_BYTES,
                    "summary": "Modelo Real-ESRGAN para ampliar 4× (cerca de 64 MB).",
                    "task": tasks[FEATURE_UPSCALE],
                },
                FEATURE_REALISMO: {
                    "ready": realismo_ready,
                    "available": realismo_ready or (preflight["ok"] and reason is None),
                    "reason": reason,
                    "warning": preflight.get("warning"),
                    "gpu": preflight.get("gpu"),
                    "packages_installed": packages,
                    "models_missing": len(missing),
                    "download_bytes": packages_bytes + models_bytes,
                    "disk_bytes": (0 if packages else AI_PACKAGES_DISK_BYTES) + models_bytes,
                    "summary": (
                        "Bibliotecas de IA (PyTorch com CUDA, cerca de 2,7 GB) e modelos do "
                        "Hugging Face (cerca de 8 GB). Requer GPU NVIDIA com 6 GB de memória."
                    ),
                    "task": tasks[FEATURE_REALISMO],
                },
            },
        }

    # -- execução -------------------------------------------------------------
    def start(self, feature: str) -> Dict[str, Any]:
        if feature not in FEATURES:
            raise KeyError(feature)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                running = next(t for t in self._tasks.values() if t.state == "running")
                if running.feature == feature:
                    return running.to_dict()
                raise RuntimeError("Outro download já está em andamento. Aguarde a conclusão.")
            task = InstallTask(feature=feature, state="running", stage="Preparando...", started_at=time.time())
            self._tasks[feature] = task
            target = self._run_upscale if feature == FEATURE_UPSCALE else self._run_realismo
            self._thread = threading.Thread(target=self._guard, args=(task, target), name=f"activate_{feature}", daemon=True)
            self._thread.start()
            return task.to_dict()

    def wait(self, timeout: Optional[float] = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _update(self, task: InstallTask, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(task, key, value)

    def _guard(self, task: InstallTask, target: Callable[[InstallTask], None]) -> None:
        try:
            target(task)
            self._update(task, state="done", progress=100.0, stage="Pronto para usar.", finished_at=time.time())
            logger.info(f"Feature {task.feature} activated.")
        except FeatureUnavailable as exc:
            logger.warning(f"Feature {task.feature} unavailable: {exc}")
            self._update(task, state="failed", unavailable=True, error=str(exc), stage="Recurso indisponível neste computador.", finished_at=time.time())
        except Exception as exc:
            logger.exception(f"Feature {task.feature} activation failed: {exc}")
            self._update(task, state="failed", error=str(exc), stage="Falha no download.", finished_at=time.time())

    def _run_upscale(self, task: InstallTask) -> None:
        from app.pipeline.super_res import default_download_dir, download_model, find_existing_model

        if find_existing_model():
            return
        target = Path(default_download_dir())
        _ensure_free_space(target, UPSCALE_DOWNLOAD_BYTES)
        self._update(task, stage="Baixando modelo Real-ESRGAN (64 MB)...", bytes_total=UPSCALE_DOWNLOAD_BYTES)

        def _progress(done: int, total: int) -> None:
            pct = 100.0 * done / total if total else 0.0
            self._update(task, bytes_done=done, bytes_total=total, progress=min(pct, 99.0),
                         stage=f"Baixando modelo Real-ESRGAN: {_format_mb(done)} de {_format_mb(total)}")

        download_model(str(target), progress_callback=_progress)

    def _run_realismo(self, task: InstallTask) -> None:
        self._update(task, stage="Verificando a placa de vídeo (CUDA)...")
        preflight = self._preflight(refresh=True)
        if not preflight["ok"]:
            raise FeatureUnavailable(preflight["reason"])

        packages = _ai_packages_installed()
        missing = _missing_realismo_files()
        # Pesos de progresso por etapa, proporcionais ao volume baixado.
        pkg_weight = 0.0 if packages else float(AI_PACKAGES_DOWNLOAD_BYTES)
        model_weight = float(REALISMO_MODELS_BYTES if missing else 0)
        total_weight = max(pkg_weight + model_weight, 1.0)

        if not packages:
            paths = _bundled_installer_paths()
            if paths is None:
                raise FeatureUnavailable(
                    "As bibliotecas de IA não estão instaladas neste ambiente. Na instalação por "
                    "código-fonte, execute `uv sync --project backend --locked`."
                )
            self._install_packages(task, *paths, span=(0.0, 100.0 * pkg_weight / total_weight))
            if not _ai_packages_installed():
                raise RuntimeError("As bibliotecas de IA foram baixadas, mas não puderam ser carregadas. Tente novamente.")

        self._update(task, stage="Confirmando que o PyTorch consegue usar a GPU (CUDA)...")
        check = _torch_cuda_in_subprocess()
        if not check.get("cuda"):
            detail = check.get("error") or f"PyTorch {check.get('torch', '?')} sem acesso à GPU"
            self._torch_cuda_cache = False
            raise FeatureUnavailable(
                "O PyTorch foi instalado, mas não conseguiu usar a GPU NVIDIA (CUDA). Atualize o driver "
                "NVIDIA, reinicie o computador e tente de novo. Os modelos de IA não foram baixados. "
                f"O modo Clássico continua disponível. Detalhe: {detail}"
            )
        self._torch_cuda_cache = None  # reavaliado no próximo status()
        logger.info(f"PyTorch CUDA check passed: {check}")

        if missing:
            start = 100.0 * pkg_weight / total_weight
            self._download_models(task, missing, span=(start, 100.0))

        importlib.invalidate_caches()
        if _missing_realismo_files():
            raise RuntimeError("Alguns modelos não foram encontrados no cache após o download. Tente novamente.")

    def _install_packages(self, task: InstallTask, uv: str, requirements: str, span: Tuple[float, float]) -> None:
        cache_dir = data_dir() / "uv-cache"
        site_root = Path(_python_for_subprocess()).parent
        _ensure_free_space(site_root, AI_PACKAGES_DOWNLOAD_BYTES + AI_PACKAGES_DISK_BYTES)
        log_dir = data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "instalacao-ia.log"
        env = dict(os.environ)
        env.update({
            "UV_CACHE_DIR": str(cache_dir),
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_NO_CONFIG": "1",
            "UV_HTTP_TIMEOUT": "120",
        })
        cmd = [
            uv, "pip", "install",
            "--python", _python_for_subprocess(),
            "--break-system-packages",
            "--requirement", requirements,
            # O arquivo já é o fechamento transitivo travado (uv export, com hashes). Sem --no-deps,
            # o uv resolveria de novo e puxaria o onnxruntime de CPU (metadado do rapidocr), que
            # sobrescreveria o onnxruntime-directml em uso pelo servidor.
            "--no-deps",
            "--require-hashes",
            "--extra-index-url", PYTORCH_INDEX_URL,
            "--index-strategy", "unsafe-best-match",
            "--link-mode", "copy",
            "--no-progress",
        ]
        self._update(task, stage="Baixando bibliotecas de IA (PyTorch com CUDA, cerca de 2,7 GB)...",
                     bytes_total=AI_PACKAGES_DOWNLOAD_BYTES, bytes_done=0, progress=span[0])
        baseline = _dir_size(cache_dir)
        with open(log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=_CREATE_NO_WINDOW)
            while proc.poll() is None:
                time.sleep(1.0)
                done = max(0, _dir_size(cache_dir) - baseline)
                frac = min(done / AI_PACKAGES_DOWNLOAD_BYTES, 0.97)
                stage = (
                    f"Baixando bibliotecas de IA: {_format_gb(done)} de cerca de {_format_gb(AI_PACKAGES_DOWNLOAD_BYTES)}"
                    if frac < 0.97 else "Instalando bibliotecas de IA..."
                )
                self._update(task, bytes_done=min(done, AI_PACKAGES_DOWNLOAD_BYTES),
                             progress=span[0] + (span[1] - span[0]) * frac, stage=stage)
        if proc.returncode != 0:
            tail = ""
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                tail = " ".join(l.strip() for l in lines[-4:])
            except OSError:
                pass
            raise RuntimeError(
                "Não foi possível baixar ou instalar as bibliotecas de IA. Verifique a conexão com a internet "
                f"e tente novamente. Detalhes em {log_path}. {tail[-400:]}"
            )
        # O cache só serve para retomar downloads interrompidos; após o sucesso, libera o espaço.
        shutil.rmtree(cache_dir, ignore_errors=True)
        self._update(task, progress=span[1], bytes_done=AI_PACKAGES_DOWNLOAD_BYTES)

    def _download_models(self, task: InstallTask, missing: List[Tuple[str, str]], span: Tuple[float, float]) -> None:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.constants import HF_HUB_CACHE

        self._update(task, stage="Consultando o tamanho dos modelos no Hugging Face...")
        api = HfApi()
        sizes: Dict[Tuple[str, str], int] = {}
        by_repo: Dict[str, List[str]] = {}
        for repo, filename in missing:
            by_repo.setdefault(repo, []).append(filename)
        for repo, files in by_repo.items():
            for info in api.get_paths_info(repo, files):
                sizes[(repo, info.path)] = int(getattr(info, "size", 0) or 0)
        total = sum(sizes.values()) or REALISMO_MODELS_BYTES
        _ensure_free_space(Path(HF_HUB_CACHE), total)
        self._update(task, bytes_total=total, bytes_done=0)

        done = 0
        for index, (repo, filename) in enumerate(missing, start=1):
            expected = sizes.get((repo, filename), 0)
            repo_dir = _hf_repo_cache_dir(repo)
            before = _dir_size(repo_dir)
            stop = threading.Event()

            def _watch(base: int = done, repo_dir: Path = repo_dir, before: int = before,
                       expected: int = expected, name: str = filename) -> None:
                while not stop.wait(1.0):
                    partial = max(0, _dir_size(repo_dir) - before)
                    if expected:
                        partial = min(partial, expected)
                    current = base + partial
                    self._update(
                        task,
                        bytes_done=current,
                        progress=span[0] + (span[1] - span[0]) * min(current / total, 0.99),
                        stage=f"Baixando modelos de IA ({index}/{len(missing)}: {name}): "
                              f"{_format_gb(current)} de {_format_gb(total)}",
                    )

            watcher = threading.Thread(target=_watch, daemon=True)
            watcher.start()
            try:
                hf_hub_download(repo_id=repo, filename=filename)
            except Exception as exc:
                raise RuntimeError(
                    f"Falha ao baixar {repo}/{filename}. Verifique a conexão com a internet e tente novamente; "
                    f"os arquivos já baixados são aproveitados. Detalhe: {exc}"
                ) from exc
            finally:
                stop.set()
                watcher.join(2.0)
            done += expected
            self._update(task, bytes_done=done,
                         progress=span[0] + (span[1] - span[0]) * min(done / total, 1.0))


_manager: Optional[FeatureManager] = None
_manager_lock = threading.Lock()


def get_feature_manager() -> FeatureManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = FeatureManager()
        return _manager
