"""
Promob Render Studio — inicializador do aplicativo instalado no Windows.

Executado pelo atalho do menu Iniciar com o ``pythonw.exe`` privado (sem janela de
console). Responsabilidades:

* manter uma única instância do servidor por usuário (abrir o app de novo só abre o
  navegador na instância que já está rodando);
* escutar apenas em 127.0.0.1, escolhendo outra porta se a 8000 estiver ocupada;
* abrir o navegador somente depois que ``/api/health`` responder;
* gravar logs e dados do usuário em ``%LOCALAPPDATA%\\PromobRenderStudio``;
* ``--stop`` encerra a instância em execução (atalho "Encerrar" e desinstalador).
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

APP_NAME = "Promob Render Studio"
APP_ID = "promob-render-studio"
HOST = "127.0.0.1"
PREFERRED_PORTS = (8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010)
STARTUP_TIMEOUT_SECONDS = 120.0

log = logging.getLogger("promob_launcher")


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Pasta com backend/ e frontend/dist (``{app}\\app`` no instalado; raiz do repo no código-fonte)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "backend" / "app" / "main.py").is_file():
            return candidate
    raise RuntimeError("Não foi possível localizar a pasta backend/ do Promob Render Studio.")


def install_root(root: Path) -> Path:
    """Pasta de instalação (contém runtime/ com uv.exe) — pai de ``app`` no instalado."""
    return root.parent if (root.parent / "runtime").is_dir() else root


def default_data_dir() -> Path:
    env = os.environ.get("PROMOB_DATA_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "PromobRenderStudio"


# ---------------------------------------------------------------------------
# Instância única
# ---------------------------------------------------------------------------

class SingleInstanceLock:
    """Trava de arquivo mantida durante toda a vida do servidor."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self._fh.close()
        self._fh = None


def state_path(data_dir: Path) -> Path:
    return data_dir / "instancia.json"


def read_state(data_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(state_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_state(data_dir: Path, state: Dict[str, Any]) -> None:
    path = state_path(data_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


def clear_state(data_dir: Path, instance: Optional[str] = None) -> None:
    current = read_state(data_dir)
    if current is None or instance is None or current.get("instance") == instance:
        try:
            state_path(data_dir).unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Rede
# ---------------------------------------------------------------------------

def probe(url: str, instance: Optional[str] = None, timeout: float = 1.5) -> bool:
    """True se ``url`` é um Promob Render Studio respondendo (e, se informado, a mesma instância)."""
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return False
    if data.get("app") != APP_ID:
        return False
    return instance is None or data.get("instance") == instance


def _port_answers(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((HOST, port)) == 0


def bind_listen_socket(ports: Iterable[int] = PREFERRED_PORTS) -> socket.socket:
    """Reserva uma porta livre em 127.0.0.1; usa uma porta do sistema se todas estiverem ocupadas."""
    for port in list(ports) + [0]:
        if port and _port_answers(port):
            log.info(f"Porta {port} ocupada por outro programa; tentando a próxima.")
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind((HOST, port))
        except OSError:
            sock.close()
            log.info(f"Porta {port} indisponível; tentando a próxima.")
            continue
        sock.listen(128)
        return sock
    raise RuntimeError("Nenhuma porta local disponível.")


# ---------------------------------------------------------------------------
# Interface mínima
# ---------------------------------------------------------------------------

def open_browser(url: str) -> None:
    if os.environ.get("PROMOB_NO_BROWSER") == "1":
        log.info(f"Navegador não aberto (PROMOB_NO_BROWSER=1). URL: {url}")
        return
    try:
        webbrowser.open(url)
    except Exception as exc:  # pragma: no cover - depende do sistema
        log.warning(f"Não foi possível abrir o navegador: {exc}")


def message_box(text: str, error: bool = True) -> None:
    log.error(text) if error else log.info(text)
    if os.environ.get("PROMOB_NO_BROWSER") == "1" or sys.platform != "win32":
        return
    try:
        import ctypes

        flags = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
        ctypes.windll.user32.MessageBoxW(None, text, APP_NAME, flags)
    except Exception:
        pass


def setup_logging(data_dir: Path) -> Path:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "servidor.log"
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if sys.stdout is None or sys.stderr is None or "pythonw" in Path(sys.executable).name.lower():
        # pythonw não tem console: tudo que for impresso vai para o log.
        stream = open(log_dir / "console.log", "a", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = stream
        sys.stderr = stream
    else:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root.addHandler(console)
    return log_file


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def configure_environment(data_dir: Path, root: Path, instance: str, token: str) -> None:
    os.environ["PROMOB_DATA_DIR"] = str(data_dir)
    os.environ.setdefault("PROMOB_CACHE_DIR", str(data_dir / "cache"))
    os.environ.setdefault("PROMOB_MODELS_DIR", str(data_dir / "models"))
    os.environ["PROMOB_INSTANCE_ID"] = instance
    os.environ["PROMOB_SHUTDOWN_TOKEN"] = token
    runtime = install_root(root) / "runtime"
    uv = runtime / "uv.exe"
    requirements = runtime / "requirements-ai.txt"
    if uv.is_file() and requirements.is_file():
        os.environ["PROMOB_BUNDLED"] = "1"
        os.environ["PROMOB_UV"] = str(uv)
        os.environ["PROMOB_AI_REQUIREMENTS"] = str(requirements)
    # Evita que pacotes do usuário (pip --user) interfiram no Python privado.
    os.environ["PYTHONNOUSERSITE"] = "1"


def wait_for_running_instance(data_dir: Path, timeout: float = STARTUP_TIMEOUT_SECONDS) -> Optional[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = read_state(data_dir)
        if state and probe(state.get("url", ""), state.get("instance")):
            return state["url"]
        time.sleep(0.5)
    return None


def run(no_browser: bool = False) -> int:
    data_dir = default_data_dir()
    log_file = setup_logging(data_dir)
    if no_browser:
        os.environ["PROMOB_NO_BROWSER"] = "1"

    lock = SingleInstanceLock(data_dir / "instancia.lock")
    if not lock.acquire():
        log.info("Outra instância já está em execução; abrindo o navegador nela.")
        url = wait_for_running_instance(data_dir)
        if url:
            open_browser(url)
            return 0
        message_box(
            "O Promob Render Studio já está sendo iniciado, mas ainda não respondeu. "
            f"Aguarde alguns segundos e abra novamente. Log: {log_file}"
        )
        return 1

    instance = secrets.token_hex(8)
    token = secrets.token_urlsafe(24)
    try:
        root = project_root()
        configure_environment(data_dir, root, instance, token)
        sock = bind_listen_socket()
        port = sock.getsockname()[1]
        url = f"http://{HOST}:{port}"
        log.info(f"Iniciando {APP_NAME} em {url} (dados em {data_dir}).")

        for path in (str(root / "backend"), str(root)):
            if path not in sys.path:
                sys.path.insert(0, path)

        import uvicorn
        from starlette.middleware.trustedhost import TrustedHostMiddleware

        from app.main import app

        # Protege contra DNS rebinding: só aceita requisições endereçadas ao próprio computador.
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
        config = uvicorn.Config(app, log_config=None, access_log=False, timeout_graceful_shutdown=5)
        server = uvicorn.Server(config)
        app.state.request_shutdown = lambda: setattr(server, "should_exit", True)

        def _when_ready() -> None:
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline and not server.should_exit:
                if probe(url, instance):
                    write_state(data_dir, {
                        "pid": os.getpid(),
                        "port": port,
                        "url": url,
                        "instance": instance,
                        "token": token,
                        "started_at": time.time(),
                    })
                    log.info(f"Servidor pronto em {url}.")
                    open_browser(url)
                    return
                time.sleep(0.3)
            if not server.should_exit:
                message_box(f"O servidor não respondeu a tempo. Consulte o log: {log_file}")
                server.should_exit = True

        threading.Thread(target=_when_ready, name="browser_opener", daemon=True).start()
        server.run(sockets=[sock])
        log.info("Servidor encerrado.")
        return 0
    except Exception as exc:
        log.exception("Falha ao iniciar o servidor")
        message_box(f"Não foi possível iniciar o {APP_NAME}.\n\n{exc}\n\nLog: {log_file}")
        return 1
    finally:
        clear_state(data_dir, instance)
        lock.release()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            WAIT_TIMEOUT = 0x102
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(timeout: float = 20.0) -> int:
    """Encerra a instância em execução (usado pelo atalho "Encerrar" e pelo desinstalador)."""
    data_dir = default_data_dir()
    state = read_state(data_dir)
    if not state:
        return 0
    pid = int(state.get("pid") or 0)
    url = state.get("url", "")
    if probe(url, state.get("instance")):
        try:
            req = urllib.request.Request(
                f"{url}/api/app/shutdown",
                method="POST",
                headers={"X-Promob-Token": state.get("token", ""), "Content-Type": "application/json"},
                data=b"{}",
            )
            urllib.request.urlopen(req, timeout=5).read()
        except (OSError, urllib.error.URLError):
            pass
    deadline = time.monotonic() + timeout
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.3)
    if _pid_alive(pid) and sys.platform == "win32":
        # Último recurso: encerra o processo e filhos (ex.: download de bibliotecas em andamento).
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=0x08000000)
    clear_state(data_dir)
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Inicializador do {APP_NAME}")
    parser.add_argument("--stop", action="store_true", help="encerra a instância em execução")
    parser.add_argument("--no-browser", action="store_true", help="não abre o navegador")
    args = parser.parse_args(argv)
    if args.stop:
        return stop()
    return run(no_browser=args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
