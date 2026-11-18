#!/usr/bin/env python3
"""
Promob Render Studio — Single-Command Launcher
Inicializador unificado para o servidor FastAPI e interface web React.
Verifica dependências, diagnostica aceleração de hardware (GPU/DirectML/CUDA)
e abre automaticamente o navegador.
"""
import sys
import os
import time
import shutil
import subprocess
import argparse
import threading
import webbrowser
from pathlib import Path

# Enforce Python >= 3.10
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    print(f"[ERRO CRÍTICO] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} ou superior é obrigatório.")
    print(f"Versão detectada: {sys.version}")
    sys.exit(1)

# Ensure project root and backend are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Exibe banner profissional no terminal."""
    banner = """
================================================================================
   PROMOB RENDER STUDIO — Aprimorador de Renders Arquitetônicos 4K
   Fidelidade Geométrica 100% | Calibrado para Marcenaria & Arquitetura
================================================================================
"""
    print(banner)


def check_environment() -> dict:
    """Verifica aceleração por GPU, VRAM e estado do frontend estático."""
    diag = {
        "device_name": "CPU Fallback",
        "acceleration": "CPU",
        "free_vram_mb": None,
        "total_vram_mb": None,
        "frontend_built": False,
    }

    # 1. Telemetria de Hardware via HardwareGuardrails
    try:
        from app.core.hardware import HardwareGuardrails
        vram_info = HardwareGuardrails.get_vram_status()
        diag["device_name"] = vram_info.get("device_name", "Desconhecido")
        diag["free_vram_mb"] = vram_info.get("free_mb")
        diag["total_vram_mb"] = vram_info.get("total_mb")
    except Exception as e:
        diag["device_name"] = f"CPU ({e})"

    # 2. Provedores de Execução ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            diag["acceleration"] = "NVIDIA CUDA (ONNX Runtime)"
        elif "DmlExecutionProvider" in providers:
            diag["acceleration"] = "DirectML GPU Hardware Acceleration"
        else:
            diag["acceleration"] = "CPU Multi-threaded Execution"
    except Exception:
        diag["acceleration"] = "CPU Standard"

    # 3. Verificação do build estático do Frontend
    dist_dir = PROJECT_ROOT / "frontend" / "dist"
    index_html = dist_dir / "index.html"
    if dist_dir.is_dir() and index_html.is_file():
        diag["frontend_built"] = True

    return diag


def ensure_frontend_built() -> bool:
    """Verifica se frontend/dist/index.html existe; caso contrário, tenta compilar via npm."""
    frontend_dir = PROJECT_ROOT / "frontend"
    dist_dir = frontend_dir / "dist"
    index_html = dist_dir / "index.html"
    if dist_dir.is_dir() and index_html.is_file():
        return True

    print("[INFO] Build do frontend (frontend/dist/index.html) não encontrado.")
    npm_bin = shutil.which("npm")
    if not npm_bin:
        print(
            "[AVISO] 'npm' não encontrado no PATH. Execute "
            "'cd frontend && npm install && npm run build' para habilitar a interface web."
        )
        return False

    try:
        node_modules = frontend_dir / "node_modules"
        if not node_modules.is_dir():
            print("[INFO] Instalando dependências do frontend (npm --prefix frontend install)...")
            subprocess.run(
                [npm_bin, "--prefix", str(frontend_dir), "install"],
                check=True,
            )

        print("[INFO] Compilando interface web (npm --prefix frontend run build)...")
        subprocess.run(
            [npm_bin, "--prefix", str(frontend_dir), "run", "build"],
            check=True,
        )
    except Exception as exc:
        print(
            f"[AVISO] Falha ao compilar o frontend automaticamente ({exc}). "
            "Execute 'cd frontend && npm install && npm run build' manualmente."
        )
        return False

    return dist_dir.is_dir() and index_html.is_file()


def open_browser_delayed(url: str, delay: float = 1.2):
    """Abre o navegador web após o servidor uvicorn ter iniciado."""
    time.sleep(delay)
    try:
        print(f"\n[INFO] Abrindo navegador em: {url}")
        webbrowser.open(url)
    except Exception as e:
        print(f"[AVISO] Não foi possível abrir o navegador automaticamente: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Promob Render Studio — Servidor Unificado FastAPI + React"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Endereço de escuta do servidor (padrão: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Porta HTTP do servidor (padrão: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Desativa a abertura automática da janela do navegador",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Ativa o modo auto-reload para desenvolvimento",
    )

    args = parser.parse_args()

    print_banner()

    # Diagnóstico de Inicialização
    print(">> Diagnosticando ambiente de execução...")
    ensure_frontend_built()
    diag = check_environment()
    print(f"   * Dispositivo Gráfico: {diag['device_name']}")
    print(f"   * Aceleração Neural:  {diag['acceleration']}")
    if diag['total_vram_mb']:
        print(f"   * Memória VRAM:       {diag['free_vram_mb']:.0f} MB livres / {diag['total_vram_mb']:.0f} MB totais")

    frontend_status = "Pronto (/frontend/dist compilado)" if diag["frontend_built"] else "Pendente (modo API)"
    browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    target_url = f"http://{browser_host}:{args.port}"
    print(f"   * Interface Web:      {frontend_status}")
    print(f"   * URL da Aplicação:   {target_url}")
    print("=" * 80)

    # Dispara abertura de navegador em thread paralela se solicitado
    if not args.no_browser:
        browser_thread = threading.Thread(
            target=open_browser_delayed,
            args=(target_url, 1.2),
            daemon=True,
        )
        browser_thread.start()

    # Inicia o servidor ASGI Uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "backend.app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n[INFO] Servidor finalizado pelo usuário. Até logo!")
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Falha ao iniciar servidor Uvicorn: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
