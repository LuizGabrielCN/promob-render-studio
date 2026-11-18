"""
Teste visual da primeira experiência, dirigido pela interface real (Chrome/Edge headless via CDP).

Percorre o fluxo que uma pessoa faz após instalar o app e salva capturas de tela:

  1. tela inicial com o botão "Experimentar com uma amostra";
  2. clique no botão -> modo Clássico, resolução original, rota real /api/enhance;
  3. comparação antes/depois concluída;
  4. download do PNG (o arquivo é verificado: assinatura PNG e mesmas dimensões da amostra);
  5. cartão de ativação da ampliação 4× (tamanho estimado, sem baixar nada);
  6. cartão do Realismo IA (ativação ou explicação de indisponibilidade).

Uso (com o app já aberto):
    python scripts/ui_walkthrough.py --url http://127.0.0.1:8000 --out docs/screenshots

Depende apenas da biblioteca padrão e de ``websockets`` (incluído no app instalado).
Sai com código 1 se alguma etapa falhar.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.sync.client import connect

BROWSERS = [
    os.environ.get("BROWSER_PATH", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


class _RedirectToHub(BaseHTTPRequestHandler):
    """Encaminha para o Hugging Face real: simula a conexão voltando depois de uma falha."""

    def do_GET(self):  # noqa: N802
        self.send_response(302)
        self.send_header("Location", "https://huggingface.co" + self.path)
        self.end_headers()

    do_HEAD = do_GET

    def log_message(self, *args):
        pass


class Page:
    def __init__(self, ws):
        self.ws = ws
        self._id = 0

    def call(self, method: str, **params):
        self._id += 1
        msg_id = self._id
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=60))
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expression: str):
        res = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if res.get("exceptionDetails"):
            raise RuntimeError(f"JS falhou: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float, what: str):
        deadline = time.time() + timeout
        while time.time() < deadline:
            value = self.js(expression)
            if value:
                return value
            time.sleep(0.25)
        raise TimeoutError(f"Tempo esgotado esperando: {what}")

    def click_text(self, text: str, tag: str = "button"):
        ok = self.js(
            f"""(() => {{
                const el = [...document.querySelectorAll({json.dumps(tag)})]
                  .find(e => e.innerText.includes({json.dumps(text)}) && !e.disabled);
                if (!el) return false;
                el.scrollIntoView({{block: 'center'}});
                el.click();
                return true;
            }})()"""
        )
        if not ok:
            raise RuntimeError(f"Botão não encontrado: {text}")

    def screenshot(self, path: Path):
        data = self.call("Page.captureScreenshot", format="png")["data"]
        path.write_bytes(base64.b64decode(data))
        print(f"  captura: {path}")


def png_size(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "arquivo baixado não é PNG"
    return struct.unpack(">II", data[16:24])


def jpeg_size(data: bytes):
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + length
    raise ValueError("JPEG sem cabeçalho SOF")


def run(url: str, out: Path, browser: str, realismo_shot: str = "06-realismo-ia.png",
        failure_port: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="promob_ui_")
    downloads = Path(tempfile.mkdtemp(prefix="promob_dl_"))
    proc = subprocess.Popen(
        [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
         "--remote-debugging-port=0", f"--user-data-dir={profile}", "--window-size=1440,900",
         "--force-device-scale-factor=1", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        port_file = Path(profile) / "DevToolsActivePort"
        deadline = time.time() + 30
        while not port_file.is_file() and time.time() < deadline:
            time.sleep(0.1)
        port = port_file.read_text().splitlines()[0].strip()
        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5).read())
        page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page")

        with connect(page_ws, max_size=64 * 1024 * 1024) as ws:
            page = Page(ws)
            page.call("Page.enable")
            page.call("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1, mobile=False)
            page.call("Browser.setDownloadBehavior", behavior="allow", downloadPath=str(downloads))

            print("1. Tela inicial")
            page.call("Page.navigate", url=url)
            page.wait_for("document.body && document.body.innerText.includes('Experimentar com uma amostra')", 60, "tela inicial")
            time.sleep(1.0)
            page.screenshot(out / "01-tela-inicial.png")

            print("2. Experimentar com uma amostra (Clássico, 1×, CPU)")
            page.click_text("Experimentar com uma amostra")
            page.wait_for("document.body.innerText.includes('Baixar PNG')", 180, "resultado da amostra")
            time.sleep(1.5)
            summary = page.js("document.body.innerText")
            assert "Concluído" in summary, "status 'Concluído' não apareceu"
            page.screenshot(out / "02-amostra-comparacao.png")

            print("3. Comparação: arrastando o divisor")
            box = page.js(
                "(() => { const r = document.querySelector('.cursor-ew-resize').getBoundingClientRect();"
                " return [r.left, r.top, r.width, r.height]; })()"
            )
            x, y = box[0] + box[2] * 0.30, box[1] + box[3] * 0.5
            for event in ("mousePressed", "mouseReleased"):
                page.call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
            time.sleep(0.5)
            page.screenshot(out / "03-comparacao-divisor.png")

            print("4. Baixar PNG")
            page.click_text("Baixar PNG")
            deadline = time.time() + 60
            pngs = []
            while time.time() < deadline:
                pngs = [p for p in downloads.glob("*.png")]
                if pngs and not list(downloads.glob("*.crdownload")):
                    break
                time.sleep(0.25)
            assert pngs, "o PNG não foi baixado"
            sample = urllib.request.urlopen(f"{url}/samples/promob_sample.jpg", timeout=10).read()
            got, want = png_size(pngs[0].read_bytes()), jpeg_size(sample)
            assert got == want, f"PNG {got} difere da amostra {want} (esperado: resolução original)"
            print(f"  PNG baixado: {pngs[0].name} {got[0]}×{got[1]} (igual à amostra)")
            shutil.copy(pngs[0], out / "04-resultado-amostra.png")

            print("5. Ampliação 4× (cartão de ativação)")
            page.click_text("Ampliar 4×")
            page.wait_for(
                "!!document.querySelector('[data-testid=feature-upscale_4x]') || "
                "document.body.innerText.includes('Aprimorar e ampliar 4×')", 15, "cartão 4×")
            time.sleep(0.5)
            page.screenshot(out / "05-ativar-4x.png")

            print("6. Realismo IA (ativação ou explicação)")
            page.click_text("requer GPU NVIDIA")
            page.wait_for(
                "!!document.querySelector('[data-testid=feature-realismo_ia]') || "
                "document.body.innerText.includes('Fidelidade Promob')", 30, "cartão Realismo IA")
            time.sleep(0.5)
            page.js("(() => { const el = document.querySelector('[data-testid=feature-realismo_ia]');"
                    " if (el) el.scrollIntoView({block: 'center'}); return true; })()")
            time.sleep(0.3)
            page.screenshot(out / realismo_shot)
            if failure_port:
                failure_and_retry(page, url, out, downloads, failure_port)
            print("OK: fluxo visual concluído.")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(downloads, ignore_errors=True)


def failure_and_retry(page: Page, url: str, out: Path, downloads: Path, port: int) -> None:
    """O app foi iniciado com HF_ENDPOINT=http://127.0.0.1:<port>, porta ainda fechada."""
    print("7. Falha no download do modelo 4× (sem conexão)")
    page.click_text("Ampliar 4×")
    page.click_text("Baixar e ativar")
    page.wait_for("document.body.innerText.includes('Tentar novamente')", 60, "mensagem de falha")
    time.sleep(0.5)
    page.js("document.querySelector('[data-testid=feature-upscale_4x]').scrollIntoView({block: 'center'})")
    page.screenshot(out / "07-falha-download.png")

    print("8. Conexão restabelecida: Tentar novamente")
    server = ThreadingHTTPServer(("127.0.0.1", port), _RedirectToHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page.click_text("Tentar novamente")
        page.wait_for("document.body.innerText.includes('Ativando Ampliação 4×')", 30, "progresso do download")
        page.screenshot(out / "08-download-progresso.png")
        page.wait_for("!document.querySelector('[data-testid=feature-upscale_4x]')", 600, "download concluído")
    finally:
        server.shutdown()
        server.server_close()

    print("9. Aprimorar e ampliar 4× com o modelo baixado")
    for old in downloads.glob("*.png"):
        old.unlink()
    page.click_text("Aprimorar e ampliar 4×")
    page.wait_for("document.body.innerText.includes('Baixar PNG') && document.body.innerText.includes('Concluído')"
                  " && !document.body.innerText.includes('Processando')", 600, "resultado 4×")
    time.sleep(1.0)
    page.screenshot(out / "09-resultado-4x.png")
    page.click_text("Baixar PNG")
    deadline = time.time() + 60
    while time.time() < deadline and not (list(downloads.glob("*.png")) and not list(downloads.glob("*.crdownload"))):
        time.sleep(0.25)
    sample = urllib.request.urlopen(f"{url}/samples/promob_sample.jpg", timeout=10).read()
    w, h = jpeg_size(sample)
    got = png_size(next(downloads.glob("*.png")).read_bytes())
    assert got == (w * 4, h * 4), f"PNG 4× com {got}, esperado {(w * 4, h * 4)}"
    print(f"  PNG 4× baixado: {got[0]}×{got[1]}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="docs/screenshots")
    ap.add_argument("--failure-proxy-port", type=int, default=0,
                    help="testa falha e nova tentativa do download 4× (app iniciado com HF_ENDPOINT nessa porta)")
    ap.add_argument("--realismo-shot", default="06-realismo-ia.png", help="nome da captura do Realismo IA")
    args = ap.parse_args()
    browser = next((b for b in BROWSERS if b and os.path.isfile(b)), None)
    if not browser:
        print("Chrome ou Edge não encontrado (defina BROWSER_PATH).", file=sys.stderr)
        return 2
    try:
        run(args.url.rstrip("/"), Path(args.out), browser, args.realismo_shot, args.failure_proxy_port)
    except Exception as exc:
        print(f"FALHOU: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
