"""Launcher do Windows: porta ocupada, instância única e identificação do servidor."""
import importlib.util
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "packaging" / "windows" / "promob_launcher.py"
spec = importlib.util.spec_from_file_location("promob_launcher", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_busy_port_is_skipped():
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    busy_port = busy.getsockname()[1]
    other = _free_port()
    try:
        sock = launcher.bind_listen_socket([busy_port, other])
        try:
            host, port = sock.getsockname()
            assert host == "127.0.0.1"
            assert port == other
        finally:
            sock.close()
    finally:
        busy.close()


def test_falls_back_to_system_port_when_all_preferred_are_busy():
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    try:
        sock = launcher.bind_listen_socket([busy.getsockname()[1]])
        try:
            assert sock.getsockname()[0] == "127.0.0.1"
            assert sock.getsockname()[1] != busy.getsockname()[1]
        finally:
            sock.close()
    finally:
        busy.close()


def test_single_instance_lock(tmp_path):
    first = launcher.SingleInstanceLock(tmp_path / "instancia.lock")
    second = launcher.SingleInstanceLock(tmp_path / "instancia.lock")
    assert first.acquire() is True
    try:
        assert second.acquire() is False
    finally:
        first.release()
    assert second.acquire() is True
    second.release()


def test_state_roundtrip_and_clear_only_own_instance(tmp_path):
    launcher.write_state(tmp_path, {"url": "http://127.0.0.1:8000", "instance": "abc"})
    assert launcher.read_state(tmp_path)["instance"] == "abc"
    launcher.clear_state(tmp_path, "outra")
    assert launcher.read_state(tmp_path) is not None
    launcher.clear_state(tmp_path, "abc")
    assert launcher.read_state(tmp_path) is None


class _Handler(BaseHTTPRequestHandler):
    payload = {}

    def do_GET(self):  # noqa: N802
        body = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize(
    "payload,instance,expected",
    [
        ({"status": "ok"}, None, False),  # outro programa na porta
        ({"app": "promob-render-studio", "instance": "x"}, None, True),
        ({"app": "promob-render-studio", "instance": "x"}, "x", True),
        ({"app": "promob-render-studio", "instance": "x"}, "y", False),
    ],
)
def test_probe_identifies_promob_instance(payload, instance, expected):
    handler = type("H", (_Handler,), {"payload": payload})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert launcher.probe(url, instance) is expected
    finally:
        server.shutdown()
        server.server_close()


def test_probe_unreachable_returns_false():
    assert launcher.probe(f"http://127.0.0.1:{_free_port()}", timeout=0.5) is False


def test_stop_without_running_instance_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMOB_DATA_DIR", str(tmp_path))
    assert launcher.stop(timeout=0.1) == 0
