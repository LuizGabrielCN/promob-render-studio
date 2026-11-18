"""Ativação sob demanda (4K e Realismo IA): pré-checagem CUDA, falha e nova tentativa."""
import os
import socket

import pytest

from app.core import features
from app.core.features import FeatureManager, cuda_preflight
from app.pipeline import super_res


def _gpu(name="NVIDIA GeForce RTX 2060", driver="617.14", vram=6144.0):
    return {"name": name, "driver_version": driver, "vram_mb": vram}


def test_preflight_without_nvidia_gpu_explains_and_keeps_classic(monkeypatch):
    monkeypatch.setattr(features, "query_nvidia_gpu", lambda: None)
    result = cuda_preflight()
    assert result["ok"] is False
    assert "NVIDIA" in result["reason"]
    assert "Clássico" in result["reason"]


def test_preflight_rejects_old_driver():
    result = cuda_preflight(_gpu(driver="472.12"))
    assert result["ok"] is False
    assert "527.41" in result["reason"]


def test_preflight_rejects_small_vram():
    result = cuda_preflight(_gpu(vram=2048.0))
    assert result["ok"] is False
    assert "memória de vídeo" in result["reason"]


def test_preflight_warns_below_recommended_vram():
    result = cuda_preflight(_gpu(vram=4096.0))
    assert result["ok"] is True
    assert result["warning"]


def test_preflight_accepts_tested_configuration():
    result = cuda_preflight(_gpu())
    assert result == {"ok": True, "gpu": _gpu(), "reason": None, "warning": None}


def test_preflight_can_be_forced_unavailable_for_tests(monkeypatch):
    monkeypatch.setenv("PROMOB_SIMULATE_NO_CUDA", "1")
    assert cuda_preflight(_gpu())["ok"] is False


@pytest.mark.parametrize("text,expected", [("617.14", (617, 14)), ("527.41", (527, 41)), ("", ())])
def test_parse_version(text, expected):
    assert features._parse_version(text) == expected


def test_upscale_activation_failure_then_retry(monkeypatch, tmp_path):
    """Uma falha de download aparece como 'failed' com mensagem; a nova tentativa conclui."""
    monkeypatch.setenv("PROMOB_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(super_res, "find_existing_model", lambda model_dir=None: None)
    calls = {"n": 0}

    def fake_download(target_dir=None, progress_callback=None, timeout=30.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Falha ao baixar o modelo Real-ESRGAN: sem conexão")
        progress_callback(10, 10)
        return os.path.join(target_dir, super_res.MODEL_FILENAME)

    monkeypatch.setattr(super_res, "download_model", fake_download)
    manager = FeatureManager()

    manager.start(features.FEATURE_UPSCALE)
    manager.wait(10)
    task = manager._tasks[features.FEATURE_UPSCALE]
    assert task.state == "failed"
    assert "sem conexão" in task.error
    assert task.unavailable is False

    manager.start(features.FEATURE_UPSCALE)
    manager.wait(10)
    task = manager._tasks[features.FEATURE_UPSCALE]
    assert task.state == "done"
    assert task.progress == 100.0
    assert calls["n"] == 2


def test_realismo_activation_stops_before_any_download_without_cuda(monkeypatch):
    monkeypatch.setattr(features, "query_nvidia_gpu", lambda: None)

    def forbidden(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("não deveria baixar nada sem CUDA")

    monkeypatch.setattr(FeatureManager, "_install_packages", forbidden)
    monkeypatch.setattr(FeatureManager, "_download_models", forbidden)
    manager = FeatureManager()
    manager.start(features.FEATURE_REALISMO)
    manager.wait(10)
    task = manager._tasks[features.FEATURE_REALISMO]
    assert task.state == "failed"
    assert task.unavailable is True
    assert "NVIDIA" in task.error


def test_only_one_activation_runs_at_a_time(monkeypatch):
    import threading

    gate = threading.Event()
    monkeypatch.setattr(FeatureManager, "_run_upscale", lambda self, task: gate.wait(5))
    manager = FeatureManager()
    manager.start(features.FEATURE_UPSCALE)
    # Repetir o mesmo recurso devolve a tarefa em andamento; outro recurso é recusado.
    assert manager.start(features.FEATURE_UPSCALE)["state"] == "running"
    with pytest.raises(RuntimeError):
        manager.start(features.FEATURE_REALISMO)
    gate.set()
    manager.wait(5)


def test_status_reports_sizes_and_reasons(monkeypatch):
    monkeypatch.setattr(features, "query_nvidia_gpu", lambda: None)
    status = FeatureManager().status()
    up = status["features"][features.FEATURE_UPSCALE]
    ia = status["features"][features.FEATURE_REALISMO]
    assert set(up) >= {"ready", "available", "download_bytes", "task"}
    if not up["ready"]:
        assert up["download_bytes"] == features.UPSCALE_DOWNLOAD_BYTES
    if not ia["ready"]:
        assert ia["available"] is False
        assert "NVIDIA" in ia["reason"]


def test_download_model_network_failure_leaves_no_partial_file(monkeypatch, tmp_path):
    # Porta fechada no próprio computador simula a falta de conexão sem acessar a internet.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{port}")
    with pytest.raises(RuntimeError, match="Real-ESRGAN"):
        super_res.download_model(str(tmp_path), timeout=2)
    assert list(tmp_path.iterdir()) == []


def test_models_dir_env_is_preferred_download_target(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMOB_MODELS_DIR", str(tmp_path))
    assert super_res.default_download_dir() == str(tmp_path)
    fake = tmp_path / super_res.MODEL_FILENAME
    with open(fake, "wb") as fh:
        fh.truncate(super_res.MODEL_EXPECTED_SIZE)
    assert super_res.find_existing_model() == str(fake)


def test_purge_partial_ai_imports_drops_namespace_leftovers(monkeypatch):
    """Um 'import torch' durante a cópia deixa um namespace package quebrado; ele deve ser descartado."""
    import sys
    import types

    # Nomes fictícios: mexer em 'torch' real apagaria submódulos usados por outros testes.
    monkeypatch.setattr(features, "AI_IMPORT_ROOTS", ("fakeai", "fakeok"))
    broken = types.ModuleType("fakeai")
    broken.__path__ = []  # namespace package: sem __file__
    monkeypatch.setitem(sys.modules, "fakeai", broken)
    monkeypatch.setitem(sys.modules, "fakeai.cuda", types.ModuleType("fakeai.cuda"))
    healthy = types.ModuleType("fakeok")
    healthy.__file__ = "fakeok/__init__.py"
    monkeypatch.setitem(sys.modules, "fakeok", healthy)

    removed = features.purge_partial_ai_imports()
    assert sorted(removed) == ["fakeai", "fakeai.cuda"]
    assert "fakeai" not in sys.modules
    assert sys.modules["fakeok"] is healthy


def test_purge_keeps_real_torch_loaded():
    import sys

    torch = pytest.importorskip("torch")
    features.purge_partial_ai_imports()
    assert sys.modules["torch"] is torch
