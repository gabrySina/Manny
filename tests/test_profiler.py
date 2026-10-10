from pathlib import Path

import pytest

from manny.hardware import profiler
from manny.hardware.profiler import GpuInfo, HardwareProfile


def make_profile(ram_total_mb: int, vram_total_mb: int | None = None) -> HardwareProfile:
    gpus = (GpuInfo("test gpu", vram_total_mb, vram_total_mb),) if vram_total_mb else ()
    return HardwareProfile(
        cpu_model="test cpu",
        physical_cores=4,
        logical_cores=8,
        has_avx2=True,
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_total_mb // 2,
        swap_total_mb=0,
        gpus=gpus,
        disk_free_mb=50000,
        models_dir=Path("/tmp"),
        os="Linux",
        arch="x86_64",
    )


@pytest.mark.parametrize(
    ("ram", "vram", "tier"),
    [
        (4000, None, "cpu-low"),
        (16000, None, "cpu"),
        (16000, 2048, "cpu"),
        (16000, 4096, "gpu-4"),
        (16000, 6144, "gpu-6"),
        (32000, 8192, "gpu-8"),
        (32000, 12288, "gpu-12+"),
        (64000, 24576, "gpu-12+"),
    ],
)
def test_tier(ram, vram, tier):
    assert make_profile(ram, vram).tier == tier


def test_gpu_picks_largest():
    profile = make_profile(16000)
    small, big = GpuInfo("a", 4096, 4000), GpuInfo("b", 8192, 100)
    profile = HardwareProfile(**{**profile.__dict__, "gpus": (small, big)})
    assert profile.gpu is big


def test_detect_gpus_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: "NVIDIA GeForce RTX 2060, 6144, 5697\n")
    assert profiler.detect_gpus() == (GpuInfo("NVIDIA GeForce RTX 2060", 6144, 5697),)


def test_detect_gpus_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: None)
    assert profiler.detect_gpus() == ()


def test_detect_smoke():
    profile = profiler.detect()
    assert profile.ram_total_mb > 0
    assert profile.logical_cores >= profile.physical_cores >= 1
    assert profile.disk_free_mb >= 0


class Completed:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.mark.parametrize(
    ("installed", "result", "expected"),
    [
        (False, None, None),
        (True, Completed(0, "| NVIDIA-SMI 595.91 |"), None),
        (True, Completed(0, "Failed to initialize NVML: Driver/library version mismatch\nNVML library version: 595.99"),
         "Failed to initialize NVML: Driver/library version mismatch"),
        (True, Completed(9, "", "NVIDIA-SMI has failed\nmore"), "NVIDIA-SMI has failed"),
    ],
)
def test_gpu_problem(monkeypatch, installed, result, expected):
    monkeypatch.setattr(profiler.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if installed else None)
    monkeypatch.setattr(profiler.subprocess, "run", lambda *a, **k: result)
    assert profiler.gpu_problem() == expected


def test_ollama_service_env(monkeypatch):
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: "Environment=PATH=/usr/bin OLLAMA_NUM_PARALLEL=1 LLAMA_ARG_CACHE_RAM=1024\n")
    assert profiler.ollama_service_env() == {"PATH": "/usr/bin", "OLLAMA_NUM_PARALLEL": "1", "LLAMA_ARG_CACHE_RAM": "1024"}
    monkeypatch.setattr(profiler, "_run", lambda *a, **k: None)
    assert profiler.ollama_service_env() is None
