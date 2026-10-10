"""Hardware detection: static profile of the machine and live memory readings."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil

MIB = 1024 * 1024

# (minimum VRAM in MiB, tier). A "6 GB" card reports 6144 MiB, hence the slack.
GPU_TIERS = ((11500, "gpu-12+"), (7500, "gpu-8"), (5500, "gpu-6"), (3500, "gpu-4"))
CPU_TIER_MIN_RAM_MB = 12000
RUNNER_PROCESSES = ("llama-server", "ollama_llama_server")


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_free_mb: int


@dataclass(frozen=True)
class MemorySnapshot:
    ram_available_mb: int
    swap_used_mb: int
    # None when there is no usable GPU.
    vram_free_mb: int | None


@dataclass(frozen=True)
class HardwareProfile:
    cpu_model: str
    physical_cores: int
    logical_cores: int
    # None when it cannot be determined on this platform.
    has_avx2: bool | None
    ram_total_mb: int
    ram_available_mb: int
    swap_total_mb: int
    gpus: tuple[GpuInfo, ...]
    disk_free_mb: int
    models_dir: Path
    os: str
    arch: str

    @property
    def gpu(self) -> GpuInfo | None:
        """The GPU with the most memory; multi-GPU setups are not combined."""
        return max(self.gpus, key=lambda g: g.vram_total_mb, default=None)

    @property
    def tier(self) -> str:
        if self.gpu is not None:
            for min_vram, tier in GPU_TIERS:
                if self.gpu.vram_total_mb >= min_vram:
                    return tier
        return "cpu" if self.ram_total_mb >= CPU_TIER_MIN_RAM_MB else "cpu-low"


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def detect_gpus() -> tuple[GpuInfo, ...]:
    """NVIDIA GPUs via nvidia-smi. Other vendors are treated as CPU-only for now."""
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"]
    )
    gpus = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            gpus.append(GpuInfo(parts[0], int(parts[1]), int(parts[2])))
    return tuple(gpus)


def gpu_problem() -> str | None:
    """Why an installed NVIDIA driver cannot be queried, or None when there is nothing wrong."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    if out.returncode == 0 and "Failed" not in out.stdout:
        return None
    return (out.stdout + out.stderr).strip().splitlines()[0]


def _cpu_info() -> tuple[str, bool | None]:
    model, avx2 = platform.processor() or platform.machine(), None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                if key.strip() == "model name":
                    model = value.strip()
                elif key.strip() == "flags":
                    avx2 = "avx2" in value.split()
                    break
    except OSError:
        pass
    return model, avx2


def ollama_models_dir() -> Path:
    """Where Ollama stores models, used to measure the disk space available for pulls."""
    candidates = [
        Path(os.environ["OLLAMA_MODELS"]) if os.environ.get("OLLAMA_MODELS") else None,
        Path("/usr/share/ollama/.ollama/models"),
        Path.home() / ".ollama" / "models",
    ]
    return next((p for p in candidates if p and p.is_dir()), Path.home())


def ollama_service_env() -> dict[str, str] | None:
    """Environment of the Ollama systemd service, or None when it cannot be read."""
    out = _run(["systemctl", "show", "ollama", "-p", "Environment"])
    if out is None or not out.startswith("Environment="):
        return None
    pairs = (item.partition("=") for item in out.strip()[len("Environment="):].split())
    return {key: value for key, _, value in pairs}


def runner_rss_mb() -> int:
    """RAM held by the model server's runner processes, which the server itself under-reports."""
    total = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        if proc.info["name"] in RUNNER_PROCESSES and proc.info["memory_info"]:
            total += proc.info["memory_info"].rss
    return total // MIB


def read_memory() -> MemorySnapshot:
    gpus = detect_gpus()
    best = max(gpus, key=lambda g: g.vram_total_mb, default=None)
    return MemorySnapshot(
        ram_available_mb=psutil.virtual_memory().available // MIB,
        swap_used_mb=psutil.swap_memory().used // MIB,
        vram_free_mb=best.vram_free_mb if best else None,
    )


def detect() -> HardwareProfile:
    cpu_model, avx2 = _cpu_info()
    vm = psutil.virtual_memory()
    models_dir = ollama_models_dir()
    logical = psutil.cpu_count(logical=True) or 1
    return HardwareProfile(
        cpu_model=cpu_model,
        physical_cores=psutil.cpu_count(logical=False) or logical,
        logical_cores=logical,
        has_avx2=avx2,
        ram_total_mb=vm.total // MIB,
        ram_available_mb=vm.available // MIB,
        swap_total_mb=psutil.swap_memory().total // MIB,
        gpus=detect_gpus(),
        disk_free_mb=shutil.disk_usage(models_dir).free // MIB,
        models_dir=models_dir,
        os=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
    )
