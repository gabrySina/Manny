"""Docker sandbox: one resource-limited container per task, workspace mounted inside."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Protocol

from manny.config import SandboxConfig
from manny.tools.fs import SANDBOX_MOUNT

# Exit status of a process killed with SIGKILL (timeout or container memory limit).
KILLED = 137
ENVIRONMENT = {
    "HOME": "/sandbox",
    "PYTHONUSERBASE": "/sandbox/.local",
    "PATH": "/sandbox/.local/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_CACHE_DIR": "1",
}


class SandboxError(RuntimeError):
    """Docker itself failed (as opposed to a command failing inside the sandbox)."""


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool = False


class Sandbox(Protocol):
    def exec(self, command: str, timeout: float | None = None, network: bool = False) -> ExecResult:
        """Run a shell command in the workspace. The network is off unless requested."""
        ...


def _docker(*args: str, input: str | None = None, timeout: float = 60.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *args], input=input, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxError(f"docker {args[0]} failed: {exc}") from exc


def ensure_image(image: str) -> None:
    """Build the sandbox image from the bundled Dockerfile unless it already exists."""
    if _docker("image", "inspect", image).returncode == 0:
        return
    dockerfile = resources.files("manny").joinpath("data/sandbox.Dockerfile").read_text(encoding="utf-8")
    built = _docker("build", "-t", image, "-", input=dockerfile, timeout=900.0)
    if built.returncode != 0:
        raise SandboxError(f"cannot build {image}: {built.stderr.strip()[-500:]}")


class DockerSandbox:
    def __init__(self, workspace: Path, cfg: SandboxConfig, memory_mb: int) -> None:
        self._workspace = workspace.resolve()
        self._cfg = cfg
        self._memory_mb = memory_mb
        self.name = f"manny-{uuid.uuid4().hex[:12]}"
        self._started = False

    def __enter__(self) -> DockerSandbox:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        ensure_image(self._cfg.image)
        args = [
            "run", "-d", "--rm", "--init", "--name", self.name, "--label", "manny=sandbox",
            "-v", f"{self._workspace}:{SANDBOX_MOUNT}", "-w", SANDBOX_MOUNT,
            # Same value for both: the container gets no swap on top of its memory.
            "--memory", f"{self._memory_mb}m", "--memory-swap", f"{self._memory_mb}m",
            "--cpus", str(self._cfg.cpus), "--pids-limit", str(self._cfg.pids_limit),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            # A resolver that drops queries otherwise stalls every lookup for 5 seconds.
            "--dns-opt", "timeout:1", "--dns-opt", "attempts:2",
        ]
        if hasattr(os, "getuid"):
            # Files created in the workspace stay owned by the host user.
            args += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for key, value in ENVIRONMENT.items():
            args += ["-e", f"{key}={value}"]
        started = _docker(*args, self._cfg.image, "sleep", "infinity")
        if started.returncode != 0:
            raise SandboxError(f"cannot start sandbox: {started.stderr.strip()[-500:]}")
        self._started = True
        self._set_network(False)

    def _set_network(self, enabled: bool) -> None:
        done = _docker("network", "connect" if enabled else "disconnect", "bridge", self.name)
        if done.returncode != 0:
            # Fail closed: a sandbox whose network state is unknown is not usable.
            self.close()
            raise SandboxError(f"cannot switch sandbox network: {done.stderr.strip()[-300:]}")

    def exec(self, command: str, timeout: float | None = None, network: bool = False) -> ExecResult:
        if not self._started:
            raise SandboxError("sandbox is not running")
        timeout = timeout or self._cfg.command_timeout
        if network:
            self._set_network(True)
        started = time.monotonic()
        try:
            done = _docker(
                "exec", self.name, "timeout", "-s", "KILL", str(int(timeout)), "sh", "-c", command,
                timeout=timeout + 30,
            )
        finally:
            if network:
                self._set_network(False)
        timed_out = done.returncode == KILLED and time.monotonic() - started >= timeout
        return ExecResult(done.returncode, (done.stdout + done.stderr).strip(), timed_out)

    def close(self) -> None:
        if self._started:
            self._started = False
            _docker("rm", "-f", self.name)
