"""Local support services. Currently only SearXNG, the metasearch engine behind web_search."""

from __future__ import annotations

import secrets
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from manny.config import SearchConfig

CONTAINER = "manny-searxng"
IMAGE = "searxng/searxng:latest"
MEMORY_MB = 512
SETTINGS_PATH = Path.home() / ".local" / "share" / "manny" / "searxng" / "settings.yml"
# The JSON output format is off by default in SearXNG and is what web_search uses.
SETTINGS = """\
use_default_settings: true
server:
  secret_key: "{secret}"
  limiter: false
search:
  formats:
    - html
    - json
outgoing:
  request_timeout: 8.0
  max_request_timeout: 15.0
"""


class ServiceError(RuntimeError):
    pass


def _docker(*args: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    if shutil.which("docker") is None:
        raise ServiceError("Docker is not installed")
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def searxng_healthy(cfg: SearchConfig) -> bool:
    try:
        httpx.get(f"{cfg.searxng_url}/healthz", timeout=3.0).raise_for_status()
    except httpx.HTTPError:
        return False
    return True


def searxng_up(cfg: SearchConfig, wait_seconds: float = 60.0) -> None:
    """Start SearXNG bound to the loopback address of `cfg.searxng_url`."""
    if searxng_healthy(cfg):
        return
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(SETTINGS.format(secret=secrets.token_hex(32)), encoding="utf-8")
    port = urlsplit(cfg.searxng_url).port or 80
    _docker("rm", "-f", CONTAINER)
    started = _docker(
        "run", "-d", "--name", CONTAINER,
        "-p", f"127.0.0.1:{port}:8080",
        "--memory", f"{MEMORY_MB}m", "--memory-swap", f"{MEMORY_MB}m",
        # A resolver that drops queries otherwise stalls every lookup for 5 seconds.
        "--dns-opt", "timeout:1", "--dns-opt", "attempts:2",
        "-v", f"{SETTINGS_PATH}:/etc/searxng/settings.yml:ro",
        IMAGE,
    )
    if started.returncode != 0:
        raise ServiceError(f"cannot start SearXNG: {started.stderr.strip()[-500:]}")
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if searxng_healthy(cfg):
            return
        time.sleep(1.0)
    logs = _docker("logs", "--tail", "20", CONTAINER)
    raise ServiceError(f"SearXNG did not become healthy:\n{(logs.stdout + logs.stderr)[-1500:]}")


def searxng_down() -> None:
    _docker("rm", "-f", CONTAINER)
