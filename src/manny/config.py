"""Configuration: built-in defaults, optionally overridden by a YAML file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

USER_CONFIG_PATHS = (Path("manny.yaml"), Path.home() / ".config" / "manny" / "config.yaml")


class MemoryConfig(BaseModel):
    # RAM always left to the rest of the system: max(min, fraction of total).
    ram_reserve_min_mb: int = 2048
    ram_reserve_fraction: float = 0.15
    vram_margin_mb: int = 500
    disk_reserve_mb: int = 2048
    # How long an admission request waits for memory before degrading.
    wait_seconds: float = 10.0
    poll_seconds: float = 1.0


class ModelConfig(BaseModel):
    # Force a specific catalog model instead of auto-selecting one.
    name: str | None = None
    catalog_path: Path | None = None
    min_ctx: int = 4096
    # 16384 costs about 20% in speed over 8192 on a 6 GB card, but lets an agent hold
    # the files it is editing in full.
    max_ctx: int = 16384
    # Largest model allowed to run without a GPU; bigger ones are too slow for agents.
    cpu_max_size_mb: int = 2500
    # Fixed runtime buffers on top of weights and context memory.
    overhead_mb: int = 250


class OllamaConfig(BaseModel):
    host: str = "http://127.0.0.1:11434"
    keep_alive: str = "5m"
    # Free the model's RAM and VRAM when a run ends, at the cost of reloading it next time.
    unload_after_run: bool = True
    request_timeout: float = 600.0


class AgentConfig(BaseModel):
    # When agents use a model's reasoning mode. "auto": off for ordinary steps, on for
    # fixers and for an agent that keeps repeating itself. Reasoning roughly doubles the
    # time per step but is what gets a fix past a problem that needs an inference.
    thinking: Literal["auto", "always", "never"] = "auto"
    max_steps: int = 25
    max_fix_attempts: int = 4
    # Review-and-fix rounds after the plan is done; 0 disables the review.
    max_review_rounds: int = 1
    # Review findings sent to the fixer per round, one at a time, most severe first.
    max_fix_issues: int = 3
    # Wall-clock limits: one agent task, and the whole run before it wraps up.
    task_minutes: float = 10.0
    run_minutes: float = 30.0
    tool_output_chars: int = 4000


class SandboxConfig(BaseModel):
    image: str = "manny-sandbox:latest"
    memory_mb: int = 1024
    cpus: float = 2.0
    pids_limit: int = 256
    command_timeout: float = 120.0


class SearchConfig(BaseModel):
    searxng_url: str = "http://127.0.0.1:8888"
    max_results: int = 5
    fetch_chars: int = 6000


class Config(BaseModel):
    memory: MemoryConfig = MemoryConfig()
    model: ModelConfig = ModelConfig()
    ollama: OllamaConfig = OllamaConfig()
    agent: AgentConfig = AgentConfig()
    sandbox: SandboxConfig = SandboxConfig()
    search: SearchConfig = SearchConfig()


def load_config(path: Path | None = None) -> Config:
    """Load the config from `path`, $MANNY_CONFIG, ./manny.yaml or ~/.config/manny/config.yaml."""
    if path is None and os.environ.get("MANNY_CONFIG"):
        path = Path(os.environ["MANNY_CONFIG"])
    if path is None:
        path = next((p for p in USER_CONFIG_PATHS if p.is_file()), None)
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as fh:
        return Config.model_validate(yaml.safe_load(fh) or {})
