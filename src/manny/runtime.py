"""Wires hardware, model, sandbox and tools together and runs one task."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from manny.agents.orchestrator import Orchestrator, RunResult
from manny.checks import check_project_pages, check_report, check_web, server_entrypoints
from manny.config import Config
from manny.hardware import profiler
from manny.hardware.governor import Budget, MemoryGovernor, PeakUsage
from manny.llm.catalog import ModelChoice, load_catalog, place, rank_choices
from manny.llm.manager import ModelManager
from manny.llm.ollama import OllamaClient
from manny.sandbox.docker import DockerSandbox, Sandbox
from manny.services import searxng_healthy
from manny.tools.base import truncate_middle
from manny.tools.exec import exec_tools, run_pytest
from manny.tools.fs import ListArgs, Workspace, fs_tools
from manny.tools.web import web_tools


# A server is given this long to crash; one still running after it has started properly.
START_SECONDS = 8.0
MAX_START_CHECKS = 2
START_OUTPUT_CHARS = 1500


def model_budget(governor: MemoryGovernor, client: OllamaClient) -> Budget:
    """Memory available for a model, counting what Ollama's loaded models would free.

    A model left loaded by a previous run occupies the very memory the next run needs;
    without this, back-to-back runs would see a full GPU and settle for a smaller model.
    """
    budget = governor.budget()
    loaded = client.loaded()
    vram = budget.vram_mb
    if vram is not None:
        vram += sum(m.vram_mb for m in loaded)
    return Budget(budget.ram_mb + sum(m.size_mb - m.vram_mb for m in loaded), vram)


def check_python_project(sandbox: Sandbox, root: Path) -> tuple[bool, str]:
    """The test suite, then what tests with a test client never exercise: that a server
    really starts, and that the web pages coming with it are wired to it."""
    passed, report = run_pytest(sandbox)
    if not passed:
        return False, report
    for entry in server_entrypoints(root)[:MAX_START_CHECKS]:
        name = entry.relative_to(root)
        started = sandbox.exec(f"exec python {shlex.quote(str(name))}", timeout=START_SECONDS)
        # Still running when the time is up is what a server should do.
        if not started.timed_out and started.exit_code != 0:
            return False, (
                f"The tests pass, but `python {name}` crashes at startup instead of serving. "
                f"Fix it, and keep the start-up code working under the real framework:\n"
                + truncate_middle(started.output, START_OUTPUT_CHARS)
            )
    pages_ok, problems = check_project_pages(root)
    if not pages_ok:
        return False, f"The tests pass, but the web pages of the project have problems:\n{problems}"
    return True, report


class SetupError(RuntimeError):
    """A prerequisite is missing; the message says what to do."""


@dataclass(frozen=True)
class RunReport:
    result: RunResult
    model: ModelChoice
    peak: PeakUsage
    web_enabled: bool
    tokens_per_second: float = 0.0
    # Memory of the model as the server reports it, and the RAM its runner really holds.
    model_size_mb: int = 0
    model_vram_mb: int = 0
    runner_ram_mb: int = 0


def choose_models(
    cfg: Config, client: OllamaClient, budget: Budget, disk_free_mb: int, model: str | None, think: bool | None
) -> list[ModelChoice]:
    """The model to use and its fallbacks: the best that fits, or the one that was asked for."""
    catalog = load_catalog(cfg.model.catalog_path)
    installed = client.installed()
    forced = model or cfg.model.name
    if forced is None:
        ranked = rank_choices(
            catalog, budget.ram_mb, budget.vram_mb, disk_free_mb, installed, cfg.model, cfg.memory.disk_reserve_mb
        )
        choices = [c for c in ranked if c.installed]
        if not choices:
            hint = f"scaricane uno con: ollama pull {ranked[0].name}" if ranked else "la memoria disponibile è insufficiente"
            raise SetupError(f"nessun modello utilizzabile è installato; {hint}")
        return choices
    if forced not in installed:
        raise SetupError(f"il modello {forced} non è installato: ollama pull {forced}")
    # A model outside the catalog is sized from the metadata the server reports.
    entry = next((m for m in catalog if m.name == forced), None) or client.describe(forced)
    if think is not None:
        entry = entry.model_copy(update={"think": think})
    choice = place(entry, budget.ram_mb, budget.vram_mb, cfg.model)
    if choice is None:
        raise SetupError(
            f"{forced} non entra nel budget di memoria (RAM {budget.ram_mb} MB, VRAM {budget.vram_mb or 0} MB)"
        )
    return [choice]


def run_task(
    task: str,
    workspace: Path,
    cfg: Config,
    on_event: Callable[[str, str, Any], None] | None = None,
    model: str | None = None,
    think: bool | None = None,
) -> RunReport:
    profile = profiler.detect()
    governor = MemoryGovernor(cfg.memory, profile.ram_total_mb)
    client = OllamaClient(cfg.ollama.host, cfg.ollama.request_timeout, cfg.ollama.keep_alive)
    if client.version() is None:
        raise SetupError(f"Ollama non è raggiungibile su {cfg.ollama.host}")

    budget = model_budget(governor, client)
    choices = choose_models(cfg, client, budget, profile.disk_free_mb, model, think)
    if think is not None:
        thinking = "always" if think else "never"
        cfg = cfg.model_copy(update={"agent": cfg.agent.model_copy(update={"thinking": thinking})})
    manager = ModelManager(client, governor, choices)

    ws = Workspace(workspace)
    tools = {t.name: t for t in fs_tools(ws)}
    seen_urls: set[str] = set()
    web_enabled = searxng_healthy(cfg.search)
    if web_enabled:
        tools.update({t.name: t for t in web_tools(cfg.search, seen=seen_urls)})

    with DockerSandbox(ws.root, cfg.sandbox, governor.sandbox_memory_mb(cfg.sandbox.memory_mb)) as sandbox:
        tools.update({t.name: t for t in exec_tools(sandbox)})
        verifiers = {"python": lambda: check_python_project(sandbox, ws.root), "web": lambda: check_web(ws.root)}
        if web_enabled:
            verifiers["research"] = lambda: check_report(ws.root, seen_urls)
        orchestrator = Orchestrator(
            manager,
            tools,
            cfg.agent,
            profile.tier,
            list_files=lambda: tools["list_files"].run(ListArgs()),
            verifiers=verifiers,
            snapshot=ws.snapshot,
            checkpoint=ws.checkpoint,
            fingerprint=ws.fingerprint,
            on_event=on_event,
        )
        try:
            with governor.track() as peak:
                result = orchestrator.run(task)
        finally:
            loaded = next((m for m in client.loaded() if m.name == manager.current.name), None)
            runner_ram_mb = profiler.runner_rss_mb()
            if cfg.ollama.unload_after_run:
                manager.unload()
    speed = manager.completion_tokens / manager.eval_seconds if manager.eval_seconds else 0.0
    return RunReport(
        result, manager.current, peak, web_enabled, speed,
        loaded.size_mb if loaded else 0, loaded.vram_mb if loaded else 0, runner_ram_mb,
    )
