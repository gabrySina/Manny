"""Benchmark: how well and how fast a model does Manny's work on this machine.

Each model runs a fixed set of tasks through the real pipeline. Python tasks are graded
with tests the agents never see, so a model cannot pass by writing weak tests of its own.
"""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from manny.checks import check_web
from manny.config import Config
from manny.hardware import profiler
from manny.hardware.governor import MemoryGovernor, MemoryPressureError
from manny.llm.base import LLMBackendError
from manny.runtime import SetupError, run_task
from manny.sandbox.docker import DockerSandbox, SandboxError

BENCH_DIR = Path(__file__).parent / "data" / "bench"
HIDDEN_DIR = "_hidden_tests"
PYTEST = f"python -m pytest -q --no-header --tb=no -p no:cacheprovider {HIDDEN_DIR}"
PASSED = re.compile(r"(\d+) passed")
THINK_SUFFIXES = {"@think": True, "@nothink": False}


@dataclass(frozen=True)
class BenchTask:
    name: str
    grade: str
    prompt: str


@dataclass
class TaskOutcome:
    task: str
    passed: int
    total: int
    seconds: float
    error: str = ""


@dataclass
class ModelOutcome:
    spec: str
    placement: str = ""
    num_ctx: int = 0
    est_mb: int = 0
    model_size_mb: int = 0
    model_vram_mb: int = 0
    runner_ram_mb: int = 0
    min_ram_available_mb: int = 0
    tokens_per_second: float = 0.0
    tasks: list[TaskOutcome] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Mean share of tests passed per task, 0-100."""
        if not self.tasks:
            return 0.0
        return 100 * sum(t.passed / t.total for t in self.tasks) / len(self.tasks)

    @property
    def seconds(self) -> float:
        return sum(t.seconds for t in self.tasks)

    def as_dict(self) -> dict:
        return {**asdict(self), "score": round(self.score, 1), "seconds": round(self.seconds)}


def load_tasks(names: list[str] | None = None) -> list[BenchTask]:
    data = yaml.safe_load((BENCH_DIR / "tasks.yaml").read_text(encoding="utf-8"))
    tasks = [BenchTask(t["name"], t["grade"], t["prompt"]) for t in data["tasks"]]
    if names:
        unknown = set(names) - {t.name for t in tasks}
        if unknown:
            raise ValueError(f"unknown benchmark tasks: {sorted(unknown)}")
        tasks = [t for t in tasks if t.name in names]
    return tasks


def parse_spec(spec: str) -> tuple[str, bool | None]:
    """'model', 'model@think' or 'model@nothink': the model and its forced reasoning mode."""
    for suffix, think in THINK_SUFFIXES.items():
        if spec.endswith(suffix):
            return spec[: -len(suffix)], think
    return spec, None


def hidden_test_count(task: BenchTask) -> int:
    hidden = BENCH_DIR / task.name / "hidden"
    return sum(p.read_text(encoding="utf-8").count("\ndef test_") for p in hidden.glob("test_*.py"))


def grade(task: BenchTask, workspace: Path, cfg: Config, governor: MemoryGovernor) -> tuple[int, int]:
    """(tests passed, tests in total) for the finished workspace."""
    if task.grade == "web":
        return int(check_web(workspace)[0]), 1
    total = hidden_test_count(task)
    target = workspace / HIDDEN_DIR
    shutil.copytree(BENCH_DIR / task.name / "hidden", target)
    try:
        with DockerSandbox(workspace, cfg.sandbox, governor.sandbox_memory_mb(cfg.sandbox.memory_mb)) as sandbox:
            output = sandbox.exec(PYTEST).output
    finally:
        shutil.rmtree(target, ignore_errors=True)
    found = PASSED.search(output)
    return min(int(found.group(1)), total) if found else 0, total


def _log_events(path: Path) -> Callable[[str, str, Any], None]:
    def sink(source: str, kind: str, data: Any) -> None:
        text = data.model_dump_json() if hasattr(data, "model_dump_json") else str(data)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"[{source}] {kind}: {text[:2000]}\n")

    return sink


def run_bench(
    specs: list[str],
    cfg: Config,
    out_dir: Path,
    task_names: list[str] | None = None,
    progress: Callable[[str], None] = print,
) -> list[ModelOutcome]:
    tasks = load_tasks(task_names)
    # The benchmark measures the model, not the review loop built around it.
    cfg = cfg.model_copy(update={"agent": cfg.agent.model_copy(update={"max_review_rounds": 0})})
    governor = MemoryGovernor(cfg.memory, profiler.detect().ram_total_mb)
    outcomes = []
    for spec in specs:
        model, think = parse_spec(spec)
        outcome = ModelOutcome(spec)
        outcomes.append(outcome)
        speeds = []
        for task in tasks:
            workspace = out_dir / spec.replace("/", "_").replace(":", "_") / task.name
            seed = BENCH_DIR / task.name / "seed"
            if seed.is_dir():
                shutil.copytree(seed, workspace)
            else:
                workspace.mkdir(parents=True)
            started = time.monotonic()
            error = ""
            try:
                report = run_task(
                    task.prompt, workspace, cfg, _log_events(workspace.parent / f"{task.name}.log"), model, think
                )
            except SetupError as exc:
                # The model cannot run here at all: no point in trying the other tasks.
                progress(f"{spec}: {exc}")
                outcome.tasks.append(TaskOutcome(task.name, 0, 1, 0.0, str(exc)))
                break
            except (MemoryPressureError, LLMBackendError, SandboxError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            else:
                choice = report.model
                outcome.placement, outcome.num_ctx, outcome.est_mb = choice.placement, choice.num_ctx, choice.est_mb
                outcome.model_size_mb = max(outcome.model_size_mb, report.model_size_mb)
                outcome.model_vram_mb = max(outcome.model_vram_mb, report.model_vram_mb)
                outcome.runner_ram_mb = max(outcome.runner_ram_mb, report.runner_ram_mb)
                lowest = report.peak.min_ram_available_mb
                outcome.min_ram_available_mb = min(outcome.min_ram_available_mb or lowest, lowest)
                if report.tokens_per_second:
                    speeds.append(report.tokens_per_second)
            seconds = time.monotonic() - started
            passed, total = grade(task, workspace, cfg, governor)
            outcome.tasks.append(TaskOutcome(task.name, passed, total, seconds, error))
            progress(f"{spec} / {task.name}: {passed}/{total} in {seconds:.0f}s {error}")
        outcome.tokens_per_second = sum(speeds) / len(speeds) if speeds else 0.0
    return outcomes
