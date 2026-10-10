"""Memory governor: decides whether something may allocate memory right now.

Every memory-hungry action (loading a model, an LLM call, starting a sandbox
container) asks for admission first. When memory is short the governor waits,
then runs the caller's relievers (cheapest first), and finally refuses with
MemoryPressureError rather than letting the machine swap or hit the OOM killer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from manny.config import MemoryConfig
from manny.hardware.profiler import MemorySnapshot, read_memory

MIN_SANDBOX_MB = 256


class MemoryPressureError(RuntimeError):
    """Not enough memory, even after waiting and degrading."""


@dataclass(frozen=True)
class Budget:
    ram_mb: int
    # None when there is no usable GPU.
    vram_mb: int | None


@dataclass
class PeakUsage:
    min_ram_available_mb: int
    max_swap_used_mb: int
    min_vram_free_mb: int | None
    start_swap_used_mb: int = 0
    samples: int = 1

    @property
    def swap_growth_mb(self) -> int:
        return self.max_swap_used_mb - self.start_swap_used_mb

    def update(self, snap: MemorySnapshot) -> None:
        self.min_ram_available_mb = min(self.min_ram_available_mb, snap.ram_available_mb)
        self.max_swap_used_mb = max(self.max_swap_used_mb, snap.swap_used_mb)
        if snap.vram_free_mb is not None:
            current = self.min_vram_free_mb
            self.min_vram_free_mb = (
                snap.vram_free_mb if current is None else min(current, snap.vram_free_mb)
            )
        self.samples += 1


class MemoryGovernor:
    def __init__(
        self,
        cfg: MemoryConfig,
        ram_total_mb: int,
        sampler: Callable[[], MemorySnapshot] = read_memory,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.ram_reserve_mb = max(
            cfg.ram_reserve_min_mb, int(ram_total_mb * cfg.ram_reserve_fraction)
        )
        self._sampler = sampler
        self._sleep = sleep
        self._clock = clock

    def snapshot(self) -> MemorySnapshot:
        return self._sampler()

    def budget(self, snap: MemorySnapshot | None = None) -> Budget:
        """Memory that can still be allocated without eating into the reserves."""
        snap = snap or self.snapshot()
        vram = None
        if snap.vram_free_mb is not None:
            vram = max(0, snap.vram_free_mb - self.cfg.vram_margin_mb)
        return Budget(max(0, snap.ram_available_mb - self.ram_reserve_mb), vram)

    def fits(self, ram_mb: int = 0, vram_mb: int = 0) -> bool:
        """Whether the request fits. With no request, whether the RAM reserve is intact."""
        snap = self.snapshot()
        if ram_mb > snap.ram_available_mb - self.ram_reserve_mb:
            return False
        if vram_mb == 0:
            return True
        return snap.vram_free_mb is not None and vram_mb <= snap.vram_free_mb - self.cfg.vram_margin_mb

    def admit(
        self,
        what: str,
        ram_mb: int = 0,
        vram_mb: int = 0,
        relievers: Sequence[Callable[[], bool]] = (),
    ) -> None:
        """Block until the request fits, or raise MemoryPressureError.

        Waits up to `wait_seconds`, then calls each reliever in order (a
        reliever frees memory and returns True if it did anything), re-checking
        after each one.
        """
        deadline = self._clock() + self.cfg.wait_seconds
        while True:
            if self.fits(ram_mb, vram_mb):
                return
            if self._clock() >= deadline:
                break
            self._sleep(self.cfg.poll_seconds)
        for relieve in relievers:
            if relieve() and self.fits(ram_mb, vram_mb):
                return
        budget = self.budget()
        raise MemoryPressureError(
            f"{what}: needs {ram_mb} MB RAM / {vram_mb} MB VRAM, "
            f"budget is {budget.ram_mb} MB RAM / {budget.vram_mb or 0} MB VRAM"
        )

    def sandbox_memory_mb(self, requested_mb: int) -> int:
        """Memory limit for a sandbox container: the request, capped to the RAM budget."""
        granted = min(requested_mb, self.budget().ram_mb)
        if granted < MIN_SANDBOX_MB:
            raise MemoryPressureError(
                f"sandbox: only {granted} MB of RAM budget left, need {MIN_SANDBOX_MB} MB"
            )
        return granted

    @contextmanager
    def track(self, interval: float = 2.0) -> Iterator[PeakUsage]:
        """Sample memory in the background and record the peaks of the enclosed block."""
        first = self.snapshot()
        peak = PeakUsage(
            first.ram_available_mb, first.swap_used_mb, first.vram_free_mb, first.swap_used_mb
        )
        stop = threading.Event()

        def sample() -> None:
            while not stop.wait(interval):
                peak.update(self.snapshot())

        thread = threading.Thread(target=sample, name="manny-memory-monitor", daemon=True)
        thread.start()
        try:
            yield peak
        finally:
            stop.set()
            thread.join()
            peak.update(self.snapshot())
