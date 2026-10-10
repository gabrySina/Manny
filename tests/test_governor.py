import pytest

from manny.config import MemoryConfig
from manny.hardware.governor import MemoryGovernor, MemoryPressureError
from manny.hardware.profiler import MemorySnapshot


class FakeMemory:
    """Sampler with settable readings and a clock that advances on sleep."""

    def __init__(self, ram: int, vram: int | None = None, swap: int = 0) -> None:
        self.ram, self.vram, self.swap = ram, vram, swap
        self.now = 0.0
        self.on_sleep = None

    def sample(self) -> MemorySnapshot:
        return MemorySnapshot(self.ram, self.swap, self.vram)

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep:
            self.on_sleep()

    def clock(self) -> float:
        return self.now


def make_governor(mem: FakeMemory, ram_total: int = 16000, **cfg) -> MemoryGovernor:
    return MemoryGovernor(MemoryConfig(**cfg), ram_total, mem.sample, mem.sleep, mem.clock)


def test_reserve_is_max_of_minimum_and_fraction():
    assert make_governor(FakeMemory(0), ram_total=8000).ram_reserve_mb == 2048
    assert make_governor(FakeMemory(0), ram_total=64000).ram_reserve_mb == 9600


def test_budget_subtracts_reserves():
    budget = make_governor(FakeMemory(ram=7500, vram=5697)).budget()
    assert (budget.ram_mb, budget.vram_mb) == (7500 - 2400, 5697 - 500)


def test_budget_never_negative_and_no_gpu():
    budget = make_governor(FakeMemory(ram=1000)).budget()
    assert (budget.ram_mb, budget.vram_mb) == (0, None)


def test_fits_rejects_vram_request_without_gpu():
    governor = make_governor(FakeMemory(ram=10000))
    assert governor.fits(ram_mb=1000)
    assert not governor.fits(vram_mb=1)


def test_fits_with_no_request_detects_pressure_on_the_reserve():
    assert make_governor(FakeMemory(ram=2400)).fits()
    assert not make_governor(FakeMemory(ram=2399)).fits()


def test_admit_returns_immediately_when_it_fits():
    mem = FakeMemory(ram=10000)
    make_governor(mem).admit("x", ram_mb=1000)
    assert mem.now == 0.0


def test_admit_waits_for_memory_to_free_up():
    mem = FakeMemory(ram=3000)
    mem.on_sleep = lambda: setattr(mem, "ram", 10000) if mem.now >= 3 else None
    make_governor(mem).admit("x", ram_mb=1000)
    assert mem.now == 3.0


def test_admit_runs_relievers_in_order_until_it_fits():
    mem = FakeMemory(ram=3000)
    calls = []

    def useless() -> bool:
        calls.append("useless")
        return False

    def frees() -> bool:
        calls.append("frees")
        mem.ram = 10000
        return True

    def never() -> bool:
        calls.append("never")
        return True

    make_governor(mem).admit("x", ram_mb=1000, relievers=[useless, frees, never])
    assert calls == ["useless", "frees"]


def test_admit_raises_when_nothing_helps():
    mem = FakeMemory(ram=3000, vram=600)
    with pytest.raises(MemoryPressureError, match="load model"):
        make_governor(mem).admit("load model", ram_mb=1000, vram_mb=4000, relievers=[lambda: True])
    assert mem.now >= 10.0


def test_sandbox_memory_is_capped_to_budget():
    assert make_governor(FakeMemory(ram=10000)).sandbox_memory_mb(1024) == 1024
    assert make_governor(FakeMemory(ram=3000)).sandbox_memory_mb(1024) == 600
    with pytest.raises(MemoryPressureError):
        make_governor(FakeMemory(ram=2500)).sandbox_memory_mb(1024)


def test_track_records_peaks():
    mem = FakeMemory(ram=8000, vram=5000, swap=100)
    governor = make_governor(mem)
    with governor.track(interval=60) as peak:
        mem.ram, mem.vram, mem.swap = 4000, 1000, 300
    assert (peak.min_ram_available_mb, peak.min_vram_free_mb, peak.max_swap_used_mb) == (4000, 1000, 300)
