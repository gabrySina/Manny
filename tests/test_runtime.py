from fakes import FakeBackend
from test_governor import FakeMemory, make_governor

from manny.llm.base import LoadedModel
from manny.runtime import model_budget


class Loaded(FakeBackend):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def loaded(self):
        return self.models


def test_memory_of_loaded_models_counts_as_available():
    # A 7B model left loaded by the previous run: 5.2 GB on the GPU, 0.7 GB in RAM.
    governor = make_governor(FakeMemory(ram=7000, vram=600))
    budget = model_budget(governor, Loaded([LoadedModel("qwen2.5-coder:7b", 5961, 5218)]))
    assert (budget.ram_mb, budget.vram_mb) == (7000 - 2400 + 743, 600 - 500 + 5218)


def test_budget_is_unchanged_when_nothing_is_loaded():
    governor = make_governor(FakeMemory(ram=7000, vram=5700))
    assert model_budget(governor, Loaded([])) == governor.budget()


def test_no_gpu_stays_no_gpu():
    governor = make_governor(FakeMemory(ram=7000))
    assert model_budget(governor, Loaded([LoadedModel("m", 3000, 0)])).vram_mb is None
