import pytest
from fakes import FakeBackend, choice
from pydantic import BaseModel
from test_governor import FakeMemory, make_governor

from manny.hardware.governor import MemoryPressureError
from manny.llm.base import ChatResult, LLMFormatError
from manny.llm.manager import ModelManager

BIG, SMALL = choice("big", 5000), choice("small", 2700)
MSG = [{"role": "user", "content": "hi"}]


class Answer(BaseModel):
    value: int


def test_uses_best_model_when_it_fits():
    backend = FakeBackend(["ok"])
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [BIG, SMALL])
    assert manager.chat(MSG).content == "ok"
    assert backend.calls[0]["model"] == "big"


def test_downgrades_when_the_model_does_not_fit():
    backend = FakeBackend(["ok"])
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=3500)), [BIG, SMALL])
    manager.chat(MSG)
    assert backend.calls[0]["model"] == "small"


def test_unloads_other_models_before_downgrading():
    mem = FakeMemory(ram=8000, vram=1000)
    backend = FakeBackend(["ok"], loaded=["someone-elses-model"])
    backend.on_unload = lambda name: setattr(mem, "vram", 5700)
    manager = ModelManager(backend, make_governor(mem), [BIG, SMALL])
    manager.chat(MSG)
    assert backend.unloaded == ["someone-elses-model"]
    assert backend.calls[0]["model"] == "big"


def test_raises_when_not_even_the_smallest_fits():
    backend = FakeBackend(["ok"])
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=1000)), [BIG, SMALL])
    with pytest.raises(MemoryPressureError):
        manager.chat(MSG)
    assert backend.calls == []


def test_no_choices_is_an_error():
    with pytest.raises(MemoryPressureError):
        ModelManager(FakeBackend(), make_governor(FakeMemory(ram=8000)), [])


def test_loaded_model_skips_load_admission():
    backend = FakeBackend(["ok"], loaded=["big"])
    mem = FakeMemory(ram=8000, vram=200)
    ModelManager(backend, make_governor(mem), [BIG, SMALL]).chat(MSG)
    assert backend.calls[0]["model"] == "big" and mem.now == 0.0


def test_ram_pressure_swaps_a_cpu_model_for_a_smaller_one():
    big, small = choice("big", 3000, "cpu"), choice("small", 1000, "cpu")
    mem = FakeMemory(ram=2000)
    backend = FakeBackend(["ok"], loaded=["big"])
    backend.on_unload = lambda name: setattr(mem, "ram", 5000)
    manager = ModelManager(backend, make_governor(mem), [big, small])
    manager.chat(MSG)
    assert backend.unloaded == ["big"] and backend.calls[0]["model"] == "small"


def test_ram_pressure_with_gpu_model_aborts_cleanly():
    backend = FakeBackend(["ok"], loaded=["big"])
    manager = ModelManager(backend, make_governor(FakeMemory(ram=2000, vram=200)), [BIG, SMALL])
    with pytest.raises(MemoryPressureError):
        manager.chat(MSG)


def test_structured_retries_malformed_output():
    backend = FakeBackend(["not json", '{"value": "x"}', '{"value": 7}'])
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [BIG])
    assert manager.structured(MSG, Answer) == Answer(value=7)
    assert "Invalid reply" in backend.calls[-1]["messages"][-1]["content"]


def test_structured_gives_up():
    backend = FakeBackend(["nope"] * 3)
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [BIG])
    with pytest.raises(LLMFormatError):
        manager.structured(MSG, Answer)


def test_ram_pressure_first_reloads_the_same_model_to_drop_its_caches():
    mem = FakeMemory(ram=2000, vram=200)
    backend = FakeBackend(["ok"], loaded=["big"])

    def freed(name):
        mem.ram, mem.vram = 9000, 5700

    backend.on_unload = freed
    manager = ModelManager(backend, make_governor(mem), [BIG, SMALL])
    manager.chat(MSG)
    assert backend.unloaded == ["big"] and backend.calls[0]["model"] == "big"


def test_cut_off_reply_asks_for_a_shorter_one_without_echoing_it_all():
    cut = ChatResult('{"value": ' + "9" * 5000, 100, 3072, truncated=True)
    backend = FakeBackend([cut, '{"value": 7}'])
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [BIG])
    assert manager.structured(MSG, Answer) == Answer(value=7)
    echoed, request = backend.calls[1]["messages"][-2:]
    assert len(echoed["content"]) < 400
    assert "cut off" in request["content"] and "shorter" in request["content"]


def test_calls_that_do_not_choose_use_the_models_default_reasoning_mode():
    backend = FakeBackend(["a", "b", "c"])
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [choice("m", 3000, think=True)])
    manager.chat(MSG)
    manager.chat(MSG, think=False)
    manager.chat(MSG, think=True)
    assert [call["think"] for call in backend.calls] == [True, False, True]


def test_reasoning_calls_get_extra_output_room():
    calls = []

    class Recording(FakeBackend):
        def chat(self, model, messages, num_ctx, schema=None, max_tokens=None, think=None, temperature=None):
            calls.append((think, max_tokens))
            return super().chat(model, messages, num_ctx, schema, max_tokens, think, temperature)

    manager = ModelManager(
        Recording(["a", "b"]), make_governor(FakeMemory(ram=8000, vram=5700)),
        [choice("m", 3000, num_ctx=16384, think=False)],
    )
    manager.chat(MSG)
    manager.chat(MSG, think=True)
    assert calls == [(False, 3072), (True, 3072 + 1536)]


def test_replies_are_sized_to_the_time_left_and_none_is_asked_when_it_is_gone():
    sizes = []

    class Recording(FakeBackend):
        def chat(self, model, messages, num_ctx, schema=None, max_tokens=None, think=None, temperature=None):
            sizes.append(max_tokens)
            return super().chat(model, messages, num_ctx, schema, max_tokens, think, temperature)

    left = iter([600.0, 60.0, 2.0, 0.0])
    manager = ModelManager(
        Recording(['{"value": 1}'] * 3), make_governor(FakeMemory(ram=8000, vram=5700)),
        [choice("m", 3000, num_ctx=16384)],
    )
    for _ in range(3):
        manager.structured(MSG, Answer, remaining=lambda: next(left))
    # 15 tokens/s assumed: plenty of time, one minute, two seconds (never below the floor).
    assert sizes == [3072, 900, 256]
    with pytest.raises(LLMFormatError, match="no time left"):
        manager.structured(MSG, Answer, remaining=lambda: next(left))
