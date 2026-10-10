import pytest

from manny.config import ModelConfig
from manny.llm.catalog import ModelEntry, entry_from_metadata, load_catalog, place, rank_choices

CATALOG = load_catalog()


def rank(ram, vram, disk=100000, installed=(), **cfg):
    return rank_choices(CATALOG, ram, vram, disk, set(installed), ModelConfig(**cfg), disk_reserve_mb=2048)


QWEN = [m for m in CATALOG if m.name.startswith("qwen2.5-coder")]


def rank_qwen(ram, vram, disk=100000, installed=(), **cfg):
    cfg.setdefault("max_ctx", 16384)
    return rank_choices(QWEN, ram, vram, disk, set(installed), ModelConfig(**cfg), disk_reserve_mb=2048)


def test_dev_machine_gets_7b_mostly_on_gpu():
    # RTX 2060 6 GB with ~5.5 GB free, 16 GB RAM.
    choices = rank_qwen(ram=8200, vram=4998)
    assert [(c.name, c.placement, c.num_ctx) for c in choices] == [
        ("qwen2.5-coder:7b", "split", 16384),
        ("qwen2.5-coder:3b", "gpu", 16384),
        ("qwen2.5-coder:1.5b", "gpu", 16384),
    ]


def test_estimates_match_measurements():
    # `ollama ps` sizes at 4096 / 8192 / 16384 tokens on Ollama 0.40.
    measured = {
        "qwen2.5-coder:7b": (4886, 5129, 5543),
        "ornith:9b": (5686, 5805, 6104),
    }
    for name, sizes in measured.items():
        entry = next(m for m in CATALOG if m.name == name)
        for ctx, size in zip((4096, 8192, 16384), sizes):
            assert abs(entry.estimate_mb(ctx, ModelConfig().overhead_mb) - size) < 150, (name, ctx)


def test_split_is_refused_when_too_little_fits_on_gpu():
    choices = rank_qwen(ram=20000, vram=6000)
    assert (choices[0].name, choices[0].placement) == ("qwen2.5-coder:7b", "gpu")
    assert "qwen2.5-coder:14b" not in [c.name for c in choices]


def test_bigger_model_mostly_on_gpu_beats_smaller_fully_on_gpu():
    choices = rank_qwen(ram=20000, vram=8000)
    assert [(c.name, c.placement) for c in choices[:2]] == [
        ("qwen2.5-coder:14b", "split"),
        ("qwen2.5-coder:7b", "gpu"),
    ]


def test_big_gpu_gets_biggest_model():
    best = rank_qwen(ram=20000, vram=23500)[0]
    assert (best.name, best.placement, best.num_ctx) == ("qwen2.5-coder:32b", "gpu", 16384)


def test_cpu_only_is_limited_to_small_models_and_contexts():
    choices = rank_qwen(ram=20000, vram=None)
    assert choices[0].name == "qwen2.5-coder:3b"
    assert all(c.placement == "cpu" and c.num_ctx <= 8192 for c in choices)


def test_low_memory_cpu_falls_back_to_smallest():
    choices = rank_qwen(ram=1500, vram=None)
    assert [c.name for c in choices] == ["qwen2.5-coder:1.5b"]


def test_nothing_fits():
    assert rank_qwen(ram=500, vram=None) == []


def test_models_not_installed_need_disk_space():
    choices = rank_qwen(ram=5100, vram=5197, disk=3000, installed=["qwen2.5-coder:3b"])
    assert [c.name for c in choices] == ["qwen2.5-coder:3b", "qwen2.5-coder:1.5b"]
    assert choices[0].installed and not choices[1].installed


def test_forced_model():
    assert [c.name for c in rank(ram=5100, vram=5197, name="qwen2.5-coder:3b")] == ["qwen2.5-coder:3b"]
    with pytest.raises(ValueError):
        rank(ram=5100, vram=5197, name="does-not-exist")


QWEN_7B_INFO = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.embedding_length": 3584,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.context_length": 32768,
}


def test_entry_from_metadata_matches_the_hand_calibrated_catalog():
    entry = entry_from_metadata("qwen2.5-coder:7b", 4683 * 1024 * 1024, QWEN_7B_INFO, ["completion", "tools"])
    reference = next(m for m in CATALOG if m.name == "qwen2.5-coder:7b")
    assert abs(entry.mb_per_1k_ctx - reference.mb_per_1k_ctx) <= 8
    assert (entry.size_mb, entry.max_ctx, entry.think) == (4683, 32768, None)


def test_entry_from_metadata_handles_hybrid_attention():
    info = {
        "general.architecture": "qwen35", "qwen35.block_count": 32, "qwen35.embedding_length": 4096,
        "qwen35.attention.head_count": 16, "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256, "qwen35.full_attention_interval": 4,
        "qwen35.context_length": 262144,
    }
    entry = entry_from_metadata("ornith:9b", 5368 * 1024 * 1024, info, ["tools", "thinking"])
    reference = next(m for m in CATALOG if m.name == "ornith:9b")
    assert abs(entry.mb_per_1k_ctx - reference.mb_per_1k_ctx) <= 5 and entry.think is True


def test_entry_from_metadata_without_architecture_details_and_with_thinking():
    entry = entry_from_metadata("mystery:9b", 6 * 1024**3, {"general.architecture": "new"}, ["thinking"])
    assert entry.mb_per_1k_ctx == 180 and entry.max_ctx == 8192 and entry.think is True


def entry(name, size_mb, **extra):
    return ModelEntry(name=name, size_mb=size_mb, mb_per_1k_ctx=50, max_ctx=32768, **extra)


def test_measured_score_outranks_size():
    catalog = [entry("big-but-weak", 4000, score=40), entry("small-but-strong", 3000, score=70)]
    choices = rank_choices(catalog, 8000, 6000, 100000, set(), ModelConfig())
    assert [c.name for c in choices] == ["small-but-strong", "big-but-weak"]
    assert choices[0].score == 70


def test_sparse_models_may_sit_mostly_in_ram():
    dense, sparse = entry("dense", 11000), entry("sparse", 11000, min_gpu_share=0.3, think=True)
    assert place(dense, 8400, 4900, ModelConfig()) is None
    choice = place(sparse, 8400, 4900, ModelConfig())
    assert (choice.placement, choice.think) == ("split", True)
    assert place(sparse, 5000, 4900, ModelConfig()) is None


def test_cpu_ok_lifts_the_size_limit_without_a_gpu():
    assert place(entry("m", 6000), 12000, None, ModelConfig()) is None
    assert place(entry("m", 6000, cpu_ok=True), 12000, None, ModelConfig()).placement == "cpu"


def test_dev_machine_defaults_to_the_best_benchmarked_model():
    # RTX 2060 6 GB, 16 GB RAM, default configuration.
    choices = rank_choices(CATALOG, 8200, 4998, 100000, {"ornith:9b", "qwen2.5-coder:7b"}, ModelConfig())
    # As at run time, only installed models are candidates.
    choices = [c for c in choices if c.installed]
    assert [(c.name, c.num_ctx, c.placement) for c in choices[:2]] == [
        ("ornith:9b", 16384, "split"),
        ("qwen2.5-coder:7b", 16384, "split"),
    ]
    assert choices[0].think is True


def test_without_a_gpu_ornith_is_too_big_and_a_small_model_is_used():
    choices = rank_choices(CATALOG, 8200, None, 100000, set(), ModelConfig())
    assert choices[0].name == "qwen2.5-coder:3b"
