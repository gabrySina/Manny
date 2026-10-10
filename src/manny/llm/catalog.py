"""Model catalog and hardware-driven model selection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel

from manny.config import ModelConfig

# Context sizes tried, largest first.
CTX_STEPS = (32768, 16384, 8192, 6144, 4096, 2048)
# On the CPU alone generation is slow, so a huge context only hurts.
CPU_MAX_CTX = 8192
# Measured context memory relative to the bare KV cache (Ollama 0.40).
CONTEXT_OVERHEAD = 1.1
# Context memory per GiB of weights, for models whose architecture cannot be read.
FALLBACK_MB_PER_1K_CTX_PER_GIB = 30


class ModelEntry(BaseModel):
    name: str
    size_mb: int
    mb_per_1k_ctx: int
    max_ctx: int
    # Measured quality (share of benchmark tests passed, 0-100). Ranks the models that fit.
    score: float = 0.0
    # Share of the model that must sit on the GPU for a split to be usable. Sparse
    # (mixture-of-experts) models compute little per token and tolerate a low share.
    min_gpu_share: float = 0.75
    # May run without any GPU regardless of size.
    cpu_ok: bool = False
    # Reasoning mode for models that have one; None leaves the server default.
    think: bool | None = None

    def estimate_mb(self, num_ctx: int, overhead_mb: int) -> int:
        return self.size_mb + self.mb_per_1k_ctx * num_ctx // 1000 + overhead_mb


@dataclass(frozen=True)
class ModelChoice:
    name: str
    num_ctx: int
    # "gpu": fully offloaded; "split": partly on GPU, rest in RAM; "cpu": RAM only.
    placement: str
    est_mb: int
    size_mb: int
    installed: bool
    score: float = 0.0
    think: bool | None = None


def load_catalog(path: Path | None = None) -> list[ModelEntry]:
    if path is None:
        text = resources.files("manny").joinpath("data/models.yaml").read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return [ModelEntry.model_validate(m) for m in yaml.safe_load(text)["models"]]


def entry_from_metadata(name: str, size_bytes: int, info: dict, capabilities: list[str]) -> ModelEntry:
    """Catalog entry for a model known only through the metadata the server reports."""
    arch = info.get("general.architecture", "")
    layers = info.get(f"{arch}.block_count")
    width = info.get(f"{arch}.embedding_length")
    heads = info.get(f"{arch}.attention.head_count")
    kv_heads = info.get(f"{arch}.attention.head_count_kv", heads)
    size_mb = size_bytes // (1024 * 1024)
    if all(isinstance(v, int) and v > 0 for v in (layers, width, heads, kv_heads)):
        head_size = info.get(f"{arch}.attention.key_length") or width // heads
        # Hybrid architectures keep a KV cache only in their full-attention layers.
        attention_layers = layers // (info.get(f"{arch}.full_attention_interval") or 1)
        # Keys and values, 2 bytes each (f16), for every attention layer and KV head.
        kv_bytes_per_token = 2 * attention_layers * head_size * kv_heads * 2
        per_1k = int(kv_bytes_per_token * 1000 / (1024 * 1024) * CONTEXT_OVERHEAD)
    else:
        per_1k = FALLBACK_MB_PER_1K_CTX_PER_GIB * size_mb // 1024
    return ModelEntry(
        name=name,
        size_mb=size_mb,
        mb_per_1k_ctx=max(per_1k, 1),
        max_ctx=info.get(f"{arch}.context_length") or 8192,
        # Reasoning models are built to think first: measured on ornith:9b, turning it
        # off made the model wander on fixes that need an inference.
        think=True if "thinking" in capabilities else None,
    )


def place(
    entry: ModelEntry, ram_mb: int, vram_mb: int | None, cfg: ModelConfig, installed: bool = True
) -> ModelChoice | None:
    """How one specific model would run within the budget, or None if it does not fit."""
    placed = _place(entry, ram_mb, vram_mb, cfg)
    if placed is None:
        return None
    placement, ctx, est = placed
    return ModelChoice(entry.name, ctx, placement, est, entry.size_mb, installed, entry.score, entry.think)


def _place(
    entry: ModelEntry, ram_mb: int, vram_mb: int | None, cfg: ModelConfig
) -> tuple[str, int, int] | None:
    """Best (placement, num_ctx, est_mb) for one model, or None if it does not fit."""
    steps = [c for c in CTX_STEPS if cfg.min_ctx <= c <= min(cfg.max_ctx, entry.max_ctx)]
    cpu_allowed = entry.cpu_ok or entry.size_mb <= cfg.cpu_max_size_mb
    # Largest context first: agents gain more from room to work than from the last
    # layers on the GPU, as long as the model's own minimum GPU share is respected.
    for ctx in steps:
        est = entry.estimate_mb(ctx, cfg.overhead_mb)
        if vram_mb is not None and est <= vram_mb:
            return "gpu", ctx, est
        if vram_mb is not None and est <= vram_mb + ram_mb and est * entry.min_gpu_share <= vram_mb:
            return "split", ctx, est
        if ctx <= CPU_MAX_CTX and est <= ram_mb and cpu_allowed:
            return "cpu", ctx, est
    return None


def rank_choices(
    catalog: list[ModelEntry],
    ram_budget_mb: int,
    vram_budget_mb: int | None,
    disk_free_mb: int,
    installed: set[str],
    cfg: ModelConfig,
    disk_reserve_mb: int = 0,
) -> list[ModelChoice]:
    """All models that fit, best first: GPU-backed before CPU-only, then by measured
    quality, then by size.

    The tail of the list is the fallback chain used when memory gets tight.
    """
    if cfg.name is not None:
        catalog = [m for m in catalog if m.name == cfg.name]
        if not catalog:
            raise ValueError(f"model '{cfg.name}' is not in the catalog")
    choices = []
    for entry in catalog:
        is_installed = entry.name in installed
        if not is_installed and entry.size_mb + disk_reserve_mb > disk_free_mb:
            continue
        placed = _place(entry, ram_budget_mb, vram_budget_mb, cfg)
        if placed is not None:
            placement, ctx, est = placed
            choices.append(
                ModelChoice(entry.name, ctx, placement, est, entry.size_mb, is_installed, entry.score, entry.think)
            )
    choices.sort(key=lambda c: (c.placement == "cpu", -c.score, -c.size_mb))
    return choices
