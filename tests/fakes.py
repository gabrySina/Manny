"""Deterministic stand-ins for the LLM backend, used across the test suite."""

from __future__ import annotations

import json

from manny.llm.base import ChatResult, LoadedModel
from manny.llm.catalog import ModelChoice


def choice(name: str, est_mb: int, placement: str = "gpu", num_ctx: int = 8192, think=None) -> ModelChoice:
    return ModelChoice(name, num_ctx, placement, est_mb, est_mb, True, think=think)


def action(tool: str, **args) -> str:
    return json.dumps({"thought": "", "tool": tool, "args": args})


class FakeBackend:
    """Replays scripted replies and tracks which models are loaded."""

    def __init__(self, replies=(), loaded=()) -> None:
        self.replies = list(replies)
        self.loaded_names = list(loaded)
        self.calls: list[dict] = []
        self.unloaded: list[str] = []
        self.on_unload = None

    def chat(self, model, messages, num_ctx, schema=None, max_tokens=None, think=None, temperature=None) -> ChatResult:
        self.calls.append({
            "model": model, "messages": [dict(m) for m in messages], "num_ctx": num_ctx,
            "think": think, "temperature": temperature,
        })
        if model not in self.loaded_names:
            self.loaded_names.append(model)
        reply = self.replies.pop(0)
        if isinstance(reply, ChatResult):
            return reply
        return ChatResult(reply, 100, 20)

    def loaded(self) -> list[LoadedModel]:
        return [LoadedModel(name, 0, 0) for name in self.loaded_names]

    def unload(self, model: str) -> None:
        self.loaded_names.remove(model)
        self.unloaded.append(model)
        if self.on_unload:
            self.on_unload(model)
