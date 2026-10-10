"""Backend-independent LLM interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

Message = dict[str, str]


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    # Time spent generating the completion, 0 when the backend does not report it.
    eval_seconds: float = 0.0
    # The reply stopped at the output token limit instead of ending by itself.
    truncated: bool = False


@dataclass(frozen=True)
class LoadedModel:
    name: str
    size_mb: int
    vram_mb: int


class LLMFormatError(RuntimeError):
    """The model kept producing output that does not match the requested schema."""


class LLMBackendError(RuntimeError):
    """The model server failed or is unreachable."""


class LLMBackend(Protocol):
    def chat(
        self,
        model: str,
        messages: list[Message],
        num_ctx: int,
        schema: dict | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """One completion. With `schema`, the reply must be JSON matching it.

        `think` switches a model's reasoning mode on or off; None leaves the default.
        `temperature` overrides the backend's low default for this one call.
        """
        ...

    def loaded(self) -> list[LoadedModel]: ...

    def unload(self, model: str) -> None: ...
