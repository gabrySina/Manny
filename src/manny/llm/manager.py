"""Keeps one model loaded within the memory budget and serves all agents through it."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from manny.hardware.governor import MemoryGovernor, MemoryPressureError
from manny.llm.base import ChatResult, LLMBackend, LLMFormatError, Message
from manny.llm.catalog import ModelChoice

T = TypeVar("T", bound=BaseModel)

# About 250 lines of code per reply: longer replies are slow and usually end up cut off.
MAX_OUTPUT_TOKENS = 3072
# How much of a cut-off reply is kept in the conversation.
TRUNCATED_ECHO_CHARS = 300
# Extra output allowance for a call that reasons before answering.
THINKING_TOKENS = 1536


class ModelManager:
    """Single entry point for LLM calls.

    Calls are serialized by construction (agents run one at a time), every call
    is admitted by the governor first, and under memory pressure the manager
    walks down `choices`, the fallback chain from the catalog.
    """

    def __init__(
        self,
        backend: LLMBackend,
        governor: MemoryGovernor,
        choices: list[ModelChoice],
        format_retries: int = 2,
    ) -> None:
        if not choices:
            raise MemoryPressureError("no model in the catalog fits the memory budget")
        self._backend = backend
        self._governor = governor
        self._chain = choices
        self._format_retries = format_retries
        self.current = choices[0]
        self.last_prompt_tokens = 0
        # Totals for the session, to report generation speed.
        self.completion_tokens = 0
        self.eval_seconds = 0.0

    @property
    def max_output_tokens(self) -> int:
        return min(MAX_OUTPUT_TOKENS, self.current.num_ctx // 3)

    def chat(
        self,
        messages: list[Message],
        schema: dict | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> ChatResult:
        """One completion. `think` overrides the model's default reasoning mode for this
        call; it is ignored for models that have no reasoning mode."""
        self._admit()
        if think is None or self.current.think is None:
            think = self.current.think
        # Reasoning tokens count against the same limit as the reply: without extra room
        # a long file written after some thought gets cut off.
        max_tokens = self.max_output_tokens + (THINKING_TOKENS if think else 0)
        result = self._backend.chat(
            self.current.name, messages, self.current.num_ctx, schema, max_tokens, think, temperature,
        )
        self.completion_tokens += result.completion_tokens
        self.eval_seconds += result.eval_seconds
        self.last_prompt_tokens = result.prompt_tokens
        return result

    def structured(
        self,
        messages: list[Message],
        model: type[T],
        schema: dict | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> T:
        """Completion parsed into `model`, re-asking when the output is malformed."""
        schema = schema or model.model_json_schema()
        messages = list(messages)
        for _ in range(self._format_retries + 1):
            result = self.chat(messages, schema, temperature, think)
            try:
                return model.model_validate_json(result.content)
            except ValidationError as exc:
                if result.truncated:
                    # Repeating the same request would only be cut off again.
                    messages += [
                        {"role": "assistant", "content": result.content[:TRUNCATED_ECHO_CHARS] + " ..."},
                        {"role": "user", "content": (
                            f"Your reply was cut off at the limit of {self.max_output_tokens} tokens, so its "
                            "JSON is incomplete. Answer again with a much shorter reply: write a smaller "
                            "part now and add the rest in later steps."
                        )},
                    ]
                    continue
                problem = "; ".join(
                    f"{'.'.join(map(str, e['loc'])) or 'reply'}: {e['msg']}" for e in exc.errors()[:3]
                )
                messages += [
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": f"Invalid reply ({problem}). Answer again with valid JSON only."},
                ]
        raise LLMFormatError(f"{self.current.name} did not produce valid output")

    def unload(self) -> None:
        if self.current.name in {m.name for m in self._backend.loaded()}:
            self._backend.unload(self.current.name)

    def _need(self, choice: ModelChoice) -> tuple[int, int]:
        """(RAM, VRAM) in MB that loading `choice` will take."""
        if choice.placement == "gpu":
            return 0, choice.est_mb
        if choice.placement == "cpu":
            return choice.est_mb, 0
        # Split: Ollama fills the GPU first, the remainder goes to RAM.
        return max(0, choice.est_mb - (self._governor.budget().vram_mb or 0)), 0

    def _unload_others(self) -> bool:
        others = [m.name for m in self._backend.loaded() if m.name != self.current.name]
        for name in others:
            self._backend.unload(name)
        return bool(others)

    def _downgrade(self) -> bool:
        """Switch to the next smaller model in the chain, if any."""
        smaller = [c for c in self._chain if c.est_mb < self.current.est_mb]
        if not smaller:
            return False
        self.current = max(smaller, key=lambda c: c.est_mb)
        return True

    def _admit(self) -> None:
        reloaded = False
        while True:
            if self.current.name in {m.name for m in self._backend.loaded()}:
                try:
                    self._governor.admit("LLM call")
                    return
                except MemoryPressureError:
                    previous = self.current
                    if not reloaded:
                        # The model runner's own caches grow in RAM during a session and
                        # are not reported by the server; reloading the model drops them.
                        reloaded = True
                        self._backend.unload(previous.name)
                        continue
                    # A model fully on the GPU holds no RAM worth freeing.
                    if previous.placement == "gpu" or not self._downgrade():
                        raise
                    self._backend.unload(previous.name)
                    continue
            ram_mb, vram_mb = self._need(self.current)
            try:
                self._governor.admit(
                    f"load {self.current.name}", ram_mb, vram_mb, relievers=[self._unload_others]
                )
                return
            except MemoryPressureError:
                if not self._downgrade():
                    raise
