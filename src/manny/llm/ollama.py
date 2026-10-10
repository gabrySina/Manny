"""Client for the local Ollama server."""

from __future__ import annotations

import re
import time

import httpx

from manny.llm.base import ChatResult, LLMBackendError, LoadedModel, Message
from manny.llm.catalog import ModelEntry, entry_from_metadata

MIB = 1024 * 1024
# First release whose `format` parameter accepts a JSON schema.
MIN_STRUCTURED_VERSION = (0, 5, 0)
# A crashed model runner is restarted by the server on the next request.
CHAT_ATTEMPTS = 2
DEFAULT_TEMPERATURE = 0.2
RETRY_PAUSE = 3.0


def parse_version(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split("."):
        digits = re.match(r"\d+", piece)
        if not digits:
            break
        parts.append(int(digits.group()))
    return tuple(parts)


class OllamaClient:
    def __init__(self, host: str, timeout: float = 600.0, keep_alive: str = "5m") -> None:
        self._http = httpx.Client(base_url=host, timeout=timeout)
        self._keep_alive = keep_alive
        self._supports_schema: bool | None = None

    def version(self) -> tuple[int, ...] | None:
        """Server version, or None when the server is unreachable."""
        try:
            resp = self._http.get("/api/version", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        return parse_version(resp.json().get("version", ""))

    def installed(self) -> set[str]:
        resp = self._http.get("/api/tags", timeout=10.0)
        resp.raise_for_status()
        return {m["name"] for m in resp.json().get("models", [])}

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
        options: dict = {"num_ctx": num_ctx, "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": self._keep_alive,
        }
        if think is not None:
            body["think"] = think
        if schema is not None:
            if self._supports_schema is None:
                self._supports_schema = (self.version() or ()) >= MIN_STRUCTURED_VERSION
            # Older servers only know plain JSON mode; the caller validates the shape.
            body["format"] = schema if self._supports_schema else "json"
        for attempt in range(1, CHAT_ATTEMPTS + 1):
            try:
                resp = self._http.post("/api/chat", json=body)
                resp.raise_for_status()
                break
            except httpx.HTTPError as exc:
                if attempt == CHAT_ATTEMPTS:
                    raise LLMBackendError(f"Ollama failed on {model}: {exc}") from exc
                time.sleep(RETRY_PAUSE)
        data = resp.json()
        return ChatResult(
            data["message"]["content"],
            data.get("prompt_eval_count", 0),
            data.get("eval_count", 0),
            data.get("eval_duration", 0) / 1e9,
            data.get("done_reason") == "length",
        )

    def describe(self, model: str) -> ModelEntry:
        """Catalog entry for an installed model, built from the server's metadata."""
        tags = self._http.get("/api/tags", timeout=10.0)
        tags.raise_for_status()
        size = next((m["size"] for m in tags.json().get("models", []) if m["name"] == model), None)
        if size is None:
            raise LLMBackendError(f"model '{model}' is not installed in Ollama")
        show = self._http.post("/api/show", json={"model": model}, timeout=30.0)
        show.raise_for_status()
        data = show.json()
        return entry_from_metadata(model, size, data.get("model_info", {}), data.get("capabilities", []))

    def loaded(self) -> list[LoadedModel]:
        resp = self._http.get("/api/ps", timeout=10.0)
        resp.raise_for_status()
        return [
            LoadedModel(m["name"], m.get("size", 0) // MIB, m.get("size_vram", 0) // MIB)
            for m in resp.json().get("models", [])
        ]

    def unload(self, model: str) -> None:
        resp = self._http.post("/api/generate", json={"model": model, "keep_alive": 0}, timeout=30.0)
        resp.raise_for_status()
