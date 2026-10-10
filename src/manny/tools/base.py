"""Tool definition shared by all agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


class ToolError(Exception):
    """An expected failure, reported back to the agent as an observation."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[Any], str]
    # The output must reach the agent whole: the tool limits its own size.
    verbatim: bool = False

    def describe(self) -> str:
        """One-line signature for the system prompt."""
        params = []
        for name, field in self.args.model_fields.items():
            kind = getattr(field.annotation, "__name__", str(field.annotation))
            default = "" if field.default is PydanticUndefined else f" = {field.default!r}"
            params.append(f"{name}: {kind}{default}")
        return f"- {self.name}({', '.join(params)}): {self.description}"


def truncate_middle(text: str, limit: int) -> str:
    """Keep the start and the end of long output; the end usually holds the error."""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters omitted] ...\n{text[-tail:]}"
