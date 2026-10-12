"""Record of a run, kept on disk so that it can be examined afterwards."""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

TRANSCRIPT = "transcript.txt"
EVENTS = "events.jsonl"
RESULT = "result.json"
# Long tool outputs are cut in the record; the files themselves stay in the workspace.
MAX_TEXT_CHARS = 20000


def runs_dir() -> Path:
    base = os.environ.get("MANNY_STATE_DIR")
    root = Path(base) if base else Path.home() / ".local" / "state" / "manny"
    return root / "runs"


def _plain(data: Any) -> Any:
    """`data` as something JSON can hold."""
    if isinstance(data, BaseModel):
        return data.model_dump()
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        return dataclasses.asdict(data)
    if isinstance(data, (list, tuple)):
        return [_plain(item) for item in data]
    if isinstance(data, str):
        return data[:MAX_TEXT_CHARS]
    if data is None or isinstance(data, (bool, int, float)):
        return data
    return str(data)[:MAX_TEXT_CHARS]


class SessionLog:
    """One folder per run: what was shown on screen, every event in full, and the outcome."""

    def __init__(self, task: str, workspace: Path) -> None:
        started = datetime.now()
        self.dir = runs_dir() / started.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta = {"task": task, "workspace": str(workspace), "started": started.isoformat(timespec="seconds")}
        self.transcript = open(self.dir / TRANSCRIPT, "w", encoding="utf-8")
        self._events = open(self.dir / EVENTS, "w", encoding="utf-8")
        # Written at once, so that an interrupted run still says what it was doing.
        self._write_result({"outcome": "interrupted"})

    def event(self, source: str, kind: str, data: Any) -> None:
        record = {"time": datetime.now().isoformat(timespec="seconds"), "source": source, "kind": kind, "data": _plain(data)}
        self._events.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._events.flush()

    def finish(self, outcome: dict) -> None:
        self._write_result({**outcome, "finished": datetime.now().isoformat(timespec="seconds")})
        self.transcript.close()
        self._events.close()

    def _write_result(self, outcome: dict) -> None:
        (self.dir / RESULT).write_text(
            json.dumps({**self._meta, **outcome}, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def recent(limit: int = 10) -> list[Path]:
    """Folders of the latest runs, newest first."""
    root = runs_dir()
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if (p / RESULT).is_file()), reverse=True)[:limit]
