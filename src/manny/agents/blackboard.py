"""Shared state between agents: outcomes of finished tasks, not whole conversations."""

from __future__ import annotations

from dataclasses import dataclass

from manny.agents.spec import AgentResult, TaskNode

SUMMARY_CHARS = 600


@dataclass(frozen=True)
class TaskRecord:
    task: TaskNode
    result: AgentResult


class Blackboard:
    def __init__(self) -> None:
        self.records: list[TaskRecord] = []

    def record(self, task: TaskNode, result: AgentResult) -> None:
        self.records.append(TaskRecord(task, result))

    def progress(self) -> str:
        """One line per finished task, for agents and for replanning."""
        lines = []
        for r in self.records:
            status = "done" if r.result.success else "FAILED"
            lines.append(f"- [{status}] {r.task.agent}: {r.task.goal} -> {r.result.summary[:SUMMARY_CHARS]}")
        return "\n".join(lines)

    def context(self, files: str, criteria: list[str] = ()) -> str:
        """What an agent starting a task needs to know about the work so far."""
        parts = []
        if criteria:
            listed = "\n".join(f"- {c}" for c in criteria)
            parts.append(f"Acceptance criteria for the whole job (your part must respect them):\n{listed}")
        if self.records:
            parts.append(f"Work done so far by the team:\n{self.progress()}")
        parts.append(f"Files currently in the workspace:\n{files}")
        return "\n\n".join(parts)
