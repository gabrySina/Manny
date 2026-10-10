"""Data models describing agents and their work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    """An agent as defined at runtime: who it is and which tools it may use."""

    name: str
    role: str
    system_prompt: str
    tools: list[str]


class Action(BaseModel):
    """One step of an agent: a tool call."""

    thought: str = ""
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class FinishArgs(BaseModel):
    success: bool
    summary: str


@dataclass(frozen=True)
class AgentResult:
    success: bool
    summary: str
    steps: int


class TaskNode(BaseModel):
    """One unit of work in a plan, assigned to an agent."""

    id: str
    agent: str
    goal: str
    depends_on: list[str] = Field(default_factory=list)
    # The task only counts as done when the automatic check for the plan's kind passes.
    gated: bool = False


class Plan(BaseModel):
    """What the orchestrator decides at runtime: which agents exist and what each does."""

    # Decides how the result is verified: pytest, static page check, grounded report, nothing.
    kind: Literal["python", "web", "research", "other"] = "python"
    # What the finished work must satisfy; given to every agent and to the reviewer.
    criteria: list[str] = Field(default_factory=list)
    agents: list[AgentSpec]
    tasks: list[TaskNode]


def ordered(tasks: list[TaskNode]) -> list[TaskNode]:
    """Tasks in dependency order (stable). Raises ValueError on a dependency cycle."""
    done: list[TaskNode] = []
    names: set[str] = set()
    pending = list(tasks)
    while pending:
        ready = [t for t in pending if set(t.depends_on) <= names]
        if not ready:
            raise ValueError("tasks depend on each other in a cycle")
        done.append(ready[0])
        names.add(ready[0].id)
        pending.remove(ready[0])
    return done


def plan_problems(plan: Plan, tool_names: set[str], max_agents: int, max_tasks: int) -> list[str]:
    """Everything that makes a plan unusable, as messages the planner can act on."""
    problems = []
    if not plan.agents or not plan.tasks:
        problems.append("the plan needs at least one agent and one task")
    if len(plan.agents) > max_agents:
        problems.append(f"too many agents: at most {max_agents}")
    if len(plan.tasks) > max_tasks:
        problems.append(f"too many tasks: at most {max_tasks}")
    agent_names = [a.name for a in plan.agents]
    task_ids = [t.id for t in plan.tasks]
    if len(set(agent_names)) != len(agent_names):
        problems.append("agent names must be unique")
    if len(set(task_ids)) != len(task_ids):
        problems.append("task ids must be unique")
    for agent in plan.agents:
        if not tool_names.intersection(agent.tools):
            problems.append(f"agent '{agent.name}' has no valid tool; choose from {sorted(tool_names)}")
    for task in plan.tasks:
        if task.agent not in agent_names:
            problems.append(f"task '{task.id}' uses unknown agent '{task.agent}'")
        for dep in task.depends_on:
            if dep not in task_ids:
                problems.append(f"task '{task.id}' depends on unknown task '{dep}'")
    if not problems:
        try:
            ordered(plan.tasks)
        except ValueError as exc:
            problems.append(str(exc))
    return problems
