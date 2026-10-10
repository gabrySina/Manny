"""The orchestrator: designs a team of agents for the task at runtime and runs their work."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from manny.agents.agent import CHARS_PER_TOKEN, Agent, Verifier
from manny.agents.blackboard import Blackboard, TaskRecord
from manny.agents.review import confirm_fixed, review_work
from manny.agents.spec import AgentSpec, Plan, TaskNode, ordered, plan_problems
from manny.config import AgentConfig
from manny.llm.base import LLMFormatError, Message
from manny.llm.manager import ModelManager
from manny.tools.base import Tool
from manny.tools.fs import NO_FILES

# (max agents, max tasks) per hardware tier: small models lose track of large plans.
TIER_LIMITS = {
    "cpu-low": (1, 2),
    "cpu": (2, 3),
    "gpu-4": (2, 3),
    "gpu-6": (2, 3),
    "gpu-8": (3, 5),
    "gpu-12+": (5, 8),
}
PLAN_ATTEMPTS = 3
# Every agent can look around; agents that write code can also edit and run it.
READ_TOOLS = ("list_files", "read_file")
WRITE_TOOLS = ("write_file", "replace_in_file")
WRITER_COMPANIONS = ("replace_in_file", "run_command", "run_tests", "install_packages")
# Static pages are judged by the page checker; Python test tools only lead agents astray.
NOT_FOR_WEB = ("run_tests", "install_packages", "run_command")
MAX_REPLANS = 1

# Room left in the reviewer's context for its instructions and the task description.
REVIEW_PROMPT_CHARS = 4000

PLANNER = """\
You are the orchestrator of a team of AI agents. Design the team for the user's task:
decide which agents are needed, what each one is (its system prompt) and which tools it
may use, then split the work into tasks.

Available tools:
{tools}

Reply ONLY with a JSON object of this shape:
{{"kind": "python",
 "criteria": ["...", "..."],
 "agents": [{{"name": "coder", "role": "writes the code", "system_prompt": "You are ...", "tools": ["write_file", "run_tests"]}}],
 "tasks": [{{"id": "t1", "agent": "coder", "goal": "...", "depends_on": [], "gated": true}}]}}

kind says how the result is verified:
- "python": Python code, verified by running pytest. Tests must be plain pytest files.
- "web": HTML/CSS/JavaScript pages, verified by a static checker (linked files exist, real
  content, real styling). One agent writes the whole page, inventing realistic content for
  the subject itself: no research, no tests, no local server.
- "research": a question answered from the web. The deliverable is the file report.md,
  with the source URL next to each claim; only URLs returned by web_search or fetch_url
  may be cited. The researcher needs web_search, fetch_url and write_file.
- "other": anything else.

criteria: 3 to 7 concrete, checkable acceptance criteria. Include what a careful
professional would deliver even if the user did not spell it out: handling of bad input,
security basics (for example hashed passwords), complete real content instead of
placeholders, in the language the user wrote the task in. Criteria describe the finished
result, never the process or the tools used to get there, and must be achievable with the
available tools (for example, no photographs: agents cannot download images).

Rules:
- The default is ONE agent doing ONE task: it keeps the whole job in a single mind. Add a
  second agent only for work of a truly different nature, such as research before coding.
  Never more than {max_agents} agents and {max_tasks} tasks.
- Add a research agent only when the task depends on information you do not already have,
  such as an unfamiliar library or API. Ordinary programming needs no research.
- Give each agent only the tools its role needs.
- Each goal must be self-contained and concrete: agents do not see this conversation.
- Set gated to true on the tasks that must leave the automatic verification passing.
- depends_on lists the ids of tasks that must finish first."""

# What every agent must know about how its kind of work is produced and judged.
KIND_NOTES = {
    "python": (
        "When you call finish, pytest runs on the workspace and must pass. Write plain pytest "
        "tests. Install third-party packages with install_packages and list them in requirements.txt. "
        "Test web applications in-process with the framework's test client (for Flask, "
        "app.test_client()): never start a server or send HTTP requests to localhost."
    ),
    "web": (
        "This is a static web page, checked automatically when you call finish: every linked "
        "file must exist in the workspace, the text must be real and substantial (no "
        "placeholders), and the CSS must be a real layout. You cannot download images or "
        "fonts: make visuals with CSS, inline SVG or emoji. Do not write tests. Put the CSS "
        "in its own file and write one file per step, so that no single reply gets too long."
    ),
    "research": (
        "Write the answer to report.md in the language of the task. Read at least two "
        "different sources with fetch_url, put the source URL next to each claim, and cite "
        "only URLs returned by web_search or fetch_url. List each item once."
    ),
}

FIX_GOAL = """\
Make this change to the files in the workspace:

{problem}

The change has NOT been made yet. Checks that pass right now do not mean you are done:
they pass on the old code. Edit the files with write_file or replace_in_file and update
any test that relied on the old behaviour. Change only what this needs and keep
everything else working. When the change is in place and the checks pass, call finish.

Background, the project is: {task}"""

UNCHANGED = (
    "You have not changed any file, so the requested change is still missing. "
    "Edit the files with write_file or replace_in_file before calling finish."
)

# The model must always decide kind and criteria, although Plan has defaults for them.
PLAN_SCHEMA = Plan.model_json_schema()
PLAN_SCHEMA["required"] = ["kind", "criteria", "agents", "tasks"]

EventSink = Callable[[str, str, Any], None]


@dataclass(frozen=True)
class RunResult:
    success: bool
    records: list[TaskRecord]
    kind: str = "python"
    # Review findings still open when the run ended.
    issues: tuple[str, ...] = ()
    # The run failed and the workspace was put back as it was at the start.
    restored: bool = False


class Orchestrator:
    def __init__(
        self,
        llm: ModelManager,
        tools: dict[str, Tool],
        cfg: AgentConfig,
        tier: str,
        list_files: Callable[[], str],
        verifiers: dict[str, Verifier] | None = None,
        snapshot: Callable[[int], str] | None = None,
        checkpoint: Callable[[], Callable[[], None]] | None = None,
        fingerprint: Callable[[], str] | None = None,
        on_event: EventSink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._deadline = float("inf")
        self._llm = llm
        self._tools = tools
        self._cfg = cfg
        self._max_agents, self._max_tasks = TIER_LIMITS[tier]
        self._list_files = list_files
        # Automatic check per kind of work; kinds without one are judged by the review only.
        self._verifiers = verifiers or {}
        # Text of the workspace files for the reviewer; without it there is no review.
        self._snapshot = snapshot
        # Saves the workspace and returns the function that restores it.
        self._checkpoint = checkpoint
        # Changes whenever a workspace file changes: tells a real fix from an empty one.
        self._fingerprint = fingerprint
        self._fixes = 0
        self._on_event = on_event or (lambda source, kind, data: None)

    def run(self, task: str) -> RunResult:
        board = Blackboard()
        self._deadline = self._clock() + self._cfg.run_minutes * 60
        plan = self._plan(task, board)
        restore = self._working_baseline(plan)
        result = self._execute(task, plan, board)
        if not result.success and restore is not None:
            # Never leave a project that worked in a broken state.
            restore()
            self._on_event("orchestrator", "restored", None)
            return RunResult(False, result.records, result.kind, result.issues, restored=True)
        return result

    def _working_baseline(self, plan: Plan) -> Callable[[], None] | None:
        """A way back to the starting files, when they already pass the checks."""
        verify = self._verifiers.get(plan.kind)
        if self._checkpoint is None or verify is None or self._list_files() == NO_FILES:
            return None
        return self._checkpoint() if verify()[0] else None

    def _execute(self, task: str, plan: Plan, board: Blackboard) -> RunResult:
        replans = 0
        queue = ordered(plan.tasks)
        while queue:
            if self._out_of_time():
                break
            node = queue.pop(0)
            spec = next(a for a in plan.agents if a.name == node.agent)
            if not self._work(spec, node, plan, board):
                if not (node.gated or self._writes(spec)):
                    # Advisory work such as research: the next agents see that it failed.
                    continue
                if replans >= MAX_REPLANS or self._out_of_time():
                    return RunResult(False, board.records, plan.kind)
                replans += 1
                plan = self._plan(task, board)
                queue = ordered(plan.tasks)
        return self._finish(task, plan, board)

    def _out_of_time(self) -> bool:
        """Past the run's time limit: no new work starts, only the final check."""
        return self._clock() > self._deadline

    def _work(
        self,
        spec: AgentSpec,
        node: TaskNode,
        plan: Plan,
        board: Blackboard,
        verify: Verifier | None = None,
        deliberate: bool = False,
    ) -> bool:
        if verify is None and node.gated:
            verify = self._verifiers.get(plan.kind)
        agent = Agent(spec, self._llm, self._tools, self._cfg, verify, self._on_event, self._clock, deliberate)
        self._on_event("orchestrator", "task_start", node)
        context = board.context(self._list_files(), plan.criteria)
        if plan.kind in KIND_NOTES:
            context = f"{KIND_NOTES[plan.kind]}\n\n{context}"
        result = agent.run(node.goal, context)
        board.record(node, result)
        self._on_event("orchestrator", "task_end", result)
        return result.success

    def _finish(self, task: str, plan: Plan, board: Blackboard) -> RunResult:
        """Check the finished work, then review it and fix what the review finds."""
        rounds = self._cfg.max_review_rounds
        passed, output = self._check(plan, board)
        if not passed and rounds and not self._out_of_time():
            self._fix(task, plan, board, f"The automatic check fails:\n{output}")
            passed, output = self._check(plan, board)
        open_issues: list[str] = []
        for _ in range(rounds):
            if not passed or self._snapshot is None or self._out_of_time():
                break
            issues = self._review(task, plan, output)
            open_issues = []
            # One finding per fixer, most severe first: a small model handed the whole
            # list tends to fix none of it.
            for position, issue in enumerate(issues):
                if not passed or position >= self._cfg.max_fix_issues or self._out_of_time():
                    open_issues.append(issue)
                    continue
                fixed, passed, output = self._fix_issue(task, plan, board, issue, output)
                if not fixed:
                    open_issues.append(issue)
            if not open_issues:
                break
        return RunResult(passed, board.records, plan.kind, tuple(open_issues))

    def _fix(self, task: str, plan: Plan, board: Blackboard, problem: str) -> None:
        fixer = AgentSpec(
            name="fixer",
            role="fixes one problem found by checks or review",
            system_prompt=(
                "You are a senior developer fixing one problem found in review. Read the "
                "relevant files first, change only what is needed and verify the result."
            ),
            tools=list(self._tools),
        )
        self._fixes += 1
        goal = FIX_GOAL.format(task=task, problem=problem)
        node = TaskNode(id=f"fix{self._fixes}", agent=fixer.name, goal=goal, gated=True)
        self._work(fixer, node, plan, board, self._fix_verifier(plan), deliberate=True)

    def _fix_verifier(self, plan: Plan) -> Verifier:
        """The usual check, refused outright while the fixer has not touched any file."""
        check = self._verifiers.get(plan.kind)
        before = self._fingerprint() if self._fingerprint is not None else None

        def verify() -> tuple[bool, str]:
            if before is not None and self._fingerprint() == before:
                return False, UNCHANGED
            return check() if check is not None else (True, "")

        return verify

    def _fix_issue(
        self, task: str, plan: Plan, board: Blackboard, issue: str, output: str
    ) -> tuple[bool, bool, str]:
        """Have one review finding fixed: (finding resolved, checks passing, check output)."""
        restore = self._checkpoint() if self._checkpoint is not None else None
        self._fix(task, plan, board, issue)
        passed, new_output = self._check(plan, board)
        if not passed:
            if restore is None:
                return False, False, new_output
            # A fix that breaks working code is worse than the finding it was after.
            restore()
            self._on_event("orchestrator", "fix_result", (issue, "reverted"))
            return False, True, output
        fixed = confirm_fixed(self._llm, issue, self._files_for_review())
        self._on_event("orchestrator", "fix_result", (issue, "fixed" if fixed else "open"))
        return fixed, True, new_output

    def _check(self, plan: Plan, board: Blackboard) -> tuple[bool, str]:
        """Evidence that the work is done: the automatic check for this kind, when there is one."""
        verify = self._verifiers.get(plan.kind)
        if verify is None:
            if not board.records:
                return False, "no task was completed in time"
            last = board.records[-1].result
            return last.success, "" if last.success else f"the last task failed: {last.summary}"
        passed, output = verify()
        self._on_event("orchestrator", "final_check", (passed, output))
        return passed, output

    def _files_for_review(self) -> str:
        room = (self._llm.current.num_ctx - self._llm.max_output_tokens) * CHARS_PER_TOKEN
        return self._snapshot(max(room - REVIEW_PROMPT_CHARS, 2000))

    def _review(self, task: str, plan: Plan, check_output: str) -> list[str]:
        review = review_work(self._llm, task, plan.kind, plan.criteria, self._files_for_review(), check_output)
        issues = [] if review is None else review.issues
        self._on_event("orchestrator", "review", issues)
        return issues

    @staticmethod
    def _writes(spec: AgentSpec) -> bool:
        return bool(set(WRITE_TOOLS).intersection(spec.tools))

    def _plan(self, task: str, board: Blackboard) -> Plan:
        described = "\n".join(tool.describe() for tool in self._tools.values())
        system = PLANNER.format(tools=described, max_agents=self._max_agents, max_tasks=self._max_tasks)
        request = f"Task: {task}"
        if board.records:
            request += (
                f"\n\nA previous plan was started and a task failed:\n{board.progress()}\n\n"
                f"Files currently in the workspace:\n{self._list_files()}\n\n"
                "Plan only the work that is still needed, taking the failure into account."
            )
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": request},
        ]
        plan = None
        for _ in range(PLAN_ATTEMPTS):
            try:
                plan = self._llm.structured(messages, Plan, PLAN_SCHEMA)
            except LLMFormatError:
                plan = None
                break
            problems = plan_problems(plan, set(self._tools), self._max_agents, self._max_tasks)
            if not problems:
                break
            messages += [
                {"role": "assistant", "content": plan.model_dump_json()},
                {"role": "user", "content": "The plan is not valid:\n- " + "\n- ".join(problems) + "\nSend a corrected plan."},
            ]
            plan = None
        if plan is None:
            plan = self._fallback_plan(task)
        for agent in plan.agents:
            wanted = [*agent.tools, *READ_TOOLS]
            if "web_search" in agent.tools:
                # Snippets are not enough to answer from, and research ends in a written report.
                wanted += ["fetch_url"] + (["write_file"] if plan.kind == "research" else [])
            if set(WRITE_TOOLS).intersection(agent.tools):
                wanted += WRITER_COMPANIONS
            if plan.kind == "web":
                wanted = [name for name in wanted if name not in NOT_FOR_WEB]
            agent.tools = [name for name in dict.fromkeys(wanted) if name in self._tools]
        self._on_event("orchestrator", "plan", plan)
        return plan

    def _fallback_plan(self, task: str) -> Plan:
        """One generalist doing everything, used when the model cannot produce a valid plan."""
        agent = AgentSpec(
            name="developer",
            role="generalist",
            system_prompt="You are a careful software developer. Work step by step and test what you write.",
            tools=list(self._tools),
        )
        node = TaskNode(id="t1", agent=agent.name, goal=task, gated="python" in self._verifiers)
        return Plan(agents=[agent], tasks=[node])
