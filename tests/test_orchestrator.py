import json

import pytest
from fakes import FakeBackend, action, choice
from test_governor import FakeMemory, make_governor

from manny.agents.orchestrator import Orchestrator
from manny.agents.spec import AgentSpec, Plan, TaskNode, ordered, plan_problems
from manny.config import AgentConfig
from manny.llm.manager import ModelManager
from manny.tools.fs import ListArgs, Workspace, fs_tools

TOOLS = {"list_files", "read_file", "write_file", "replace_in_file"}


def agent(name, tools=("write_file",)):
    return {"name": name, "role": name, "system_prompt": f"You are the {name}.", "tools": list(tools)}


def task(id, agent, deps=(), tests=False):
    return {"id": id, "agent": agent, "goal": f"do {id}", "depends_on": list(deps), "gated": tests}


def plan(agents, tasks):
    return json.dumps({"agents": agents, "tasks": tasks})


def make(tmp_path, replies, verify=None, tier="gpu-6", rounds=0, verifiers=None, **agent_cfg):
    backend = FakeBackend(replies)
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [choice("m", 3000)])
    ws = Workspace(tmp_path / "ws")
    tools = {t.name: t for t in fs_tools(ws)}
    events = []
    if verify is not None:
        verifiers = {"python": verify}
    orchestrator = Orchestrator(
        manager, tools, AgentConfig(max_review_rounds=rounds, **agent_cfg), tier,
        list_files=lambda: tools["list_files"].run(ListArgs()),
        verifiers=verifiers, snapshot=ws.snapshot, checkpoint=ws.checkpoint,
        fingerprint=ws.fingerprint, on_event=lambda *e: events.append(e),
    )
    return orchestrator, backend, ws, events


def test_agents_are_created_from_the_plan_and_run_in_dependency_order(tmp_path):
    replies = [
        plan([agent("researcher"), agent("coder")], [task("t2", "coder", deps=["t1"]), task("t1", "researcher")]),
        action("write_file", path="notes.md", content="use configparser"),
        action("finish", success=True, summary="notes written to notes.md"),
        action("finish", success=True, summary="code written"),
    ]
    orchestrator, backend, _, events = make(tmp_path, replies)
    result = orchestrator.run("build it")
    assert result.success and [r.task.id for r in result.records] == ["t1", "t2"]
    # Each agent gets the system prompt the plan invented for it.
    assert backend.calls[1]["messages"][0]["content"].startswith("You are the researcher.")
    coder_start = backend.calls[3]["messages"]
    assert coder_start[0]["content"].startswith("You are the coder.")
    # The second agent sees the first one's result and the files, not its conversation.
    assert "notes written to notes.md" in coder_start[1]["content"]
    assert "notes.md" in coder_start[1]["content"] and len(coder_start) == 2
    assert [kind for _, kind, _ in events if kind in ("plan", "task_start", "task_end")] == [
        "plan", "task_start", "task_end", "task_start", "task_end",
    ]


def test_planned_agents_get_the_basic_tools_they_were_not_given(tmp_path):
    replies = [
        plan([agent("coder", ["write_file", "fly"]), agent("reader", ["read_file"])], [task("t1", "coder"), task("t2", "reader")]),
        action("finish", success=True, summary="a"),
        action("finish", success=True, summary="b"),
    ]
    orchestrator, _, _, events = make(tmp_path, replies)
    assert orchestrator.run("x").success
    coder, reader = next(data for _, kind, data in events if kind == "plan").agents
    # run_command and run_tests are not available in this test setup, only file tools.
    assert coder.tools == ["write_file", "list_files", "read_file", "replace_in_file"]
    assert reader.tools == ["read_file", "list_files"]


def test_invalid_plan_is_sent_back_with_the_problems(tmp_path):
    replies = [
        plan([agent("coder")], [task("t1", "ghost")]),
        plan([agent("coder")], [task("t1", "coder")]),
        action("finish", success=True, summary="ok"),
    ]
    orchestrator, backend, _, _ = make(tmp_path, replies)
    assert orchestrator.run("x").success
    assert "unknown agent 'ghost'" in backend.calls[1]["messages"][-1]["content"]


def test_unusable_planner_output_falls_back_to_one_generalist(tmp_path):
    replies = ["garbage"] * 3 + [action("finish", success=True, summary="ok")]
    orchestrator, _, _, events = make(tmp_path, replies, verify=lambda: (True, "passed"))
    assert orchestrator.run("x").success
    fallback = next(data for _, kind, data in events if kind == "plan")
    assert [a.name for a in fallback.agents] == ["developer"]
    assert set(fallback.agents[0].tools) == TOOLS and fallback.tasks[0].gated


def test_tests_gate_only_the_tasks_that_ask_for_it(tmp_path):
    checks = []
    replies = [
        plan([agent("coder")], [task("t1", "coder"), task("t2", "coder", deps=["t1"], tests=True)]),
        action("finish", success=True, summary="a"),
        action("finish", success=True, summary="b"),
    ]
    orchestrator, _, _, _ = make(tmp_path, replies, verify=lambda: checks.append(1) or (True, "passed"))
    assert orchestrator.run("x").success
    # Once when t2 finishes, once as the final check of the whole run.
    assert len(checks) == 2


def test_failed_research_does_not_stop_the_coder(tmp_path):
    replies = [
        plan([agent("researcher", ["read_file"]), agent("coder")], [task("t1", "researcher"), task("t2", "coder", deps=["t1"])]),
        action("finish", success=False, summary="the web was unreachable"),
        action("write_file", path="a.py", content="x = 1"),
        action("finish", success=True, summary="written from my own knowledge"),
    ]
    orchestrator, backend, _, events = make(tmp_path, replies, verify=lambda: (True, "3 passed"))
    result = orchestrator.run("x")
    assert result.success and [r.result.success for r in result.records] == [False, True]
    assert "[FAILED] researcher: do t1 -> the web was unreachable" in backend.calls[2]["messages"][1]["content"]
    assert [kind for _, kind, _ in events].count("plan") == 1
    assert ("orchestrator", "final_check", (True, "3 passed")) in events


def test_final_check_decides_the_outcome_when_code_was_written(tmp_path):
    replies = [plan([agent("coder")], [task("t1", "coder")]), action("finish", success=True, summary="looks fine to me")]
    orchestrator, _, _, _ = make(tmp_path, replies, verify=lambda: (False, "1 failed"))
    assert not orchestrator.run("x").success


def test_failed_task_triggers_one_replan_with_the_history(tmp_path):
    replies = [
        plan([agent("coder")], [task("t1", "coder")]),
        action("finish", success=False, summary="library X does not exist"),
        plan([agent("fixer")], [task("t9", "fixer")]),
        action("finish", success=True, summary="used library Y"),
    ]
    orchestrator, backend, _, _ = make(tmp_path, replies)
    result = orchestrator.run("x")
    assert result.success and [r.result.success for r in result.records] == [False, True]
    assert "[FAILED] coder: do t1 -> library X does not exist" in backend.calls[2]["messages"][1]["content"]


def test_second_failure_ends_the_run(tmp_path):
    replies = [
        plan([agent("coder")], [task("t1", "coder")]),
        action("finish", success=False, summary="no"),
        plan([agent("coder")], [task("t1", "coder"), task("t2", "coder")]),
        action("finish", success=False, summary="still no"),
    ]
    orchestrator, backend, _, _ = make(tmp_path, replies)
    assert not orchestrator.run("x").success and backend.replies == []


def test_plan_limits_follow_the_hardware_tier(tmp_path):
    too_big = plan([agent(f"a{i}") for i in range(3)], [task("t1", "a0")])
    replies = [too_big, plan([agent("a0")], [task("t1", "a0")]), action("finish", success=True, summary="ok")]
    orchestrator, backend, _, _ = make(tmp_path, replies, tier="cpu-low")
    assert orchestrator.run("x").success
    assert "Never more than 1 agents and 2 tasks" in backend.calls[0]["messages"][0]["content"]
    assert "too many agents: at most 1" in backend.calls[1]["messages"][-1]["content"]


def spec(name, tools=("write_file",)):
    return AgentSpec(name=name, role="r", system_prompt="p", tools=list(tools))


def node(id, agent="a", deps=()):
    return TaskNode(id=id, agent=agent, goal="g", depends_on=list(deps))


def test_ordered_is_stable_and_detects_cycles():
    assert [t.id for t in ordered([node("c", deps=["a"]), node("a"), node("b")])] == ["a", "c", "b"]
    with pytest.raises(ValueError):
        ordered([node("a", deps=["b"]), node("b", deps=["a"])])


@pytest.mark.parametrize(
    ("bad_plan", "expected"),
    [
        (Plan(agents=[], tasks=[]), "at least one agent"),
        (Plan(agents=[spec("a"), spec("a")], tasks=[node("t")]), "agent names must be unique"),
        (Plan(agents=[spec("a")], tasks=[node("t"), node("t")]), "task ids must be unique"),
        (Plan(agents=[spec("a", ["fly"])], tasks=[node("t")]), "no valid tool"),
        (Plan(agents=[spec("a")], tasks=[node("t", deps=["zz"])]), "unknown task 'zz'"),
        (Plan(agents=[spec("a")], tasks=[node("x", deps=["y"]), node("y", deps=["x"])]), "cycle"),
        (Plan(agents=[spec("a")], tasks=[node(str(i)) for i in range(7)]), "too many tasks"),
    ],
)
def test_plan_problems(bad_plan, expected):
    assert any(expected in p for p in plan_problems(bad_plan, TOOLS, max_agents=4, max_tasks=6))


def review(*issues, severities=None):
    severities = severities or ["medium"] * len(issues)
    found = [
        {"file": "server.py", "problem": issue, "fix": "do it", "severity": severity}
        for issue, severity in zip(issues, severities)
    ]
    return json.dumps({"issues": found, "approved": not issues})


def confirm(fixed):
    return json.dumps({"evidence": "seen in server.py", "fixed": fixed})


def finding(problem):
    return f"server.py: {problem} Fix: do it"


CODED = [plan([agent("coder")], [task("t1", "coder")]), action("finish", success=True, summary="written")]
def fixed(n):
    """A fixer that changes a file (different content each time) and finishes."""
    return [action("write_file", path="server.py", content=f"version {n}"), action("finish", success=True, summary="fixed")]


def test_each_finding_gets_its_own_fixer_most_severe_first(tmp_path):
    replies = CODED + [
        review("no requirements.txt", "password stored in plain text", severities=["low", "high"]),
        action("write_file", path="server.py", content="hash_password(password)"),
        action("finish", success=True, summary="passwords are hashed now"),
        confirm(True),
        action("write_file", path="requirements.txt", content="flask"),
        action("finish", success=True, summary="requirements added"),
        confirm(True),
    ]
    orchestrator, backend, ws, events = make(tmp_path, replies, verify=lambda: (True, "2 passed"), rounds=1)
    result = orchestrator.run("login server")
    assert result.success and result.issues == () and backend.replies == []
    assert [r.task.id for r in result.records] == ["t1", "fix1", "fix2"]
    first_fixer, second_fixer = backend.calls[3]["messages"][1]["content"], backend.calls[6]["messages"][1]["content"]
    # The high-severity finding goes first, and each fixer sees only its own finding.
    assert "password stored in plain text" in first_fixer and "requirements.txt" not in first_fixer.split("Context:")[0]
    assert "no requirements.txt" in second_fixer and "plain text" not in second_fixer.split("Context:")[0]
    # The confirmation question is about that one finding and shows the current files.
    confirmation = backend.calls[5]["messages"][1]["content"]
    assert "password stored in plain text" in confirmation and "hash_password(password)" in confirmation
    assert [data[1] for _, kind, data in events if kind == "fix_result"] == ["fixed", "fixed"]


def test_unconfirmed_and_unattempted_findings_stay_open(tmp_path):
    replies = CODED + [review("a", "b", "c")] + fixed(1) + [confirm(False)] + fixed(2) + [confirm(True)]
    orchestrator, backend, _, _ = make(tmp_path, replies, verify=lambda: (True, "ok"), rounds=1, max_fix_issues=2)
    result = orchestrator.run("x")
    # "a" was attempted but not confirmed, "b" was fixed, "c" is beyond the limit.
    assert result.success and result.issues == (finding("a"), finding("c")) and backend.replies == []


def test_a_fix_that_breaks_the_checks_is_reverted(tmp_path):
    outcomes = [(True, "2 passed"), (False, "1 failed"), (False, "1 failed"), (False, "1 failed"), (False, "1 failed")]
    replies = [
        plan([agent("coder")], [task("t1", "coder")]),
        action("write_file", path="server.py", content="working code"),
        action("finish", success=True, summary="written"),
        review("weak hash"),
        action("write_file", path="server.py", content="broken code"),
        action("write_file", path="extra.py", content="junk"),
        action("finish", success=True, summary="rewritten"),
        action("finish", success=False, summary="cannot make the tests pass"),
    ]
    orchestrator, backend, ws, events = make(
        tmp_path, replies, verify=lambda: outcomes.pop(0), rounds=1, max_fix_attempts=0
    )
    result = orchestrator.run("x")
    # The run stays successful on the last working state, with the finding reported as open.
    assert result.success and result.issues == (finding("weak hash"),)
    assert (ws.root / "server.py").read_text() == "working code" and not (ws.root / "extra.py").exists()
    assert [data[1] for _, kind, data in events if kind == "fix_result"] == ["reverted"]


def test_failing_check_is_sent_to_the_fixer_before_any_review(tmp_path):
    outcomes = [(False, "1 failed: test_login"), (True, "2 passed"), (True, "2 passed")]
    replies = CODED + fixed(1) + [review()]
    orchestrator, backend, _, _ = make(tmp_path, replies, verify=lambda: outcomes.pop(0), rounds=1)
    result = orchestrator.run("x")
    assert result.success and outcomes == [] and backend.replies == []
    assert "1 failed: test_login" in backend.calls[2]["messages"][1]["content"]


def test_unreadable_review_does_not_block_the_run(tmp_path):
    replies = CODED + ["garbage"] * 3
    orchestrator, _, _, _ = make(tmp_path, replies, verify=lambda: (True, "ok"), rounds=2)
    result = orchestrator.run("x")
    assert result.success and result.issues == ()


def test_second_round_reviews_again_only_while_findings_remain(tmp_path):
    replies = CODED + [review("a")] + fixed(1) + [confirm(False), review("a again")] + fixed(2) + [confirm(True)]
    orchestrator, backend, _, _ = make(tmp_path, replies, verify=lambda: (True, "ok"), rounds=2)
    result = orchestrator.run("x")
    assert result.success and result.issues == () and backend.replies == []


def kinded(kind, agents, tasks, criteria=()):
    return json.dumps({"kind": kind, "criteria": list(criteria), "agents": agents, "tasks": tasks})


def test_each_kind_of_work_uses_its_own_check_and_criteria_reach_the_agents(tmp_path):
    used = []
    verifiers = {"python": lambda: used.append("pytest") or (True, ""), "web": lambda: used.append("web") or (True, "page ok")}
    replies = [
        kinded("web", [agent("designer")], [task("t1", "designer", tests=True)], ["testi in italiano"]),
        action("finish", success=True, summary="page written"),
    ]
    orchestrator, backend, _, _ = make(tmp_path, replies, verifiers=verifiers)
    result = orchestrator.run("landing page")
    assert result.success and result.kind == "web" and used == ["web", "web"]
    assert "- testi in italiano" in backend.calls[1]["messages"][1]["content"]
    assert "You cannot download images" in backend.calls[1]["messages"][1]["content"]


def test_kind_without_a_check_relies_on_the_agents_outcome(tmp_path):
    replies = [kinded("other", [agent("writer")], [task("t1", "writer")]), action("finish", success=True, summary="ok")]
    orchestrator, _, _, events = make(tmp_path, replies, verifiers={"python": lambda: (False, "must not run")})
    assert orchestrator.run("x").success
    assert not [e for e in events if e[1] == "final_check"]


def test_researchers_get_the_tools_to_read_pages_and_write_the_report(tmp_path):
    replies = [
        kinded("research", [agent("researcher", ["web_search"])], [task("t1", "researcher")]),
        action("finish", success=True, summary="ok"),
    ]
    orchestrator, _, _, events = make(tmp_path, replies)
    # Stand-ins: this setup has no web tools, and only their names matter here.
    orchestrator._tools["web_search"] = orchestrator._tools["fetch_url"] = orchestrator._tools["read_file"]
    orchestrator.run("x")
    researcher = next(data for _, kind, data in events if kind == "plan").agents[0]
    assert {"web_search", "fetch_url", "write_file", "read_file"} <= set(researcher.tools)


def test_planner_schema_forces_kind_and_criteria():
    from manny.agents.orchestrator import PLAN_SCHEMA

    assert {"kind", "criteria"} <= set(PLAN_SCHEMA["required"])
    assert PLAN_SCHEMA["properties"]["kind"]["enum"] == ["python", "web", "research", "other"]


def test_web_work_gets_no_python_test_tools(tmp_path):
    replies = [
        kinded("web", [agent("designer", ["write_file", "run_tests"])], [task("t1", "designer")]),
        action("finish", success=True, summary="ok"),
    ]
    orchestrator, _, _, events = make(tmp_path, replies)
    orchestrator._tools["run_tests"] = orchestrator._tools["run_command"] = orchestrator._tools["read_file"]
    orchestrator.run("x")
    designer = next(data for _, kind, data in events if kind == "plan").agents[0]
    assert designer.tools == ["write_file", "list_files", "read_file", "replace_in_file"]


def test_run_time_limit_stops_new_work_but_still_checks_the_result(tmp_path):
    now = {"t": 0.0}

    def on_event(source, kind, data):
        if kind == "task_end":
            now["t"] = 10_000.0  # the first task used up the whole run

    replies = [
        plan([agent("coder")], [task("t1", "coder"), task("t2", "coder", deps=["t1"])]),
        action("finish", success=True, summary="first part done"),
    ]
    backend = FakeBackend(replies)
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [choice("m", 3000)])
    ws = Workspace(tmp_path / "ws")
    tools = {t.name: t for t in fs_tools(ws)}
    checks = []
    orchestrator = Orchestrator(
        manager, tools, AgentConfig(max_review_rounds=1, run_minutes=30), "gpu-6",
        list_files=lambda: "", verifiers={"python": lambda: checks.append(1) or (False, "1 failed")},
        snapshot=ws.snapshot, on_event=on_event, clock=lambda: now["t"],
    )
    result = orchestrator.run("x")
    # t2 never starts and no fixer runs, yet the outcome comes from a real check.
    assert [r.task.id for r in result.records] == ["t1"] and backend.replies == []
    assert checks == [1] and not result.success


def test_a_failed_run_puts_back_a_workspace_that_was_working(tmp_path):
    outcomes = [(True, "5 passed")] + [(False, "3 failed")] * 20
    replies = [
        plan([agent("coder")], [task("t1", "coder", tests=True)]),
        action("write_file", path="app.py", content="rewritten and broken"),
        action("write_file", path="extra.py", content="junk"),
        action("finish", success=False, summary="cannot make it work"),
        plan([agent("coder")], [task("t1", "coder", tests=True)]),
        action("finish", success=False, summary="still broken"),
    ]
    orchestrator, backend, ws, events = make(tmp_path, replies, verify=lambda: outcomes.pop(0))
    (ws.root / "app.py").write_text("working server")
    result = orchestrator.run("improve it")
    assert not result.success and result.restored and backend.replies == []
    assert (ws.root / "app.py").read_text() == "working server" and not (ws.root / "extra.py").exists()
    assert ("orchestrator", "restored", None) in events


def test_no_baseline_for_an_empty_or_already_failing_workspace(tmp_path):
    checks = []
    replies = [plan([agent("coder")], [task("t1", "coder")]), action("finish", success=False, summary="no")] * 2
    orchestrator, _, ws, _ = make(tmp_path, replies, verify=lambda: checks.append(1) or (False, "fails"))
    result = orchestrator.run("x")
    # Empty workspace: the checks are not even run to look for a baseline.
    assert not result.restored and checks == []
    replies = [plan([agent("coder")], [task("t1", "coder")]), action("finish", success=False, summary="no")] * 2
    orchestrator, _, ws, _ = make(tmp_path / "second", replies, verify=lambda: (False, "fails"))
    (ws.root / "app.py").write_text("already broken")
    assert not orchestrator.run("x").restored


def test_a_fixer_that_changes_nothing_is_not_allowed_to_finish(tmp_path):
    replies = CODED + [
        review("weak hash"),
        action("finish", success=True, summary="the tests already pass"),
        action("write_file", path="server.py", content="salted hash"),
        action("finish", success=True, summary="now it is salted"),
        confirm(True),
    ]
    orchestrator, backend, ws, _ = make(tmp_path, replies, verify=lambda: (True, "19 passed"), rounds=1)
    result = orchestrator.run("x")
    assert result.success and result.issues == () and backend.replies == []
    refusal = backend.calls[4]["messages"][-1]["content"]
    assert "You have not changed any file" in refusal
    # The request itself leads with the change and warns that passing checks prove nothing.
    request = backend.calls[3]["messages"][1]["content"]
    assert request.startswith("Goal: Make this change to the files in the workspace:\n\nserver.py: weak hash")
    assert "do not mean you are done" in request


def test_fixers_think_while_ordinary_agents_do_not(tmp_path):
    replies = CODED + [review("weak hash")] + fixed(1) + [confirm(True)]
    backend = FakeBackend(replies)
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [choice("m", 3000, think=True)])
    ws = Workspace(tmp_path / "ws")
    tools = {t.name: t for t in fs_tools(ws)}
    orchestrator = Orchestrator(
        manager, tools, AgentConfig(max_review_rounds=1), "gpu-6", list_files=lambda: "",
        verifiers={"python": lambda: (True, "ok")}, snapshot=ws.snapshot,
        checkpoint=ws.checkpoint, fingerprint=ws.fingerprint,
    )
    assert orchestrator.run("x").success
    # plan, coder, review, fixer write, fixer finish, confirmation
    assert [call["think"] for call in backend.calls] == [True, False, True, True, True, True]
