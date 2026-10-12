from fakes import FakeBackend, action, choice
from test_governor import FakeMemory, make_governor

from manny.agents.agent import Agent
from manny.agents.spec import AgentSpec
from manny.config import AgentConfig
from manny.llm.manager import ModelManager
from manny.tools.fs import Workspace, fs_tools

SPEC = AgentSpec(
    name="coder",
    role="coder",
    system_prompt="You write code.",
    tools=["list_files", "read_file", "write_file", "replace_in_file"],
)


def make_agent(tmp_path, replies, verify=None, num_ctx=8192, spec=SPEC, **cfg):
    backend = FakeBackend(replies)
    manager = ModelManager(
        backend, make_governor(FakeMemory(ram=8000, vram=5700)), [choice("m", 3000, num_ctx=num_ctx)]
    )
    ws = Workspace(tmp_path / "ws")
    tools = {t.name: t for t in fs_tools(ws)}
    events = []
    agent = Agent(spec, manager, tools, AgentConfig(**cfg), verify, lambda *e: events.append(e))
    return agent, backend, ws, events


def last_observation(backend):
    return backend.calls[-1]["messages"][-1]["content"]


def test_writes_a_file_and_finishes(tmp_path):
    agent, backend, ws, events = make_agent(
        tmp_path,
        [action("write_file", path="a.py", content="x = 1\n"), action("finish", success=True, summary="done")],
    )
    result = agent.run("create a.py")
    assert (result.success, result.summary, result.steps) == (True, "done", 2)
    assert (ws.root / "a.py").read_text() == "x = 1\n"
    assert "wrote 6 characters" in last_observation(backend)
    assert [kind for _, kind, _ in events] == ["action", "observation", "action"]


def test_system_prompt_lists_only_the_agents_tools(tmp_path):
    spec = SPEC.model_copy(update={"tools": ["read_file", "not_a_tool"]})
    agent, backend, _, _ = make_agent(tmp_path, [action("finish", success=True, summary="x")], spec=spec)
    agent.run("goal")
    system = backend.calls[0]["messages"][0]["content"]
    assert "- read_file(path: str, start_line: int = 1)" in system and "write_file" not in system and "- finish(" in system


def test_errors_come_back_as_observations(tmp_path):
    replies = [
        action("write_file", path="a.py", content="x"),
        action("read_file", path="../../etc/passwd"),
        action("read_file"),
        action("finish", success=True),
        action("finish", success=False, summary="gave up"),
    ]
    spec = SPEC.model_copy(update={"tools": ["read_file"]})
    agent, backend, ws, _ = make_agent(tmp_path, replies, spec=spec)
    result = agent.run("goal")
    observations = [c["messages"][-1]["content"] for c in backend.calls[1:]]
    assert "unknown tool 'write_file'" in observations[0]
    assert "outside the workspace" in observations[1]
    assert "invalid arguments for read_file" in observations[2]
    assert "finish needs" in observations[3]
    assert not result.success and not (ws.root / "a.py").exists()


def test_failed_verification_sends_the_agent_back_to_work(tmp_path):
    outcomes = [(False, "1 failed: assert add(1, 2) == 3"), (True, "1 passed")]
    replies = [
        action("finish", success=True, summary="first try"),
        action("write_file", path="a.py", content="fixed"),
        action("finish", success=True, summary="second try"),
    ]
    agent, backend, _, _ = make_agent(tmp_path, replies, verify=lambda: outcomes.pop(0))
    result = agent.run("goal")
    assert (result.success, result.summary) == (True, "second try")
    assert "assert add(1, 2) == 3" in backend.calls[1]["messages"][-1]["content"]


def test_verification_attempts_are_limited(tmp_path):
    replies = [action("finish", success=True, summary="done")] * 3
    agent, _, _, _ = make_agent(tmp_path, replies, verify=lambda: (False, "boom"), max_fix_attempts=2)
    result = agent.run("goal")
    assert not result.success and result.steps == 3


def test_step_limit(tmp_path):
    replies = [action("list_files", path=p) for p in (".", "a", "b")]
    agent, _, _, _ = make_agent(tmp_path, replies, max_steps=3)
    result = agent.run("goal")
    assert not result.success and "step limit" in result.summary


def test_identical_consecutive_calls_are_refused_then_abort(tmp_path):
    replies = [action("read_file", path="missing.py")] * 3
    agent, backend, _, _ = make_agent(tmp_path, replies)
    result = agent.run("goal")
    assert not result.success and "stuck repeating" in result.summary and result.steps == 3
    reminder = backend.calls[2]["messages"][-1]["content"]
    assert "Do not call read_file again now" in reminder and reminder.endswith("Your goal: goal")


def test_rewriting_the_same_content_between_test_runs_is_detected(tmp_path):
    write = action("write_file", path="t.py", content="broken")
    replies = [write, action("list_files"), write, action("list_files"), write, action("list_files"), write]
    agent, backend, _, _ = make_agent(tmp_path, replies)
    result = agent.run("goal")
    assert not result.success and "stuck repeating the same failing write_file" in result.summary
    assert result.steps == 7
    assert "already contains exactly this content" in backend.calls[3]["messages"][-1]["content"]


def test_empty_write_is_flagged(tmp_path):
    replies = [action("write_file", path="a.py", content=""), action("finish", success=True, summary="x")]
    agent, backend, _, _ = make_agent(tmp_path, replies)
    agent.run("goal")
    assert "EMPTY file" in backend.calls[1]["messages"][-1]["content"]


def test_invalid_model_output_fails_the_agent(tmp_path):
    agent, _, _, _ = make_agent(tmp_path, ["garbage"] * 3)
    assert not agent.run("goal").success


def test_old_steps_are_dropped_when_context_is_full(tmp_path):
    big = "x" * 3000
    replies = [action("write_file", path=f"f{i}.txt", content=big) for i in range(6)]
    replies.append(action("finish", success=True, summary="done"))
    agent, backend, _, _ = make_agent(tmp_path, replies, num_ctx=3000)
    assert agent.run("goal").success
    final = backend.calls[-1]["messages"]
    # num_ctx 3000 leaves 2000 tokens (~6000 characters) of input.
    assert sum(len(m["content"]) for m in final) <= 6000 + 200
    assert final[0]["role"] == "system" and "oldest steps were dropped" in final[1]["content"]
    assert len(final) < 2 + 2 * 6


class Clock:
    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def test_agent_stops_at_the_task_time_limit(tmp_path):
    backend = FakeBackend([action("list_files", path=str(i)) for i in range(10)])
    manager = ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [choice("m", 3000)])
    tools = {t.name: t for t in fs_tools(Workspace(tmp_path / "ws"))}
    # Each look at the clock advances it by 100 seconds; the limit is 4 minutes.
    agent = Agent(SPEC, manager, tools, AgentConfig(task_minutes=4), clock=Clock(100))
    result = agent.run("goal")
    # Each step looks at the clock twice: once for the limit, once to size its reply.
    assert not result.success and "time limit (4 minutes)" in result.summary and result.steps == 1


def test_file_reads_reach_the_agent_whole_while_other_outputs_are_shortened(tmp_path):
    body = "\n".join(f"line {i:04d} of the file" for i in range(300))
    replies = [
        action("write_file", path="big.py", content=body),
        action("read_file", path="big.py"),
        action("list_files"),
        action("finish", success=True, summary="x"),
    ]
    agent, backend, ws, _ = make_agent(tmp_path, replies, tool_output_chars=500)
    for i in range(200):
        (ws.root / f"file_{i:03d}_with_a_long_name.txt").write_text("x")
    agent.run("goal")
    read, listed = backend.calls[2]["messages"][-1]["content"], backend.calls[3]["messages"][-1]["content"]
    assert "line 0000" in read and "line 0150" in read and "line 0299" in read and "omitted" not in read
    assert "omitted" in listed and len(listed) < 700


def test_repeated_calls_stay_out_of_the_history_and_raise_the_temperature(tmp_path):
    same = action("list_files")
    replies = [same, same, same, action("finish", success=True, summary="moved on")]
    agent, backend, _, _ = make_agent(tmp_path, replies)
    assert agent.run("goal").success
    temperatures = [call["temperature"] for call in backend.calls]
    assert temperatures == [None, None, 0.4, 0.8]
    final = backend.calls[-1]["messages"]
    # One executed call, one reminder: the third identical call left no trace.
    assert [m["role"] for m in final] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "Do not call list_files again now" in final[-1]["content"]


def thinking_agent(tmp_path, replies, think=True, deliberate=False, **cfg):
    backend = FakeBackend(replies)
    manager = ModelManager(
        backend, make_governor(FakeMemory(ram=8000, vram=5700)), [choice("m", 3000, think=think)]
    )
    tools = {t.name: t for t in fs_tools(Workspace(tmp_path / "ws"))}
    return Agent(SPEC, manager, tools, AgentConfig(**cfg), deliberate=deliberate), backend


STEPS = [
    action("read_file", path="missing.py"),
    action("write_file", path="a.py", content="x = 1"),
    action("list_files"),
    action("finish", success=True, summary="done"),
]


def test_auto_reasoning_stays_off_for_ordinary_work_even_after_an_error(tmp_path):
    agent, backend = thinking_agent(tmp_path, STEPS)
    agent.run("goal")
    assert [call["think"] for call in backend.calls] == [False] * 4


def test_auto_reasoning_switches_on_when_the_agent_repeats_itself(tmp_path):
    same = action("list_files")
    replies = [same, same, action("write_file", path="a.py", content="x"), action("finish", success=True, summary="x")]
    agent, backend = thinking_agent(tmp_path, replies)
    agent.run("goal")
    # The repeated call makes the next step think; once it moves on, reasoning is off again.
    assert [call["think"] for call in backend.calls] == [False, False, True, False]


def test_fixers_reason_at_every_step(tmp_path):
    agent, backend = thinking_agent(tmp_path, STEPS, deliberate=True)
    agent.run("goal")
    assert [call["think"] for call in backend.calls] == [True] * 4


def test_reasoning_can_be_forced_on_or_off(tmp_path):
    agent, backend = thinking_agent(tmp_path, STEPS, thinking="always")
    agent.run("goal")
    assert {call["think"] for call in backend.calls} == {True}
    agent, backend = thinking_agent(tmp_path / "b", STEPS, deliberate=True, thinking="never")
    agent.run("goal")
    assert {call["think"] for call in backend.calls} == {False}


def test_models_without_a_reasoning_mode_are_never_asked_to_think(tmp_path):
    agent, backend = thinking_agent(tmp_path, STEPS, think=None, deliberate=True)
    agent.run("goal")
    assert {call["think"] for call in backend.calls} == {None}


def test_alternating_between_two_look_around_calls_is_a_loop(tmp_path):
    look, read = action("list_files"), action("read_file", path="a.py")
    replies = [action("write_file", path="a.py", content="x = 1"), look, read, look, read, look, read]
    agent, backend, _, _ = make_agent(tmp_path, replies)
    result = agent.run("goal")
    # First look and read are fine; the second of each is refused, the third ends the task.
    assert not result.success and "stuck repeating" in result.summary and result.steps == 6
    assert "you already made this exact list_files call" in backend.calls[4]["messages"][-1]["content"]


def test_reading_again_after_a_change_is_not_a_loop(tmp_path):
    read, tests = action("read_file", path="a.py"), action("list_files")
    replies = [
        action("write_file", path="a.py", content="x = 1"), read, tests,
        action("write_file", path="a.py", content="x = 2"), read, tests,
        action("finish", success=True, summary="done"),
    ]
    agent, backend, _, _ = make_agent(tmp_path, replies)
    assert agent.run("goal").success
    assert not any("you already made" in m["content"] for m in backend.calls[-1]["messages"])
