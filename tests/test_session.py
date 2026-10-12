import json

import pytest
from typer.testing import CliRunner

from manny import session
from manny.agents.spec import Action, AgentResult
from manny.cli import app
from manny.session import SessionLog, recent


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MANNY_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def test_a_run_leaves_transcript_events_and_result(tmp_path):
    log = SessionLog("crea un server", tmp_path / "ws")
    log.transcript.write("Tipo di lavoro: python\n")
    log.event("coder", "action", Action(tool="write_file", args={"path": "a.py", "content": "x = 1"}))
    log.event("orchestrator", "task_end", AgentResult(True, "done", 3))
    log.event("orchestrator", "final_check", (True, "4 passed"))
    log.finish({"outcome": "success", "seconds": 42})

    events = [json.loads(line) for line in (log.dir / "events.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["action", "task_end", "final_check"]
    # Events keep the full arguments, unlike the shortened lines shown on screen.
    assert events[0]["data"]["args"] == {"path": "a.py", "content": "x = 1"}
    assert events[1]["data"] == {"success": True, "summary": "done", "steps": 3}
    assert events[2]["data"] == [True, "4 passed"]
    result = json.loads((log.dir / "result.json").read_text())
    assert result["task"] == "crea un server" and result["outcome"] == "success" and result["seconds"] == 42
    assert (log.dir / "transcript.txt").read_text() == "Tipo di lavoro: python\n"


def test_an_interrupted_run_is_still_on_record(tmp_path):
    log = SessionLog("task", tmp_path)
    log.event("coder", "observation", "x" * 50000)
    result = json.loads((log.dir / "result.json").read_text())
    assert result["outcome"] == "interrupted"
    stored = json.loads((log.dir / "events.jsonl").read_text())
    assert len(stored["data"]) == session.MAX_TEXT_CHARS


def make_run(state_dir, name, task, outcome):
    folder = state_dir / "runs" / name
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"task": task, "outcome": outcome, "started": name, "seconds": 5}))
    (folder / "transcript.txt").write_text(f"trascrizione di {task} [con parentesi]\n")


def test_recent_runs_are_listed_newest_first(state_dir):
    assert recent() == []
    make_run(state_dir, "20261010-100000", "primo", "success")
    make_run(state_dir, "20261012-140000", "secondo", "failure")
    assert [p.name for p in recent()] == ["20261012-140000", "20261010-100000"]


def test_log_command_shows_the_latest_run_and_lists_them(state_dir):
    runner = CliRunner()
    assert runner.invoke(app, ["log"]).exit_code == 1
    make_run(state_dir, "20261010-100000", "primo", "success")
    make_run(state_dir, "20261012-140000", "secondo", "failure")
    latest = runner.invoke(app, ["log"])
    assert latest.exit_code == 0 and "trascrizione di secondo [con parentesi]" in latest.output
    assert "trascrizione di primo" in runner.invoke(app, ["log", "2"]).output
    listing = runner.invoke(app, ["log", "--list"]).output
    assert "secondo" in listing and "primo" in listing and "failure" in listing
    assert runner.invoke(app, ["log", "5"]).exit_code == 1
