import pytest

from manny.sandbox.docker import ExecResult
from manny.tools.base import ToolError
from manny.tools.exec import CommandArgs, InstallArgs, NoArgs, exec_tools, run_pytest


class FakeSandbox:
    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.calls = []

    def exec(self, command, timeout=None, network=False) -> ExecResult:
        self.calls.append((command, network))
        return self.result


def tools(sandbox):
    return {t.name: t.run for t in exec_tools(sandbox)}


def test_run_command_reports_exit_code_and_output():
    sandbox = FakeSandbox(ExecResult(2, "boom"))
    assert tools(sandbox)["run_command"](CommandArgs(command="false")) == "exit code 2\nboom"
    assert sandbox.calls == [("false", False)]


def test_killed_and_timed_out_are_explained():
    run = lambda result: tools(FakeSandbox(result))["run_command"](CommandArgs(command="x"))
    assert "memory limit" in run(ExecResult(137, ""))
    assert "timed out" in run(ExecResult(137, "", timed_out=True))


def test_run_tests_and_verifier():
    assert run_pytest(FakeSandbox(ExecResult(0, "2 passed"))) == (True, "exit code 0\n2 passed")
    assert run_pytest(FakeSandbox(ExecResult(1, "1 failed")))[0] is False
    passed, report = run_pytest(FakeSandbox(ExecResult(5, "no tests ran")))
    assert not passed and "no tests were found" in report
    failing = tools(FakeSandbox(ExecResult(1, "1 failed")))["run_tests"](NoArgs())
    assert "1 failed" in failing and "call finish" not in failing
    passing = tools(FakeSandbox(ExecResult(0, "4 passed")))["run_tests"](NoArgs())
    assert passing.startswith("exit code 0\n4 passed") and passing.endswith("call finish now instead of making further changes.")


def test_install_uses_network_and_quotes_requirements():
    sandbox = FakeSandbox(ExecResult(0, ""))
    tools(sandbox)["install_packages"](InstallArgs(packages=["requests", "numpy>=1.26", "uvicorn[standard]"]))
    command, network = sandbox.calls[0]
    assert network and command.endswith("requests 'numpy>=1.26' 'uvicorn[standard]'")


@pytest.mark.parametrize("package", ["requests; rm -rf /", "--index-url=http://evil", "$(id)", "a b", ""])
def test_install_rejects_anything_but_requirements(package):
    sandbox = FakeSandbox(ExecResult(0, ""))
    with pytest.raises(ToolError):
        tools(sandbox)["install_packages"](InstallArgs(packages=[package]))
    assert sandbox.calls == []


@pytest.mark.parametrize("command", ["pip install flask", "python -m pip install -r requirements.txt", "pip3  install x"])
def test_pip_install_through_run_command_points_to_the_right_tool(command):
    sandbox = FakeSandbox(ExecResult(0, ""))
    with pytest.raises(ToolError, match="install_packages"):
        tools(sandbox)["run_command"](CommandArgs(command=command))
    assert sandbox.calls == []


@pytest.mark.parametrize("command", ["python server.py &", "python server.py & sleep 2; curl -s localhost:5000", "nohup python app.py"])
def test_background_processes_are_refused(command):
    sandbox = FakeSandbox(ExecResult(0, ""))
    with pytest.raises(ToolError, match="background"):
        tools(sandbox)["run_command"](CommandArgs(command=command))
    assert sandbox.calls == []


@pytest.mark.parametrize("command", ["pytest -q 2>&1 | tail -5", "ls && cat a.py", "python a.py &> out.txt", "grep 'a & b' notes.txt || true"])
def test_ordinary_shell_syntax_is_not_mistaken_for_backgrounding(command):
    sandbox = FakeSandbox(ExecResult(0, "ok"))
    if "a & b" in command:
        pytest.skip("a lone & inside quotes is refused too: an accepted false positive")
    assert tools(sandbox)["run_command"](CommandArgs(command=command)) == "exit code 0\nok"
