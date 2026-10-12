"""Tools that run code inside the sandbox."""

from __future__ import annotations

import re
import shlex

from pydantic import BaseModel

from manny.sandbox.docker import KILLED, ExecResult, Sandbox
from manny.tools.base import Tool, ToolError

# One failure at a time: small models fix a single clear error far better than several at once.
PYTEST = "python -m pytest -q --no-header --tb=short --maxfail=1 -p no:cacheprovider"
# pytest exit status when no test was collected.
NO_TESTS = 5
INSTALL_TIMEOUT = 300.0
TESTS_PASS_HINT = (
    "All tests pass. If everything your goal asks for is in place, call finish now "
    "instead of making further changes."
)
PIP_INSTALL = re.compile(r"\bpip3?\s+install\b")
# A lone & (not && and not part of a redirect such as 2>&1), or a detaching wrapper.
BACKGROUND = re.compile(r"(?<![&>|])&(?![&>])|\bnohup\b|\bsetsid\b")
# A requirement such as "requests" or "numpy>=1.26"; rejects options and shell syntax.
REQUIREMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?([<>=!~]=?[A-Za-z0-9.*+!-]+)?")


class CommandArgs(BaseModel):
    command: str


class NoArgs(BaseModel):
    pass


class InstallArgs(BaseModel):
    packages: list[str]


def render(result: ExecResult) -> str:
    if result.timed_out:
        status = "timed out and was killed"
    elif result.exit_code == KILLED:
        status = "killed (exit code 137): it probably exceeded the sandbox memory limit"
    else:
        status = f"exit code {result.exit_code}"
    return f"{status}\n{result.output}" if result.output else status


def run_pytest(sandbox: Sandbox) -> tuple[bool, str]:
    """Run the workspace test suite: (passed, report)."""
    result = sandbox.exec(PYTEST)
    if result.exit_code == NO_TESTS:
        return False, "no tests were found: write pytest tests (test_*.py) for the code"
    return result.exit_code == 0, render(result)


def exec_tools(sandbox: Sandbox) -> list[Tool]:
    def run_command(args: CommandArgs) -> str:
        if PIP_INSTALL.search(args.command):
            raise ToolError("run_command has no network access: install packages with install_packages")
        if BACKGROUND.search(args.command):
            raise ToolError(
                "run_command cannot start background processes or servers: nothing survives the "
                "command. The final check starts your server itself; test a web application "
                "through its framework's test client"
            )
        return render(sandbox.exec(args.command))

    def run_tests(args: NoArgs) -> str:
        passed, report = run_pytest(sandbox)
        # Models tend to keep polishing working code until they run out of time.
        return f"{report}\n\n{TESTS_PASS_HINT}" if passed else report

    def install_packages(args: InstallArgs) -> str:
        bad = [p for p in args.packages if not REQUIREMENT.fullmatch(p)]
        if bad or not args.packages:
            raise ToolError(f"not valid package names: {bad or 'empty list'}")
        command = "python -m pip install -q " + " ".join(shlex.quote(p) for p in args.packages)
        return render(sandbox.exec(command, timeout=INSTALL_TIMEOUT, network=True))

    return [
        Tool("run_command", "Run a shell command in the workspace. No network access. The command must end by itself: never start a server, test web apps through their test client.", CommandArgs, run_command),
        Tool("run_tests", "Run the pytest test suite of the workspace.", NoArgs, run_tests),
        Tool("install_packages", "Install Python packages from PyPI with pip.", InstallArgs, install_packages),
    ]
