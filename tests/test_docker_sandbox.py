"""Integration tests against a real Docker daemon; skipped when Docker is unavailable."""

import os
import shutil
import subprocess

import pytest

from manny.config import SandboxConfig
from manny.sandbox.docker import DockerSandbox


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0


pytestmark = pytest.mark.skipif(not docker_available(), reason="Docker is not available")


def running(name: str) -> bool:
    out = subprocess.run(["docker", "ps", "-aq", "--filter", f"name={name}"], capture_output=True, text=True)
    return bool(out.stdout.strip())


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("ws")
    with DockerSandbox(workspace, SandboxConfig(), memory_mb=256) as box:
        box.workspace = workspace
        yield box
    assert not running(box.name)


def test_runs_commands_in_the_workspace(sandbox):
    (sandbox.workspace / "hello.py").write_text("print('hi from sandbox')")
    result = sandbox.exec("python hello.py && pwd")
    assert (result.exit_code, result.output) == (0, "hi from sandbox\n/workspace")


def test_files_created_inside_belong_to_the_host_user(sandbox):
    assert sandbox.exec("echo data > made_inside.txt").exit_code == 0
    assert (sandbox.workspace / "made_inside.txt").stat().st_uid == os.getuid()


def test_pytest_is_available(sandbox):
    (sandbox.workspace / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    result = sandbox.exec("python -m pytest -q -p no:cacheprovider test_ok.py")
    assert result.exit_code == 0 and "1 passed" in result.output


def test_network_is_off_by_default_and_on_when_requested(sandbox):
    probe = "python -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)\""
    assert sandbox.exec(probe).exit_code != 0
    assert sandbox.exec(probe, network=True).exit_code == 0
    assert sandbox.exec(probe).exit_code != 0


def test_memory_hog_is_killed_inside_the_container(sandbox):
    result = sandbox.exec("python -c \"x = bytearray(1024 * 1024 * 1024)\"")
    assert result.exit_code == 137 and not result.timed_out
    assert sandbox.exec("echo still alive").output == "still alive"


def test_timeout_kills_the_command(sandbox):
    result = sandbox.exec("sleep 30", timeout=2)
    assert result.exit_code == 137 and result.timed_out


def test_cannot_write_outside_workspace_as_root(sandbox):
    assert sandbox.exec("id -u").output == str(os.getuid())
    assert sandbox.exec("touch /etc/owned").exit_code != 0


def test_nothing_started_by_a_command_outlives_it(sandbox):
    # A server left in the background would keep its port for the rest of the task.
    serve = "python -m http.server 8765 >/dev/null 2>&1 & sleep 1; echo started"
    assert sandbox.exec(serve).output == "started"
    assert sandbox.exec(serve).output == "started"
    listening = sandbox.exec("python -c \"import socket; s = socket.socket(); print(s.connect_ex(('127.0.0.1', 8765)))\"")
    assert listening.output != "0"
    assert sandbox.exec("echo still alive").output == "still alive"
