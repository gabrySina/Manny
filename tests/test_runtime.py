from fakes import FakeBackend
from test_governor import FakeMemory, make_governor

from manny.llm.base import LoadedModel
from manny.runtime import model_budget


class Loaded(FakeBackend):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def loaded(self):
        return self.models


def test_memory_of_loaded_models_counts_as_available():
    # A 7B model left loaded by the previous run: 5.2 GB on the GPU, 0.7 GB in RAM.
    governor = make_governor(FakeMemory(ram=7000, vram=600))
    budget = model_budget(governor, Loaded([LoadedModel("qwen2.5-coder:7b", 5961, 5218)]))
    assert (budget.ram_mb, budget.vram_mb) == (7000 - 2400 + 743, 600 - 500 + 5218)


def test_budget_is_unchanged_when_nothing_is_loaded():
    governor = make_governor(FakeMemory(ram=7000, vram=5700))
    assert model_budget(governor, Loaded([])) == governor.budget()


def test_no_gpu_stays_no_gpu():
    governor = make_governor(FakeMemory(ram=7000))
    assert model_budget(governor, Loaded([LoadedModel("m", 3000, 0)])).vram_mb is None


from manny.runtime import check_python_project
from manny.sandbox.docker import ExecResult


class SuiteResult:
    def __init__(self, exit_code, output):
        self.result = ExecResult(exit_code, output)

    def exec(self, command, timeout=None, network=False):
        return self.result


SERVER = "@app.route('/login', methods=['POST'])\ndef login(): ...\n"
PAGE = ('<html><head><title>t</title><meta name="viewport" content="width=device-width"></head>'
        "<body><script>fetch('/login')</script></body></html>")


def test_passing_tests_are_not_enough_when_a_page_is_wired_to_nothing(tmp_path):
    (tmp_path / "server.py").write_text(SERVER)
    (tmp_path / "index.html").write_text(PAGE)
    passed, report = check_python_project(SuiteResult(0, "11 passed"), tmp_path)
    assert not passed and report.startswith("The tests pass, but the web pages")
    assert "never serves index.html" in report


def test_failing_tests_are_reported_first_and_plain_python_is_untouched(tmp_path):
    (tmp_path / "server.py").write_text(SERVER)
    (tmp_path / "index.html").write_text(PAGE)
    passed, report = check_python_project(SuiteResult(1, "1 failed"), tmp_path)
    assert not passed and "1 failed" in report and "web pages" not in report
    (tmp_path / "index.html").unlink()
    assert check_python_project(SuiteResult(0, "11 passed"), tmp_path) == (True, "exit code 0\n11 passed")


class Scripted:
    """Sandbox answering each command by its first matching prefix."""

    def __init__(self, **results):
        self.results, self.commands = results, []

    def exec(self, command, timeout=None, network=False):
        self.commands.append(command)
        key = "start" if command.startswith("exec python") else "tests"
        return self.results[key]


STARTABLE = SERVER + "\n@app.route('/')\ndef home(): ...\n\nif __name__ == '__main__':\n    db.create_all()\n    app.run(debug=False)\n"


def test_a_server_that_crashes_at_startup_fails_the_check_even_with_green_tests(tmp_path):
    (tmp_path / "server.py").write_text(STARTABLE)
    crash = ExecResult(1, "Traceback ...\nRuntimeError: Working outside of application context.")
    sandbox = Scripted(tests=ExecResult(0, "12 passed"), start=crash)
    passed, report = check_python_project(sandbox, tmp_path)
    assert not passed and "`python server.py` crashes at startup" in report
    assert "Working outside of application context" in report
    assert sandbox.commands[1] == "exec python server.py"


def test_a_server_still_running_when_time_is_up_has_started(tmp_path):
    (tmp_path / "server.py").write_text(STARTABLE)
    running = ExecResult(137, " * Running on http://127.0.0.1:5000", timed_out=True)
    passed, _ = check_python_project(Scripted(tests=ExecResult(0, "12 passed"), start=running), tmp_path)
    assert passed


def test_scripts_that_are_not_servers_are_not_started(tmp_path):
    (tmp_path / "wordfreq.py").write_text("import sys\nif __name__ == '__main__':\n    print(sys.argv[1])\n")
    (tmp_path / "test_wordfreq.py").write_text("if __name__ == '__main__':\n    unittest.main()\n    app.run()\n")
    sandbox = Scripted(tests=ExecResult(0, "7 passed"), start=ExecResult(1, "IndexError"))
    assert check_python_project(sandbox, tmp_path)[0] and len(sandbox.commands) == 1
