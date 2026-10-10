import pytest

from manny.bench import BENCH_DIR, ModelOutcome, TaskOutcome, hidden_test_count, load_tasks, parse_spec


def test_tasks_are_well_formed():
    tasks = load_tasks()
    assert [t.name for t in tasks] == ["bugfix", "roman", "userstore", "wordfreq", "landing"]
    for task in tasks:
        assert task.prompt.strip()
        if task.grade == "pytest":
            assert hidden_test_count(task) >= 7
        else:
            assert task.grade == "web"


def test_bugfix_seed_really_fails_its_visible_tests():
    seed = (BENCH_DIR / "bugfix" / "seed" / "stats.py").read_text()
    namespace = {}
    exec(seed, namespace)
    assert namespace["median"]([5, 1, 3]) != 3
    assert namespace["mode"]([3, 1, 3, 1, 2]) != 1


def test_task_filter_and_unknown_task():
    assert [t.name for t in load_tasks(["roman"])] == ["roman"]
    with pytest.raises(ValueError):
        load_tasks(["nope"])


def test_parse_spec():
    assert parse_spec("ornith:9b") == ("ornith:9b", None)
    assert parse_spec("ornith:9b@think") == ("ornith:9b", True)
    assert parse_spec("hf.co/org/model:Q6_K@nothink") == ("hf.co/org/model:Q6_K", False)


def test_score_is_the_mean_share_of_tests_passed_per_task():
    outcome = ModelOutcome("m", tasks=[TaskOutcome("a", 5, 10, 30.0), TaskOutcome("b", 1, 1, 12.0)])
    assert outcome.score == 75 and outcome.seconds == 42
    assert ModelOutcome("empty").score == 0
