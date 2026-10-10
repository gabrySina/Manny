import json

from fakes import FakeBackend, choice
from test_governor import FakeMemory, make_governor

from manny.agents.review import confirm_fixed, review_work
from manny.llm.manager import ModelManager


def manager(replies):
    backend = FakeBackend(replies)
    return ModelManager(backend, make_governor(FakeMemory(ram=8000, vram=5700)), [choice("m", 3000)]), backend


def issue(problem, severity):
    return {"file": "app.py", "problem": problem, "fix": "fix it", "severity": severity}


def test_findings_are_ordered_by_severity_keeping_the_reviewers_order_within_one():
    found = [issue("style", "low"), issue("sha256", "high"), issue("no test", "medium"), issue("injection", "high")]
    llm, backend = manager([json.dumps({"issues": found, "approved": False})])
    review = review_work(llm, "task", "python", ["hash passwords"], "### app.py\ncode", "2 passed")
    assert [text.split(": ")[1].split(" Fix")[0] for text in review.issues] == ["sha256", "injection", "no test", "style"]
    request = backend.calls[0]["messages"][1]["content"]
    assert "- hash passwords" in request and "### app.py" in request and "2 passed" in request


def test_approved_review_has_no_findings_and_severity_defaults_to_medium():
    llm, _ = manager([json.dumps({"issues": [], "approved": True})])
    assert review_work(llm, "t", "python", [], "f", "").approved
    legacy = {"issues": [{"file": "a.py", "problem": "p", "fix": "f"}], "approved": False}
    llm, _ = manager([json.dumps(legacy)])
    assert review_work(llm, "t", "python", [], "f", "").issues == ["a.py: p Fix: f"]


def test_confirm_fixed_asks_about_one_defect_only():
    llm, backend = manager([json.dumps({"evidence": "bcrypt.hashpw on line 12", "fixed": True})])
    assert confirm_fixed(llm, "app.py: sha256 Fix: use bcrypt", "### app.py\nbcrypt.hashpw(...)") is True
    system, request = (m["content"] for m in backend.calls[0]["messages"])
    assert "Judge only that defect" in system
    assert "app.py: sha256 Fix: use bcrypt" in request and "bcrypt.hashpw" in request


def test_confirm_fixed_treats_no_and_unreadable_answers_as_not_fixed():
    llm, _ = manager([json.dumps({"evidence": "still sha256", "fixed": False})])
    assert confirm_fixed(llm, "issue", "files") is False
    llm, _ = manager(["garbage"] * 3)
    assert confirm_fixed(llm, "issue", "files") is False


def test_the_same_finding_reported_twice_is_kept_once():
    found = [
        issue("La funzione hash_password usa SHA-256 senza salt, vulnerabile a forza bruta", "high"),
        issue("la funzione hash_password usa SHA-256 senza salt, vulnerabile ad attacchi di forza bruta", "high"),
        issue("Manca requirements.txt", "medium"),
        {"file": "test_app.py", "problem": "Manca requirements.txt", "fix": "x", "severity": "low"},
    ]
    llm, _ = manager([json.dumps({"issues": found, "approved": False})])
    review = review_work(llm, "t", "python", [], "f", "")
    assert len(review.issues) == 3 and review.issues[0].startswith("app.py: La funzione hash_password")
