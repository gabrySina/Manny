import subprocess
import sys

import pytest

from wordfreq import top_words

TEXT = "Il pane, il PANE e ancora il pane!\nOlio; sale... olio? Acqua.\n"


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "testo.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


def test_counts_ignore_case_and_punctuation(sample):
    assert top_words(sample, 2) == [("il", 3), ("pane", 3)]


def test_ties_are_alphabetical(sample):
    assert top_words(sample, 10) == [
        ("il", 3), ("pane", 3), ("olio", 2), ("acqua", 1), ("ancora", 1), ("e", 1), ("sale", 1),
    ]


def test_default_n_is_ten(tmp_path):
    path = tmp_path / "molte.txt"
    path.write_text(" ".join(f"parola{i:02d}" for i in range(15)), encoding="utf-8")
    assert len(top_words(path)) == 10


def test_accented_words(tmp_path):
    path = tmp_path / "accenti.txt"
    path.write_text("perché Perché città", encoding="utf-8")
    assert top_words(path, 1) == [("perché", 2)]


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        top_words(tmp_path / "manca.txt")


def test_invalid_n(sample):
    with pytest.raises(ValueError):
        top_words(sample, 0)


def test_command_line(sample):
    out = subprocess.run(
        [sys.executable, "wordfreq.py", str(sample), "3"], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0
    assert out.stdout.strip().splitlines() == ["il: 3", "pane: 3", "olio: 2"]
