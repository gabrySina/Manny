import pytest

from manny.tools.base import ToolError, truncate_middle
from manny.tools.fs import ListArgs, ReadArgs, ReplaceArgs, Workspace, WriteArgs, fs_tools


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


@pytest.fixture
def tools(ws):
    return {t.name: t.run for t in fs_tools(ws)}


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "a/../../outside.txt", "/workspace/../x"])
def test_paths_outside_workspace_are_refused(ws, path):
    with pytest.raises(ToolError):
        ws.resolve(path)


def test_symlink_escape_is_refused(ws, tmp_path):
    (ws.root / "link").symlink_to(tmp_path)
    with pytest.raises(ToolError):
        ws.resolve("link/secret.txt")


def test_sandbox_mount_prefix_maps_to_workspace(ws):
    assert ws.resolve("/workspace/src/a.py") == ws.root / "src" / "a.py"
    assert ws.resolve("/workspace") == ws.root


def test_write_read_list(tools):
    assert "wrote 5 characters to pkg/a.py" == tools["write_file"](WriteArgs(path="pkg/a.py", content="x = 1"))
    assert tools["read_file"](ReadArgs(path="pkg/a.py")) == "x = 1"
    tools["write_file"](WriteArgs(path="__pycache__/a.pyc", content="junk"))
    assert tools["list_files"](ListArgs()) == "pkg/a.py"


def test_list_empty_and_read_missing(tools):
    assert tools["list_files"](ListArgs()) == "(no files)"
    with pytest.raises(ToolError):
        tools["read_file"](ReadArgs(path="nope.py"))


def test_replace_requires_exactly_one_match(tools):
    tools["write_file"](WriteArgs(path="a.py", content="a = 1\nb = 1\n"))
    with pytest.raises(ToolError, match="2 times"):
        tools["replace_in_file"](ReplaceArgs(path="a.py", old="= 1", new="= 2"))
    tools["replace_in_file"](ReplaceArgs(path="a.py", old="a = 1", new="a = 2"))
    assert tools["read_file"](ReadArgs(path="a.py")) == "a = 2\nb = 1\n"


def test_truncate_middle_keeps_both_ends():
    text = "START" + "x" * 1000 + "END"
    short = truncate_middle(text, 90)
    assert short.startswith("START") and short.endswith("END") and "omitted" in short
    assert truncate_middle("short", 90) == "short"


def test_snapshot_fits_the_budget_and_skips_binaries(ws):
    (ws.root / "a.py").write_text("print('a')")
    (ws.root / "big.py").write_text("x" * 5000)
    (ws.root / "image.bin").write_bytes(bytes([0xFF, 0xFE, 0x00, 0x80]))
    (ws.root / "__pycache__").mkdir()
    (ws.root / "__pycache__" / "junk.py").write_text("junk")
    snap = ws.snapshot(3000)
    assert "### a.py\nprint('a')" in snap and "### big.py" in snap and "[truncated]" in snap
    assert "image.bin" not in snap and "junk" not in snap and len(snap) < 3500


def test_snapshot_of_empty_workspace(ws):
    assert ws.snapshot(1000) == "(no files)"


def test_double_escaped_code_is_refused(tools):
    broken = "def f():\\n    x = 1\\n    y = 2\\n    return x + y"
    with pytest.raises(ToolError, match="real line breaks"):
        tools["write_file"](WriteArgs(path="a.py", content=broken))
    tools["write_file"](WriteArgs(path="a.py", content="a = 1\n"))
    with pytest.raises(ToolError, match="real line breaks"):
        tools["replace_in_file"](ReplaceArgs(path="a.py", old="a = 1", new=broken))
    assert tools["read_file"](ReadArgs(path="a.py")) == "a = 1\n"


def test_legitimate_backslash_n_in_code_is_accepted(tools):
    one_liner = 'print("a\\nb\\nc\\nd")'
    tools["write_file"](WriteArgs(path="multi.py", content=f"x = 1\n{one_liner}\n"))
    tools["write_file"](WriteArgs(path="short.py", content='print("a\\nb")'))


def test_checkpoint_restores_changed_deleted_and_new_files(ws):
    (ws.root / "pkg").mkdir()
    (ws.root / "pkg" / "a.py").write_text("original")
    (ws.root / "b.bin").write_bytes(bytes([0xFF, 0x00]))
    restore = ws.checkpoint()
    (ws.root / "pkg" / "a.py").write_text("changed")
    (ws.root / "b.bin").unlink()
    (ws.root / "new.py").write_text("added later")
    restore()
    assert (ws.root / "pkg" / "a.py").read_text() == "original"
    assert (ws.root / "b.bin").read_bytes() == bytes([0xFF, 0x00])
    assert not (ws.root / "new.py").exists()


def test_fingerprint_changes_with_content_names_and_new_files(ws):
    empty = ws.fingerprint()
    (ws.root / "a.py").write_text("x = 1")
    one = ws.fingerprint()
    (ws.root / "a.py").write_text("x = 2")
    two = ws.fingerprint()
    (ws.root / "a.py").rename(ws.root / "b.py")
    assert len({empty, one, two, ws.fingerprint()}) == 4
    (ws.root / "__pycache__").mkdir()
    (ws.root / "__pycache__" / "b.pyc").write_text("cache")
    (ws.root / "b.py").rename(ws.root / "a.py")
    assert ws.fingerprint() == two


def test_long_files_are_read_in_pages_of_whole_lines(tools):
    lines = [f"value_{i:04d} = {i}" for i in range(1000)]
    tools["write_file"](WriteArgs(path="big.py", content="\n".join(lines) + "\n"))
    first = tools["read_file"](ReadArgs(path="big.py"))
    body, _, note = first.rpartition("\n... [")
    assert len(body) <= 8000 and body.endswith("\n") and body.startswith("value_0000 = 0\n")
    shown = body.count("\n")
    assert note == f"lines 1-{shown} of 1000: continue with read_file(path, start_line={shown + 1})]"
    # Paging through to the end gives back exactly the file.
    collected, start = body, shown + 1
    while True:
        page = tools["read_file"](ReadArgs(path="big.py", start_line=start))
        text, marker, rest = page.rpartition("\n... [")
        if not marker:
            collected += page
            break
        collected += text
        start = int(rest.split("start_line=")[1].rstrip(")]"))
    assert collected == "\n".join(lines) + "\n"


def test_read_file_start_line_bounds_and_empty_file(tools):
    tools["write_file"](WriteArgs(path="a.py", content="one\ntwo\n"))
    assert tools["read_file"](ReadArgs(path="a.py", start_line=2)) == "two\n"
    with pytest.raises(ToolError, match="between 1 and 2"):
        tools["read_file"](ReadArgs(path="a.py", start_line=3))
    tools["write_file"](WriteArgs(path="empty.py", content=""))
    assert tools["read_file"](ReadArgs(path="empty.py")) == ""
