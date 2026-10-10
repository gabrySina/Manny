"""File tools, confined to the task workspace."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from manny.tools.base import Tool, ToolError

# Where the workspace is mounted inside the sandbox; agents may use either form.
SANDBOX_MOUNT = "/workspace"
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
MAX_LISTED = 200
NO_FILES = "(no files)"
# One read_file call returns at most this much, in whole lines; longer files are paged.
READ_PAGE_CHARS = 8000
MIN_SNAPSHOT_SHARE = 1500
# Literal backslash-n sequences, with no real line break, that mark double-escaped text.
MIN_ESCAPED_BREAKS = 3


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        """Absolute path for `path`, refusing anything outside the workspace."""
        if path == SANDBOX_MOUNT or path.startswith(SANDBOX_MOUNT + "/"):
            path = path[len(SANDBOX_MOUNT):].lstrip("/") or "."
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ToolError(f"'{path}' is outside the workspace")
        return resolved


    def files(self) -> list[Path]:
        return [
            p for p in sorted(self.root.rglob("*"))
            if p.is_file() and not IGNORED_DIRS.intersection(p.relative_to(self.root).parts)
        ]

    def fingerprint(self) -> str:
        """A value that changes whenever any file in the workspace changes."""
        digest = hashlib.sha256()
        for path in self.files():
            digest.update(str(path.relative_to(self.root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def checkpoint(self) -> Callable[[], None]:
        """Remember the current files; the returned function puts them back exactly."""
        saved = {path.relative_to(self.root): path.read_bytes() for path in self.files()}

        def restore() -> None:
            for path in self.files():
                if path.relative_to(self.root) not in saved:
                    path.unlink()
            for name, content in saved.items():
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

        return restore

    def snapshot(self, limit_chars: int) -> str:
        """The text files of the workspace in one string, cut to fit `limit_chars`."""
        texts = []
        for path in self.files():
            try:
                texts.append((path.relative_to(self.root), path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                continue
        if not texts:
            return NO_FILES
        share = max(MIN_SNAPSHOT_SHARE, limit_chars // len(texts))
        parts, used = [], 0
        for index, (name, text) in enumerate(texts):
            if used >= limit_chars:
                parts.append(f"({len(texts) - index} more files not shown)")
                break
            if len(text) > share:
                text = text[:share] + "\n... [truncated]"
            parts.append(f"### {name}\n{text}")
            used += len(text)
        return "\n\n".join(parts)


def reject_escaped_text(text: str) -> None:
    """Refuse text whose line breaks arrived as the two characters backslash and n.

    Models writing code inside JSON sometimes escape twice; saved as is, the file would
    be a single broken line.
    """
    if "\n" not in text and text.count("\\n") >= MIN_ESCAPED_BREAKS:
        raise ToolError(
            "the text contains the characters \\n instead of real line breaks, so the file "
            "would be one broken line. Send it again with real line breaks (a single \\n in JSON)"
        )


class ListArgs(BaseModel):
    path: str = "."


class ReadArgs(BaseModel):
    path: str
    start_line: int = 1


class WriteArgs(BaseModel):
    path: str
    content: str


class ReplaceArgs(BaseModel):
    path: str
    old: str
    new: str


def fs_tools(ws: Workspace) -> list[Tool]:
    def list_files(args: ListArgs) -> str:
        base = ws.resolve(args.path)
        if not base.is_dir():
            raise ToolError(f"'{args.path}' is not a directory")
        found = [str(p.relative_to(ws.root)) for p in ws.files() if p.is_relative_to(base)]
        if not found:
            return NO_FILES
        extra = f"\n... and {len(found) - MAX_LISTED} more" if len(found) > MAX_LISTED else ""
        return "\n".join(found[:MAX_LISTED]) + extra

    def read_file(args: ReadArgs) -> str:
        target = ws.resolve(args.path)
        if not target.is_file():
            raise ToolError(f"'{args.path}' does not exist")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"'{args.path}' is not a text file") from None
        lines = text.splitlines(keepends=True)
        if args.start_line < 1 or args.start_line > max(len(lines), 1):
            raise ToolError(f"start_line must be between 1 and {max(len(lines), 1)}")
        page: list[str] = []
        for line in lines[args.start_line - 1:]:
            if page and sum(map(len, page)) + len(line) > READ_PAGE_CHARS:
                break
            page.append(line)
        last = args.start_line - 1 + len(page)
        shown = "".join(page)
        if last < len(lines):
            shown += (
                f"\n... [lines {args.start_line}-{last} of {len(lines)}: "
                f"continue with read_file(path, start_line={last + 1})]"
            )
        return shown

    def write_file(args: WriteArgs) -> str:
        target = ws.resolve(args.path)
        if target.is_dir():
            raise ToolError(f"'{args.path}' is a directory")
        reject_escaped_text(args.content)
        if target.is_file() and args.content and target.read_text(encoding="utf-8", errors="replace") == args.content:
            raise ToolError(
                f"'{args.path}' already contains exactly this content, nothing changed. "
                "Writing it again cannot fix anything: change the content or look elsewhere"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        if not args.content:
            return f"wrote an EMPTY file {target.relative_to(ws.root)}; if that is not intended, write it again with its full content"
        return f"wrote {len(args.content)} characters to {target.relative_to(ws.root)}"

    def replace_in_file(args: ReplaceArgs) -> str:
        target = ws.resolve(args.path)
        if not target.is_file():
            raise ToolError(f"'{args.path}' does not exist")
        reject_escaped_text(args.new)
        text = target.read_text(encoding="utf-8")
        count = text.count(args.old)
        if count != 1:
            raise ToolError(
                f"'old' must match exactly once, it matches {count} times. Read the file and copy "
                "the text exactly (same spaces and line breaks), or rewrite the file with write_file"
            )
        target.write_text(text.replace(args.old, args.new), encoding="utf-8")
        return f"updated {target.relative_to(ws.root)}"

    return [
        Tool("list_files", "List the files in the workspace.", ListArgs, list_files),
        Tool("read_file", "Read a text file. Long files come in pages: pass start_line to continue.", ReadArgs, read_file, verbatim=True),
        Tool("write_file", "Create or overwrite a file with the full content.", WriteArgs, write_file),
        Tool("replace_in_file", "Replace one exact occurrence of 'old' with 'new' in a file.", ReplaceArgs, replace_in_file),
    ]
