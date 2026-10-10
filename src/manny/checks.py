"""Deterministic acceptance checks for work that pytest cannot judge."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from manny.tools.fs import IGNORED_DIRS

REPORT_NAME = "report.md"
MIN_REPORT_CHARS = 500
MIN_PAGE_TEXT_CHARS = 400
MIN_CSS_RULES = 8
URL = re.compile(r"https?://[^\s)>\]\"'`]+")
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", re.IGNORECASE)
# Text that models leave behind instead of real content.
PLACEHOLDERS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"path/to/",
        r"lorem ipsum",
        r"\b(?:product|item|feature|service|section) \d+\b",
        r"description of \w+",
        r"your (?:text|content|company|name|logo) here",
        r"add (?:any|your|more) [^<\n]{0,40}(?:here|as needed)",
        r"\bTODO\b",
        r"example\.com",
        r"placeholder\.com|placehold\.co",
    )
]


def _files(root: Path, *suffixes: str) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
        and not IGNORED_DIRS.intersection(p.relative_to(root).parts)
    )


def normalize_url(url: str) -> str:
    return url.rstrip(".,;:!?").rstrip("/")


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.has_title = False
        self.has_viewport = False
        self.inline_css = ""
        self.text = ""
        self._inside: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self._inside.append(tag)
        if tag == "meta" and (values.get("name") or "").lower() == "viewport":
            self.has_viewport = True
        for attribute in ("href", "src"):
            if values.get(attribute) and tag in ("link", "script", "img", "source", "a", "iframe"):
                self.refs.append(values[attribute])

    def handle_endtag(self, tag: str) -> None:
        if tag in self._inside:
            while self._inside.pop() != tag:
                pass

    def handle_data(self, data: str) -> None:
        current = self._inside[-1] if self._inside else ""
        if current == "title" and data.strip():
            self.has_title = True
        elif current == "style":
            self.inline_css += data
        elif current != "script":
            self.text += data.strip() + " "


def check_web(root: Path) -> tuple[bool, str]:
    """Static check of the HTML pages in `root`: (passed, report)."""
    pages = _files(root, ".html", ".htm")
    if not pages:
        return False, "no .html file was found in the workspace"
    problems = []
    css = ""
    for page in pages:
        name = page.relative_to(root)
        parsed = _Page()
        parsed.feed(page.read_text(encoding="utf-8", errors="replace"))
        css += parsed.inline_css
        if not parsed.has_title:
            problems.append(f"{name}: no <title>")
        if not parsed.has_viewport:
            problems.append(f"{name}: no viewport <meta>, the page will not adapt to phones")
        if len(parsed.text.strip()) < MIN_PAGE_TEXT_CHARS:
            problems.append(f"{name}: almost no text content ({len(parsed.text.strip())} characters)")
        for ref in parsed.refs:
            if EXTERNAL.match(ref):
                continue
            target = (page.parent / ref.split("#")[0].split("?")[0]).resolve()
            if not target.exists():
                problems.append(f"{name}: links to '{ref}', which does not exist in the workspace")
            elif target.suffix.lower() == ".css":
                css += target.read_text(encoding="utf-8", errors="replace")
    for source in _files(root, ".html", ".htm", ".css", ".js"):
        text = source.read_text(encoding="utf-8", errors="replace")
        found = sorted({m.group(0) for pattern in PLACEHOLDERS for m in pattern.finditer(text)})
        if found:
            problems.append(
                f"{source.relative_to(root)}: placeholder text {found[:5]}. Replace it with real "
                "content, or delete those lines completely (a comment saying what could be added is a placeholder too)"
            )
    if css.count("{") < MIN_CSS_RULES:
        problems.append(f"only {css.count('{')} CSS rules: the page is barely styled")
    if problems:
        return False, "\n".join(f"- {p}" for p in problems)
    return True, f"{len(pages)} page(s) checked: structure, local links, content and styling are fine"


def check_report(root: Path, seen_urls: set[str]) -> tuple[bool, str]:
    """Check that the research report exists and cites only sources actually consulted."""
    report = root / REPORT_NAME
    if not report.is_file():
        return False, f"write the final answer to {REPORT_NAME}"
    text = report.read_text(encoding="utf-8", errors="replace")
    problems = []
    if len(text) < MIN_REPORT_CHARS:
        problems.append(f"{REPORT_NAME} is too short ({len(text)} characters) to answer the question")
    cited = {normalize_url(u) for u in URL.findall(text)}
    known = {normalize_url(u) for u in seen_urls}
    if not cited:
        problems.append("the report cites no source URL: put the URL next to each claim")
    invented = sorted(cited - known)
    if invented:
        problems.append(
            "these URLs were never returned by web_search or fetch_url in this session, "
            f"so they must not be cited: {invented[:5]}"
        )
    if problems:
        return False, "\n".join(f"- {p}" for p in problems)
    return True, f"{REPORT_NAME} cites {len(cited)} source(s), all of them consulted in this session"
