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
# Route decorators of the common Python web frameworks: @app.route("/x"), @router.post("/x").
ROUTE = re.compile(r"""@\w+(?:\.\w+)*\.(?:route|get|post|put|delete|patch)\(\s*["']([^"']+)["']""")
# Requests a page makes from script.
CALL = re.compile(r"""(?:fetch|axios(?:\.\w+)?)\(\s*[`"']([^`"'$]+)[`"']""")
LOCAL_SERVER = re.compile(r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?")
# Signs that the server hands out pages itself, or accepts calls from pages it did not serve.
SERVES_PAGES = re.compile(r"send_from_directory|send_file|render_template|static_folder|StaticFiles|FileResponse|TemplateResponse")
CROSS_ORIGIN = re.compile(r"CORS|Access-Control-Allow-Origin")
MAIN_GUARD = re.compile(r"""if\s+__name__\s*==\s*["']__main__["']""")
SERVER_START = re.compile(r"\b\w+\.run\(|\bserve_forever\(|\buvicorn\.run\(")
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
        # URLs the page sends data to: form actions here, script calls are found by regex.
        self.calls: list[str] = []
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
        if tag == "form" and values.get("action"):
            self.calls.append(values["action"])

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


def page_problems(root: Path, content: bool = True) -> list[str]:
    """What is wrong with the HTML pages in `root`.

    Structure, local links and placeholders are always checked. With `content`, the page
    is also held to the standard of a site meant to be read: enough text, real styling.
    """
    problems = []
    css = ""
    for page in _files(root, ".html", ".htm"):
        name = page.relative_to(root)
        parsed = _Page()
        parsed.feed(page.read_text(encoding="utf-8", errors="replace"))
        css += parsed.inline_css
        if not parsed.has_title:
            problems.append(f"{name}: no <title>")
        if not parsed.has_viewport:
            problems.append(f"{name}: no viewport <meta>, the page will not adapt to phones")
        if content and len(parsed.text.strip()) < MIN_PAGE_TEXT_CHARS:
            problems.append(f"{name}: almost no text content ({len(parsed.text.strip())} characters)")
        for ref in parsed.refs:
            if EXTERNAL.match(ref) or ref.startswith("/"):
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
    if content and css.count("{") < MIN_CSS_RULES:
        problems.append(f"only {css.count('{')} CSS rules: the page is barely styled")
    return problems


def _route_pattern(route: str) -> re.Pattern[str]:
    """Regex for a route such as /user/<int:id> or /items/{item_id}."""
    parts = re.split(r"<[^>]+>|\{[^}]+\}", route.rstrip("/") or "/")
    return re.compile("[^/]+".join(re.escape(part) for part in parts) + "/?")


def wiring_problems(root: Path) -> list[str]:
    """Pages that talk to a server in the same workspace in a way that cannot work.

    A page can look complete and still do nothing: it calls a URL the server does not
    define, or it uses relative URLs while the server never serves the page itself.
    """
    sources = [
        p.read_text(encoding="utf-8", errors="replace")
        for p in _files(root, ".py") if not p.name.startswith("test_")
    ]
    routes = sorted({route for text in sources for route in ROUTE.findall(text)})
    if not routes:
        return []
    serves_pages = "/" in routes or any(SERVES_PAGES.search(text) for text in sources)
    allows_other_origins = any(CROSS_ORIGIN.search(text) for text in sources)
    patterns = [_route_pattern(route) for route in routes]
    problems = []
    for page in _files(root, ".html", ".htm"):
        name = page.relative_to(root)
        text = page.read_text(encoding="utf-8", errors="replace")
        parsed = _Page()
        parsed.feed(text)
        for ref in parsed.refs:
            script = (page.parent / ref.split("?")[0]).resolve()
            if not EXTERNAL.match(ref) and script.suffix == ".js" and script.is_file():
                text += script.read_text(encoding="utf-8", errors="replace")
        calls = sorted({*parsed.calls, *CALL.findall(text)})
        relative = [c for c in calls if c.startswith("/") and not c.startswith("//")]
        absolute = [c for c in calls if LOCAL_SERVER.match(c)]
        for call in [*relative, *absolute]:
            path = LOCAL_SERVER.sub("", call).split("?")[0].split("#")[0] or "/"
            if not any(pattern.fullmatch(path) for pattern in patterns):
                problems.append(f"{name}: calls '{call}', but the server defines no such route (it has {routes})")
        if relative and not serves_pages:
            problems.append(
                f"{name}: calls the server with relative URLs such as '{relative[0]}', but the server "
                f"never serves {name}, so opened as a file those calls go nowhere. Add a route for '/' "
                "that returns the page, and a test that GET / answers with it"
            )
        if absolute and not serves_pages and not allows_other_origins:
            problems.append(
                f"{name}: calls '{absolute[0]}' from a page the server does not serve and the server does "
                "not allow other origins (CORS), so the browser will block it. Serve the page from the server"
            )
    return problems


def server_entrypoints(root: Path) -> list[Path]:
    """Python files that start a server when run directly."""
    return [
        p for p in _files(root, ".py")
        if not p.name.startswith("test_")
        and (text := p.read_text(encoding="utf-8", errors="replace"))
        and MAIN_GUARD.search(text) and SERVER_START.search(text)
    ]


def check_web(root: Path) -> tuple[bool, str]:
    """Check of a web site: (passed, report)."""
    pages = _files(root, ".html", ".htm")
    if not pages:
        return False, "no .html file was found in the workspace"
    problems = page_problems(root) + wiring_problems(root)
    if problems:
        return False, "\n".join(f"- {p}" for p in problems)
    return True, f"{len(pages)} page(s) checked: structure, local links, content and styling are fine"


def check_project_pages(root: Path) -> tuple[bool, str]:
    """Check of the pages that come with a program: they must be whole and wired to it."""
    problems = page_problems(root, content=False) + wiring_problems(root)
    if problems:
        return False, "\n".join(f"- {p}" for p in problems)
    return True, ""


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
