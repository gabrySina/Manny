"""Review: one model call that reads the finished work against the task and its criteria."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field

from manny.llm.base import LLMFormatError
from manny.llm.manager import ModelManager

MAX_ISSUES = 6
# Similarity of two problem descriptions above which they count as one finding.
SAME_FINDING_RATIO = 0.75
# A model told that its reply was cut off sometimes reports that as a defect of the work.
ABOUT_THE_REPLY = re.compile(r"(?:response|reply|output|answer) was cut off|cut off at \d+ tokens", re.IGNORECASE)

BAD_INPUT = "Bad input is handled: missing fields, wrong types, duplicates and not-found cases give a clear error, not a crash."
SECURITY = "Security basics: passwords are hashed and never stored or compared in plain text, no secrets in the code, no debug mode left on."
TEST_COVERAGE = "The tests cover every acceptance criterion, including the error cases."
REQUIREMENTS = "Third-party packages are listed in requirements.txt."

CHECKLISTS = {
    "python": "\n".join(f"- {item}" for item in (BAD_INPUT, SECURITY, TEST_COVERAGE, REQUIREMENTS)),
    "web": """\
- The content is real, specific to the subject and written in the language of the task: no placeholders or generic filler.
- The page is complete: a clear headline, the main content sections, a call to action and a footer.
- The styling is a deliberate layout (spacing, colours, typography) that also works on a phone.
- Scripts do something useful or are not there at all.""",
    "research": """\
- The report answers the question that was asked, completely.
- Names, numbers and attributions are correct and not garbled or truncated.
- Every claim has its source URL next to it, and uncertain points are marked as uncertain.
- The report is organised for a reader: a short answer first, then the details.""",
    "other": "- The result does what the task asked, completely.",
}

REVIEWER = """\
You are a strict but fair reviewer. Compare the finished work with the task and report
real defects only.

A defect is one of:
- something the task or an acceptance criterion asks for that is missing or wrong;
- a violation of the checklist below;
- a bug.
It is NOT a defect that the work lacks a feature nobody asked for (extra sections, extra
endpoints, extra options), and style preferences are not defects either.

For each defect give the file it is in, what is wrong, the change to make, and its
severity: "high" for security holes, wrong results or missing requirements, "medium" for
unhandled errors or missing tests, "low" for the rest. At most {max_issues}, each defect
once. If you cannot name the file and the exact problem, do not report it.

Checklist:
{checklist}

Reply ONLY with a JSON object:
{{"issues": [{{"file": "...", "problem": "...", "fix": "...", "severity": "high"}}], "approved": false}}
Set approved to true, with an empty issues list, when the work is ready for the user."""


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Requirements checked one at a time, with the severity of a violation. Asked about the
# whole checklist at once, a small model approves work that breaks it; asked about one
# item and told to trace a concrete case, it finds the violation. Vague items are left out:
# they cost minutes and return nothing usable.
AUDITS: dict[str, list[tuple[str, str]]] = {
    "python": [(BAD_INPUT, "high"), (SECURITY, "high"), (REQUIREMENTS, "medium")],
}

AUDIT = """\
You audit finished work against ONE requirement. Look for a concrete case in the files
where the requirement is not met, and trace what the code does in that case.

Reply ONLY with a JSON object:
{"evidence": "<the code path you traced>", "violated": true, "file": "...", "problem": "...", "fix": "..."}
Set violated to false, with empty file, problem and fix, if you find no such case."""

CONFIRM = """\
You check whether one specific defect is still present in the files you are given.
Judge only that defect: ignore every other problem you may notice.

Reply ONLY with a JSON object:
{"evidence": "<the line or fact in the files that decides it>", "fixed": true}
Set fixed to true only if the files show the defect is gone."""


class Issue(BaseModel):
    file: str
    problem: str
    fix: str
    severity: Literal["high", "medium", "low"] = "medium"

    def __str__(self) -> str:
        return f"{self.file}: {self.problem} Fix: {self.fix}"


def same_finding(a: Issue, b: Issue) -> bool:
    """Whether two findings are the same defect reported twice."""
    if a.file != b.file:
        return False
    return SequenceMatcher(None, a.problem.lower(), b.problem.lower()).ratio() >= SAME_FINDING_RATIO


class Verdict(BaseModel):
    issues: list[Issue] = Field(default_factory=list)
    approved: bool


class Review(BaseModel):
    issues: list[str] = Field(default_factory=list)
    approved: bool


class Audit(BaseModel):
    evidence: str = ""
    violated: bool
    file: str = ""
    problem: str = ""
    fix: str = ""


def audit_requirement(llm: ModelManager, requirement: str, severity: str, files: str) -> Issue | None:
    """The violation of one requirement found in the files, if any."""
    request = f"Requirement:\n{requirement}\n\nFiles:\n{files}"
    try:
        # No retries: an answer that does not come out right the first time is not worth minutes.
        audit = llm.structured(
            [{"role": "system", "content": AUDIT}, {"role": "user", "content": request}], Audit, retries=0
        )
    except LLMFormatError:
        return None
    if not audit.violated or not audit.problem.strip():
        return None
    return Issue(file=audit.file or "the project", problem=audit.problem, fix=audit.fix, severity=severity)


def review_work(
    llm: ModelManager,
    task: str,
    kind: str,
    criteria: list[str],
    files: str,
    check_report: str,
    audit: bool = True,
) -> Review | None:
    """The reviewer's verdict, or None when the model could not produce one."""
    system = REVIEWER.format(max_issues=MAX_ISSUES, checklist=CHECKLISTS.get(kind, CHECKLISTS["other"]))
    request = f"Task: {task}\n\n"
    if criteria:
        request += "Acceptance criteria:\n" + "\n".join(f"- {c}" for c in criteria) + "\n\n"
    request += f"Result of the automatic checks:\n{check_report or '(none)'}\n\nFiles:\n{files}"
    found: list[Issue] = []
    answered = False
    try:
        verdict = llm.structured(
            [{"role": "system", "content": system}, {"role": "user", "content": request}], Verdict
        )
    except LLMFormatError:
        pass
    else:
        answered = True
        found += [] if verdict.approved else verdict.issues
    for requirement, severity in AUDITS.get(kind, []) if audit else []:
        answered = True
        violation = audit_requirement(llm, requirement, severity, files)
        if violation is not None:
            found.append(violation)
    if not answered:
        return None
    if not found:
        return Review(approved=True)
    distinct: list[Issue] = []
    for issue in found:
        if ABOUT_THE_REPLY.search(issue.problem):
            continue
        if not any(same_finding(issue, kept) for kept in distinct):
            distinct.append(issue)
    if not distinct:
        return Review(approved=True)
    # Most severe first: the fixer takes them one at a time, in this order.
    ordered = sorted(distinct[:MAX_ISSUES], key=lambda issue: SEVERITY_ORDER[issue.severity])
    return Review(issues=[str(issue) for issue in ordered], approved=False)


class Confirmation(BaseModel):
    evidence: str = ""
    fixed: bool


def confirm_fixed(llm: ModelManager, issue: str, files: str) -> bool:
    """Whether the files show that one specific defect is gone. Unclear answers count as no."""
    request = f"Defect to check:\n{issue}\n\nFiles:\n{files}"
    try:
        answer = llm.structured(
            [{"role": "system", "content": CONFIRM}, {"role": "user", "content": request}], Confirmation
        )
    except LLMFormatError:
        return False
    return answer.fixed
