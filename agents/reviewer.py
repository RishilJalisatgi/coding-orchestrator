"""
agents/reviewer.py

The independent review agent: turns "is this code good?" into a verdict the review/revise loop can act on.

    from agents.reviewer import ReviewerAgent

    reviewer = ReviewerAgent(registry.generate)
    review = reviewer.review(requirement, design, repo, reviewer_model, coder_model=coder_model,
                             analysis=analysis, evaluation=evaluation,        # Evaluation from evaluation.evaluator
                             previous=last_review, iteration=2, diff_text=repo.diff(snap).unified,
                             changed_files=result.changed)
    review.passed                       # decided by CODE from the findings, not by the model's say-so
    review.findings                     # [Finding] sorted worst first (objective facts + model findings)
    review.to_revision_prompt()         # text for CoderAgent.revise(...)
    review.architecture_feedback()      # text for HLDAgent.revise(...), "" if no design-level problems
    review.stuck, review.severity_score, review.improved_over(previous)   # loop control: escalate / roll back
    state_manager.record_review(state, review.passed, review.for_state(), reviewer_model, coder_model)

Principles:
  * Objective evidence comes first. Compile errors, import failures, failing tests and failed runs from the
    Evaluation become findings BY CODE (source="objective"), so a lazy or over-friendly reviewer can never let
    broken code pass. The model adds judgement on top: requirement coverage, logic bugs, security, design.
  * The verdict is computed: any blocker/major finding -> fail; otherwise pass. A model that says "fail" without a
    single blocking finding gives the coder nothing to fix, so it is treated as pass (with a warning); a model
    that says "pass" next to a blocker is overruled.
  * Findings are checked: a file that does not exist in the workspace is cleared and the finding is downgraded one
    level (hallucination guard); impossible line numbers are dropped; duplicates of objective findings are merged.
  * Requirement coverage is explicit: a requirement reported "missing" becomes a major finding.
  * Findings are tracked between rounds (resolved / persisting / new). A blocking finding that survives a revision
    is flagged `stuck`, the signal to escalate (revise the architecture, switch coder) instead of repeating itself.
  * Different model from the coder is the scheduler's job; this agent only warns if they are the same.

Model errors from the first call propagate (the orchestrator asks scheduler.fallback_model("review")). If the reply is
not valid JSON, ONE repair call is made; if that also fails the result is returned with reliable=False and a
verdict based on the objective findings only, so the orchestrator can get a second opinion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from context.indexer import RepoIndex
from context.selector import ContextSelector
from core.logger import get_logger
from core.task_analyzer import extract_json_object
from models.base import ModelRequest, ModelResponse

logger = get_logger("agents.reviewer")

GenerateFn = Callable[[str, ModelRequest], ModelResponse]

SEVERITIES = ("blocker", "major", "minor", "nit")
CATEGORIES = ("correctness", "requirements", "security", "design", "tests", "performance", "maintainability", "docs")
SEVERITY_WEIGHT = {"blocker": 8.0, "major": 3.0, "minor": 1.0, "nit": 0.2}
SEVERITY_ALIASES = {"critical": "blocker", "fatal": "blocker", "high": "major", "important": "major", "medium": "minor",
                    "moderate": "minor", "low": "minor", "trivial": "nit", "info": "nit", "style": "nit", "suggestion": "nit"}
CATEGORY_ALIASES = {"bug": "correctness", "logic": "correctness", "error handling": "correctness", "requirement": "requirements",
                    "architecture": "design", "structure": "design", "test": "tests", "testing": "tests", "perf": "performance",
                    "documentation": "docs", "style": "maintainability", "quality": "maintainability", "readability": "maintainability"}
DEFAULT_FAIL_ON = ("blocker", "major")
MAX_MODEL_FINDINGS = 15
MAX_OBJECTIVE_FINDINGS = 20
CONTEXT_BUDGET_TOKENS = 7000
MATCH_THRESHOLD = 0.5
STOPWORDS = {"the", "a", "an", "in", "of", "to", "is", "are", "for", "and", "or", "with", "on", "at", "by", "be", "it",
             "this", "that", "does", "do", "not"}


class ReviewError(Exception):
    """Misuse: nothing to review."""


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #

@dataclass
class Finding:
    title: str
    severity: str = "minor"                  # blocker | major | minor | nit
    category: str = "correctness"
    file: str = ""
    line: Optional[int] = None
    detail: str = ""
    suggestion: str = ""
    source: str = "model"                    # objective (tests/compile/run facts) | model
    key: str = ""                            # objective findings: an identifier a model finding may mention (test name...)
    id: str = ""
    rounds: int = 1                          # consecutive reviews this finding has appeared in
    status: str = "new"                      # new | persisting

    @property
    def rank(self) -> int:
        return SEVERITIES.index(self.severity)

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}" if self.file and self.line else self.file

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, raw: dict) -> "Finding":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if t not in STOPWORDS}


def similar(a: Finding, b: Finding, threshold: float = MATCH_THRESHOLD) -> bool:
    """Do two findings describe the same problem? Same file (or one unlocated) and similar wording / shared key."""
    if a.file and b.file and a.file != b.file:
        return False
    for obj, other in ((a, b), (b, a)):
        if obj.key and obj.key.lower() in f"{other.title} {other.detail}".lower():
            return True
    ta, tb = _tokens(a.title), _tokens(b.title)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= threshold


def severity_score(findings: list) -> float:
    return sum(SEVERITY_WEIGHT[f.severity] for f in findings)


# --------------------------------------------------------------------------- #
# Objective findings (from evaluation.evaluator.Evaluation, duck-typed)
# --------------------------------------------------------------------------- #

def _test_file(name: str) -> str:
    m = re.match(r"^([\w./\\-]+\.py)::", name or "")
    return m.group(1).replace("\\", "/") if m else ""


def objective_findings(evaluation: Any) -> list:
    """Facts the code can assert without a model: compile/import errors, failing tests, failed runs, serious static issues."""
    if evaluation is None:
        return []
    out: list = []
    if getattr(evaluation, "empty", False):
        return [Finding("No files were produced", "blocker", "correctness", detail="The workspace is empty.",
                        source="objective", key="no files")]
    static = getattr(evaluation, "static", None)
    flagged_stems = set()
    for path, message in (getattr(static, "compile_errors", None) or [])[:6]:
        flagged_stems.add(re.sub(r"\.\w+$", "", str(path).split("/")[-1]))
        out.append(Finding(f"File does not compile: {path}", "blocker", "correctness", file=str(path), detail=str(message),
                           source="objective", key=str(path)))
    imports = getattr(evaluation, "imports", None)
    for module, err in (getattr(imports, "failed", None) or [])[:6]:
        if str(module).split(".")[-1] in flagged_stems:
            continue                                                       # already reported as a compile error
        out.append(Finding(f"Module fails to import: {module}", "blocker", "correctness", detail=str(err),
                           source="objective", key=str(module)))
    for module in (getattr(imports, "timed_out", None) or [])[:3]:
        out.append(Finding(f"Module hangs on import: {module}", "major", "correctness",
                           detail="Importing it did not finish (infinite loop or blocking call at import time).",
                           source="objective", key=str(module)))
    tests = getattr(evaluation, "tests", None)
    for label, suite in (("Test", getattr(tests, "candidate", None)), ("Acceptance check", getattr(tests, "acceptance", None))):
        if suite is None or getattr(suite, "skipped_reason", ""):
            continue
        if getattr(suite, "timed_out", False):
            out.append(Finding(f"{label} suite timed out", "blocker", "tests",
                               detail=f"The {label.lower()} suite did not finish within the time limit.", source="objective",
                               key=f"{label.lower()} suite"))
            continue
        for failure in (getattr(suite, "failures", None) or [])[:8]:
            out.append(Finding(f"{label} failing: {failure.name}", "major", "tests", file=_test_file(failure.name),
                               detail=(failure.message or "").strip()[:600], source="objective", key=failure.name))
        if label == "Test" and getattr(suite, "no_tests", False):
            out.append(Finding("The project has no tests", "minor", "tests",
                               detail="No test file was found or collected.", source="objective", key="no tests"))
    for run in getattr(evaluation, "runs", None) or []:
        if getattr(run, "error", "") and not getattr(run, "runs", 0):
            continue                                                       # could not be launched here: not a verdict
        if not getattr(run, "completed", True):
            why = "timed out" if run.timed_out else ("produced the wrong output" if run.output_ok is False
                                                      else f"exited with code {run.exit_code}")
            tail = (run.stderr_tail or run.stdout_tail or "").strip()[-500:]
            out.append(Finding(f"Run '{run.name}' failed: {why}", "blocker", "correctness", detail=tail,
                               source="objective", key=f"run '{run.name}'"))
    shown = 0
    for issue in getattr(static, "issues", None) or []:
        severity = getattr(issue, "severity", "")
        if severity not in ("high", "medium") or shown >= 6:
            continue
        shown += 1
        out.append(Finding(f"{issue.tool}: {issue.message}", "major" if severity == "high" else "minor",
                           "security" if issue.tool == "bandit" else "correctness", file=str(issue.path),
                           line=issue.line or None, detail=f"{issue.code} {issue.message}".strip(), source="objective",
                           key=f"{issue.path}:{issue.line}"))
    return out[:MAX_OBJECTIVE_FINDINGS]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

@dataclass
class ReviewResult:
    passed: bool
    findings: list = field(default_factory=list)          # [Finding], worst first
    summary: str = ""
    verdict: Optional[str] = None                         # what the model said: pass | fail | None (unusable)
    requirements: list = field(default_factory=list)      # [{"requirement", "status", "note"}]
    resolved: list = field(default_factory=list)          # findings from the previous round that are gone
    stuck: bool = False
    reliable: bool = True                                 # False: the model's review could not be used
    warnings: list = field(default_factory=list)
    reviewer_model: str = ""
    coder_model: str = ""
    iteration: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def blocking(self) -> list:
        return [f for f in self.findings if f.severity in DEFAULT_FAIL_ON]

    @property
    def blockers(self) -> list:
        return [f for f in self.findings if f.severity == "blocker"]

    @property
    def severity_score(self) -> float:
        return severity_score(self.findings)

    def improved_over(self, previous: "ReviewResult") -> bool:
        """Strictly less severe than the previous round: a revision that made things worse should be rolled back."""
        return self.severity_score < previous.severity_score

    def counts(self) -> dict:
        return {s: sum(1 for f in self.findings if f.severity == s) for s in SEVERITIES}

    def summary_line(self) -> str:
        c = self.counts()
        text = (f"{'PASS' if self.passed else 'FAIL'} (iteration {self.iteration}): {c['blocker']} blocker, {c['major']} major, "
                f"{c['minor']} minor, {c['nit']} nit; {len(self.resolved)} resolved")
        if self.stuck:
            text += "; STUCK on a recurring blocking finding"
        if not self.reliable:
            text += "; model review unusable (objective findings only)"
        return text

    def to_revision_prompt(self, max_chars: int = 6000, include_minor: bool = True) -> str:
        """Findings as text for CoderAgent.revise(): blocking first, concrete locations and suggestions."""
        chosen = [f for f in self.findings if f.severity in DEFAULT_FAIL_ON]
        if include_minor and len(chosen) < 8:
            chosen += [f for f in self.findings if f.severity == "minor"][: 8 - len(chosen)]
        if not chosen:
            return ""
        lines = ["Review findings to fix (worst first):"]
        for n, f in enumerate(chosen, 1):
            where = f" [{f.location}]" if f.location else ""
            lines.append(f"{n}. ({f.severity}/{f.category}){where} {f.title}")
            if f.detail:
                lines.append("   Problem: " + " ".join(f.detail.split())[:700])
            if f.suggestion:
                lines.append("   Fix: " + " ".join(f.suggestion.split())[:500])
            if f.status == "persisting":
                lines.append(f"   NOTE: this was reported in {f.rounds - 1} earlier review(s) and is still not fixed.")
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 15] + "\n...(truncated)"

    def architecture_feedback(self) -> str:
        """Design-level problems (category 'design', major or worse) for HLDAgent.revise(); '' if there are none."""
        design = [f for f in self.findings if f.category == "design" and f.severity in DEFAULT_FAIL_ON]
        if self.stuck:
            design += [f for f in self.blocking if f.status == "persisting" and f not in design]
        if not design:
            return ""
        return "\n".join(f"- ({f.severity}) {f.title}: {' '.join(f.detail.split())[:500]}"
                         + (f" Suggested direction: {' '.join(f.suggestion.split())[:300]}" if f.suggestion else "")
                         for f in design)

    def for_state(self) -> list:
        """Compact findings for StateManager.record_review(...)."""
        return [{"severity": f.severity, "category": f.category, "title": f.title, "file": f.file, "line": f.line,
                 "source": f.source, "status": f.status, "rounds": f.rounds} for f in self.findings]

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if k not in ("findings", "resolved")}
        data["findings"] = [f.to_dict() for f in self.findings]
        data["resolved"] = [f.to_dict() for f in self.resolved]
        data["summary_line"] = self.summary_line()
        return data

    @classmethod
    def from_dict(cls, raw: dict) -> "ReviewResult":
        known = set(cls.__dataclass_fields__)
        data = {k: v for k, v in raw.items() if k in known and k not in ("findings", "resolved")}
        data["findings"] = [Finding.from_dict(f) for f in raw.get("findings", [])]
        data["resolved"] = [Finding.from_dict(f) for f in raw.get("resolved", [])]
        return cls(**data)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are an independent senior code reviewer. Another engineer (a different AI model) wrote this code and you did "
    "not. Judge it against the requirement and the architecture.\n"
    "- Report only REAL problems you can point to in the code shown, with file and line. Never invent issues. Do not "
    "report style preferences or what a linter would catch unless it causes a bug.\n"
    "- Severity: blocker = does not work, crashes, gives wrong results, has a security hole, or a required feature is "
    "absent; major = a significant defect or missing important behaviour/edge case that users will hit; minor = small "
    "defect or maintainability issue; nit = trivial.\n"
    "- Prefer a few high-value findings over many. Explain the root cause and say how to fix it.\n"
    "- The objective results (tests, compile, runtime) are facts. Do not contradict them; explain their causes.\n"
    "- Reply with JSON only.")

PROMPT_TEMPLATE = """Review this implementation.

REQUIREMENT
{requirement}
{architecture}{evidence}{previous}{diff}
CODE UNDER REVIEW
{code}

Reply with ONE JSON object and nothing else:
{{"verdict": "pass" or "fail",
  "summary": "2-3 sentences: does it meet the requirement, what is the main risk",
  "requirements": [{{"requirement": "short phrase", "status": "met" or "partial" or "missing", "note": "optional"}}],
  "findings": [{{"severity": "blocker|major|minor|nit",
                "category": "correctness|requirements|security|design|tests|performance|maintainability|docs",
                "file": "path or empty", "line": 12, "title": "short", "detail": "what is wrong and why",
                "suggestion": "how to fix"}}]}}
Use verdict "pass" only if there is no blocker and no major finding. List the 3-8 most important requirements
in "requirements"."""

REPAIR_PROMPT = """Your previous reply could not be parsed as the required JSON object.

PREVIOUS REPLY
{reply}

Reply again with ONLY the JSON object (verdict, summary, requirements, findings), no prose and no markdown fence."""


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _as_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


class ReviewerAgent:
    def __init__(self, generate: GenerateFn, *, temperature: float = 0.1, context_budget_tokens: int = CONTEXT_BUDGET_TOKENS,
                 max_tokens: int = 3000, fail_on: tuple = DEFAULT_FAIL_ON, stuck_after: int = 2,
                 max_findings: int = MAX_MODEL_FINDINGS):
        """
        fail_on:      severities that make a review fail.
        stuck_after:  a blocking finding present in this many consecutive reviews marks the result `stuck`.
        """
        self.generate = generate
        self.temperature = temperature
        self.context_budget_tokens = context_budget_tokens
        self.max_tokens = max_tokens
        self.fail_on = tuple(fail_on)
        self.stuck_after = max(2, stuck_after)
        self.max_findings = max_findings

    # ---- public ------------------------------------------------------------------- #

    def review(self, requirement: str, design: Any, repo: Any, reviewer_model: str, *, coder_model: str = "",
               analysis: Any = None, evaluation: Any = None, previous: Any = None, iteration: int = 1,
               diff_text: str = "", changed_files: Optional[list] = None) -> ReviewResult:
        """Review the workspace. Raises ReviewError (empty workspace) or propagates the first call's ModelError."""
        if not (requirement or "").strip():
            raise ReviewError("empty requirement")
        files = repo.list_files()
        if not files:
            raise ReviewError("the workspace is empty: nothing to review")
        objective = objective_findings(evaluation)
        prev_findings = self._coerce_previous(previous)
        result = ReviewResult(passed=False, reviewer_model=reviewer_model, coder_model=coder_model, iteration=iteration)
        if coder_model and coder_model == reviewer_model:
            result.warnings.append("reviewer and coder are the same model: the review is not independent")

        prompt = self._prompt(requirement, design, repo, evaluation, objective, prev_findings, diff_text,
                              list(changed_files or []))
        response = self._ask(reviewer_model, prompt)                       # ModelError propagates
        self._add_usage(result, response)
        data = extract_json_object(response.text)
        if not self._usable(data):
            try:
                repair = self._ask(reviewer_model, REPAIR_PROMPT.format(reply=(response.text or "")[:6000]))
                self._add_usage(result, repair)
                data = extract_json_object(repair.text)
            except Exception as exc:                                       # keep going with objective facts only
                logger.warning("Review repair call failed: %s: %s", type(exc).__name__, exc)
                result.warnings.append(f"repair call failed: {exc}")
                data = None
        if not self._usable(data):
            result.reliable = False
            result.warnings.append("the reviewer's reply was not valid JSON: verdict based on objective findings only")
            data = {}
        result.verdict = str(data.get("verdict", "")).strip().lower() or None if data else None
        if result.verdict not in ("pass", "fail"):
            result.verdict = None
        result.summary = str(data.get("summary", "")).strip()[:800] if data else ""

        model_findings = self._normalize(data.get("findings") if data else [], repo, result)
        result.requirements, req_findings = self._requirements(data.get("requirements") if data else [], model_findings)
        merged = self._merge(objective, model_findings + req_findings)
        self._finish(result, merged, prev_findings)
        logger.info("Review by %s: %s", reviewer_model, result.summary_line())
        return result

    # ---- internals ---------------------------------------------------------------- #

    def _prompt(self, requirement: str, design: Any, repo: Any, evaluation: Any, objective: list, previous: list,
                diff_text: str, changed: list) -> str:
        architecture = ""
        if design is not None:
            digest = design.digest(3500) if hasattr(design, "digest") else str(design)[:3500]
            architecture = f"\nARCHITECTURE\n{digest}\n"
        evidence = ""
        if evaluation is not None:
            try:
                evidence = (f"\nOBJECTIVE RESULTS (facts from running the code)\n{evaluation.summary()}\n"
                            f"{evaluation.findings(2500)}\n")
            except Exception:
                evidence = ""
        prev = ""
        if previous:
            lines = [f"- [{f.severity}] {f.location or 'general'}: {f.title}" for f in previous[:15]]
            prev = ("\nPREVIOUS REVIEW FINDINGS (the code was revised since: check each against the current code; "
                    "re-report the ones that still apply, do not repeat fixed ones)\n" + "\n".join(lines) + "\n")
        diff = f"\nCHANGES SINCE THE LAST REVIEW (unified diff)\n{diff_text.strip()[:6000]}\n" if diff_text.strip() else ""
        code = self._context(repo, requirement, changed, objective)
        return PROMPT_TEMPLATE.format(requirement=requirement.strip(), architecture=architecture, evidence=evidence,
                                      previous=prev, diff=diff, code=code or "(no code could be selected)")

    def _context(self, repo: Any, query: str, changed: list, objective: list) -> str:
        try:
            index = RepoIndex()
            index.update(repo)
            focus: list = []
            for path in changed + [f.file for f in objective] + index.entry_points() + index.ranked_files(include_tests=True):
                if path and path in index and path not in focus:
                    focus.append(path)
            # the selector does not count its "=== path ===" headers: keep 5% headroom so the prompt stays in budget
            selected = ContextSelector(repo, index).select(query[:500], budget_tokens=int(self.context_budget_tokens * 0.95),
                                                           focus_files=focus[:30], include_tests=True, include_map=True)
            return selected.render()
        except Exception as exc:
            logger.warning("Review context selection failed (%s: %s)", type(exc).__name__, exc)
            return ""

    def _ask(self, model_id: str, prompt: str) -> ModelResponse:
        return self.generate(model_id, ModelRequest(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=self.max_tokens,
                                                     temperature=self.temperature, purpose="review",
                                                     metadata={"stage": "review"}))

    @staticmethod
    def _add_usage(result: ReviewResult, response: ModelResponse) -> None:
        result.input_tokens += int(getattr(response, "input_tokens", 0) or 0)
        result.output_tokens += int(getattr(response, "output_tokens", 0) or 0)
        result.cost_usd += float(getattr(response, "cost_usd", 0.0) or 0.0)
        result.latency_seconds += float(getattr(response, "latency_seconds", 0.0) or 0.0)
        result.calls += 1

    @staticmethod
    def _usable(data: Any) -> bool:
        return isinstance(data, dict) and ("findings" in data or "verdict" in data)

    @staticmethod
    def _coerce_previous(previous: Any) -> list:
        if previous is None:
            return []
        items = previous.findings if isinstance(previous, ReviewResult) else previous
        out = []
        for item in items or []:
            if isinstance(item, Finding):
                out.append(item)
            elif isinstance(item, dict) and item.get("title"):
                out.append(Finding.from_dict({k: v for k, v in item.items() if k in Finding.__dataclass_fields__}))
        return out

    def _normalize(self, raw: Any, repo: Any, result: ReviewResult) -> list:
        """Validate the model's findings: known severity/category, real files and lines, bounded size."""
        out: list = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or not str(item.get("title", "")).strip():
                continue
            severity = str(item.get("severity", "minor")).strip().lower()
            severity = SEVERITY_ALIASES.get(severity, severity)
            category = str(item.get("category", "correctness")).strip().lower()
            category = CATEGORY_ALIASES.get(category, category)
            finding = Finding(title=" ".join(str(item["title"]).split())[:160],
                              severity=severity if severity in SEVERITIES else "minor",
                              category=category if category in CATEGORIES else "correctness",
                              file=str(item.get("file", "") or "").strip().replace("\\", "/"),
                              line=_as_int(item.get("line")), detail=str(item.get("detail", "")).strip()[:900],
                              suggestion=str(item.get("suggestion", "")).strip()[:600], source="model")
            if finding.file:
                finding = self._verify_location(finding, repo, result)
            out.append(finding)
        out.sort(key=lambda f: f.rank)
        if len(out) > self.max_findings:
            result.warnings.append(f"{len(out) - self.max_findings} lower-priority findings dropped (limit {self.max_findings})")
            out = out[: self.max_findings]
        return out

    @staticmethod
    def _verify_location(finding: Finding, repo: Any, result: ReviewResult) -> Finding:
        try:
            exists = repo.exists(finding.file)
        except Exception:
            exists = False
        if not exists:
            result.warnings.append(f"finding '{finding.title}' cites {finding.file}, which is not in the workspace: "
                                   "location cleared, severity lowered")
            finding.detail = (finding.detail + f" (reviewer cited {finding.file}, which does not exist)").strip()
            finding.file, finding.line = "", None
            finding.severity = SEVERITIES[min(finding.rank + 1, len(SEVERITIES) - 1)]
            return finding
        if finding.line:
            try:
                total = len(repo.read_file(finding.file, errors="replace").splitlines())
            except Exception:
                total = 0
            if total and finding.line > total:
                finding.line = None
        return finding

    @staticmethod
    def _requirements(raw: Any, model_findings: list) -> tuple:
        """Normalise requirement coverage; a 'missing' requirement becomes a major finding unless already reported."""
        rows, extra = [], []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or not str(item.get("requirement", "")).strip():
                continue
            status = str(item.get("status", "")).strip().lower()
            row = {"requirement": " ".join(str(item["requirement"]).split())[:160],
                   "status": status if status in ("met", "partial", "missing") else "partial",
                   "note": str(item.get("note", "")).strip()[:300]}
            rows.append(row)
            if row["status"] == "met":
                continue
            probe = Finding(f"Requirement not met: {row['requirement']}",
                            "major" if row["status"] == "missing" else "minor", "requirements",
                            detail=row["note"] or ("The requirement is not implemented." if row["status"] == "missing"
                                                    else "The requirement is only partly implemented."), source="model")
            if not any(similar(probe, f, 0.4) for f in model_findings if f.category == "requirements"):
                extra.append(probe)
        return rows, extra

    @staticmethod
    def _merge(objective: list, model: list) -> list:
        """Objective findings win; a model finding that restates one is dropped (its detail is appended if richer)."""
        merged = list(objective)
        for finding in model:
            twin = next((o for o in objective if similar(o, finding, 0.4)), None)
            if twin is None:
                merged.append(finding)
            elif finding.suggestion and not twin.suggestion:
                twin.suggestion = finding.suggestion
        merged.sort(key=lambda f: (f.rank, 0 if f.source == "objective" else 1))
        return merged

    def _finish(self, result: ReviewResult, merged: list, previous: list) -> None:
        unmatched = list(previous)
        for finding in merged:
            best = next((p for p in unmatched if similar(finding, p)), None)
            if best is not None:
                unmatched.remove(best)
                finding.status, finding.rounds = "persisting", best.rounds + 1
        result.resolved = unmatched
        for n, finding in enumerate(merged, 1):
            finding.id = f"R{result.iteration}-{n}"
        result.findings = merged
        blocking = [f for f in merged if f.severity in self.fail_on]
        result.passed = not blocking
        result.stuck = any(f.rounds >= self.stuck_after for f in blocking)
        if result.verdict == "pass" and blocking:
            result.warnings.append("the reviewer said pass but findings are blocking: overruled")
        elif result.verdict == "fail" and not blocking:
            result.warnings.append("the reviewer said fail but gave no blocking finding: treated as pass")


# --------------------------------------------------------------------------- #
# Self-test:  python -m agents.reviewer     (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace as NS

    from agents.hld import HLDResult, PlannedFile
    from context.repository import Repository
    from core.logger import setup_logging
    from models.base import FatalModelError, TransientModelError

    setup_logging("ERROR")
    tmp = tempfile.TemporaryDirectory()
    counter = [0]

    def new_repo(files):
        counter[0] += 1
        repo = Repository(Path(tmp.name) / f"w{counter[0]}")
        repo.write_files(files)
        return repo

    class Fake:
        def __init__(self, replies):
            self.replies, self.calls = list(replies), []

        def __call__(self, model_id, request):
            self.calls.append((model_id, request))
            item = self.replies.pop(0)
            if isinstance(item, Exception):
                raise item
            text = item if isinstance(item, str) else json.dumps(item)
            return ModelResponse(text=text, model_id=model_id, input_tokens=2000, output_tokens=500, cost_usd=0.02,
                                 latency_seconds=2.0, attempts=1, purpose=request.purpose)

    CODE = {"app/rules.py": "def categorise(name):\n    return 'docs' if name.endswith('.pdf') else 'other'\n",
            "app/watcher.py": "from app.rules import categorise\n\n\ndef scan(names):\n    return [categorise(n) for n in names]\n",
            "main.py": "from app.watcher import scan\n\nprint(scan(['a.pdf']))\n",
            "tests/test_rules.py": "import unittest\nfrom app.rules import categorise\n\n\nclass T(unittest.TestCase):\n"
                                   "    def test_pdf(self):\n        self.assertEqual(categorise('a.pdf'), 'docs')\n"}
    DESIGN = HLDResult(markdown="# Sorter\n\n## Overview\n\nSorts downloads.\n\n## Interfaces\n\n```python\ndef categorise(name: str) -> str\n```\n",
                       files=[PlannedFile("app/rules.py", "categorise"), PlannedFile("main.py", "entry", ["app/rules.py"])],
                       entry_point="main.py")
    REQ = "Build a desktop app that monitors my downloads and automatically categorizes them"
    repo = new_repo(CODE)

    def finding(**kw):
        base = {"severity": "minor", "category": "correctness", "file": "app/rules.py", "line": 2, "title": "t",
                "detail": "d", "suggestion": "s"}
        base.update(kw)
        return base

    PASS_REPLY = {"verdict": "pass", "summary": "Meets the requirement.",
                  "requirements": [{"requirement": "categorise files", "status": "met"}],
                  "findings": [finding(title="Extension matching is case sensitive", suggestion="lower() the name")]}

    # 1. Objective findings from a REAL evaluation of broken code
    from evaluation.evaluator import Evaluator
    from evaluation.runtime import RunSpec
    from evaluation.static_analysis import StaticAnalyzer
    from evaluation.tests import TestRunner
    from config.settings import OrchestratorSettings
    cfg = OrchestratorSettings()
    broken = new_repo({"calc.py": "def add(a, b)\n    return a + b\n", "main.py": "print('hi')\n",
                       "test_calc.py": "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n"})
    real_eval = Evaluator(cfg, static=StaticAnalyzer(cfg, commands={}), tests=TestRunner(cfg, commands={"pytest": None})).evaluate(
        broken, "backend_api", run_specs=[RunSpec(command=["python", "main.py"], name="main", expected_stdout="ok\n")])
    obj = objective_findings(real_eval)
    titles = [f.title for f in obj]
    assert any(t.startswith("File does not compile: calc.py") for t in titles) and all(f.source == "objective" for f in obj)
    assert not any("fails to import: calc" in t for t in titles)                     # not reported twice
    assert any(t.startswith("Run 'main' failed") for t in titles) and obj[0].severity == "blocker"
    fake_eval = NS(empty=False, static=NS(compile_errors=[], issues=[NS(tool="bandit", severity="high", path="app/db.py", line=9,
                                                                       code="B608", message="SQL injection"),
                                                              NS(tool="ruff", severity="low", path="x.py", line=1, code="E1", message="m")]),
                   imports=NS(failed=[("pkg.mod", "ImportError: nope")], timed_out=[]),
                   tests=NS(candidate=NS(skipped_reason="", timed_out=False, no_tests=False, failures=[
                       NS(name="tests/test_a.py::T::test_x", kind="failure", message="assert 1 == 2")]),
                       acceptance=NS(skipped_reason="", timed_out=True, no_tests=False, failures=[])), runs=[])
    of = objective_findings(fake_eval)
    by = {f.title: f for f in of}
    assert by["Test failing: tests/test_a.py::T::test_x"].file == "tests/test_a.py" and by["Module fails to import: pkg.mod"].severity == "blocker"
    assert by["Acceptance check suite timed out"].severity == "blocker" and by["bandit: SQL injection"].category == "security"
    assert len(of) == 4 and objective_findings(None) == [] and objective_findings(NS(empty=True))[0].title == "No files were produced"
    print(f"PASS 1: objective findings from a real evaluation ({len(obj)}) and from test/static/import reports ({len(of)})")

    # 2. Happy review: prompt contents, purpose, usage, pass with only a minor
    fk = Fake([PASS_REPLY])
    good_eval = Evaluator(cfg, static=StaticAnalyzer(cfg, commands={}), tests=TestRunner(cfg, commands={"pytest": None})).evaluate(repo, "backend_api")
    r = ReviewerAgent(fk).review(REQ, DESIGN, repo, "gpt", coder_model="claude", evaluation=good_eval)
    p = fk.calls[0][1].prompt
    assert r.passed and r.verdict == "pass" and r.reliable and r.calls == 1 and r.total_tokens == 2500 and not r.warnings
    assert fk.calls[0][1].purpose == "review" and fk.calls[0][1].system and fk.calls[0][0] == "gpt"
    assert REQ in p and "def categorise(name: str)" in p and "OBJECTIVE RESULTS" in p and "return 'docs' if name.endswith" in p
    assert [f.severity for f in r.findings] == ["minor"] and r.findings[0].id == "R1-1" and r.requirements[0]["status"] == "met"
    print(f"PASS 2: healthy review passes with a minor note -> {r.summary_line()}")

    # 3. Findings decide, not the verdict
    over = Fake([dict(PASS_REPLY, findings=[finding(severity="major", title="Crashes on names without extension")])])
    r3 = ReviewerAgent(over).review(REQ, DESIGN, repo, "gpt")
    assert not r3.passed and r3.verdict == "pass" and any("overruled" in w for w in r3.warnings)
    soft = Fake([dict(PASS_REPLY, verdict="fail")])
    r3b = ReviewerAgent(soft).review(REQ, DESIGN, repo, "gpt")
    assert r3b.passed and any("treated as pass" in w for w in r3b.warnings)
    print("PASS 3: a 'pass' next to a major finding is overruled; a 'fail' with nothing blocking is treated as pass")

    # 4. Objective blockers cannot be waved through; model duplicates are merged into them
    dup = Fake([{"verdict": "pass", "summary": "Looks fine.", "requirements": [], "findings": [
        finding(severity="major", file="calc.py", line=1, title="calc.py does not compile", suggestion="add the colon")]}])
    r4 = ReviewerAgent(dup).review(REQ, DESIGN, broken, "gpt", evaluation=real_eval)
    assert not r4.passed and r4.blockers and sum("calc.py" in f.title for f in r4.findings) == 1
    assert next(f for f in r4.findings if "calc.py" in f.title).suggestion == "add the colon"
    assert r4.findings[0].source == "objective" and "OBJECTIVE RESULTS" in dup.calls[0][1].prompt
    print(f"PASS 4: objective blockers force a fail; duplicate model finding merged ({len(r4.findings)} findings)")

    # 5. Normalisation: synonyms, hallucinated files, impossible lines, junk, caps
    noisy = {"verdict": "fail", "summary": "x" * 2000, "requirements": [], "findings": [
        finding(severity="Critical", category="Bug", title="Wrong label", file="app\\rules.py", line=2),
        finding(severity="high", category="weird", title="Ghost file bug", file="app/ghost.py", line=3),
        finding(severity="major", title="Far line", file="app/rules.py", line=999),
        {"title": ""}, "junk", {"severity": "low", "title": "Tiny", "file": "", "line": "x"}] +
        [finding(severity="nit", title=f"nit number {i}", file="") for i in range(20)]}
    r5 = ReviewerAgent(Fake([noisy]), max_findings=6).review(REQ, DESIGN, repo, "gpt")
    by = {f.title: f for f in r5.findings}
    assert by["Wrong label"].severity == "blocker" and by["Wrong label"].category == "correctness" and by["Wrong label"].file == "app/rules.py"
    assert by["Ghost file bug"].severity == "minor" and by["Ghost file bug"].file == "" and "does not exist" in by["Ghost file bug"].detail
    assert by["Far line"].line is None and by["Tiny"].severity == "minor" and by["Tiny"].line is None
    assert len(r5.findings) == 6 and len(r5.summary) == 800 and any("dropped" in w for w in r5.warnings)
    assert [f.rank for f in r5.findings] == sorted(f.rank for f in r5.findings)
    print("PASS 5: severities/categories normalised, hallucinated files downgraded, bad lines/junk/cap handled")

    # 6. Requirement coverage
    cov = {"verdict": "pass", "summary": "ok", "requirements": [
        {"requirement": "watch the downloads folder", "status": "missing", "note": "no watcher loop"},
        {"requirement": "categorise by type", "status": "partial"}, {"requirement": "log actions", "status": "met"},
        {"requirement": "undo moves", "status": "bogus"}], "findings": [
        finding(category="requirements", severity="major", file="", title="Requirement not met: undo moves is absent")]}
    r6 = ReviewerAgent(Fake([cov])).review(REQ, DESIGN, repo, "gpt")
    titles = [f.title for f in r6.findings]
    assert not r6.passed and "Requirement not met: watch the downloads folder" in titles
    assert next(f for f in r6.findings if "watch the downloads" in f.title).severity == "major"
    assert next(f for f in r6.findings if "categorise by type" in f.title).severity == "minor"
    assert sum("undo moves" in t for t in titles) == 1 and not any("log actions" in t for t in titles)
    assert [x["status"] for x in r6.requirements] == ["missing", "partial", "met", "partial"]
    print("PASS 6: missing requirement -> major finding, partial -> minor, met -> nothing, duplicates avoided")

    # 7. Unparseable replies: repair call, then unreliable fallback; errors
    ok_after = Fake(["Sorry, here is my review in prose: it is fine.", PASS_REPLY])
    r7 = ReviewerAgent(ok_after).review(REQ, DESIGN, repo, "gpt")
    assert r7.reliable and r7.calls == 2 and r7.passed and "PREVIOUS REPLY" in ok_after.calls[1][1].prompt and r7.total_tokens == 5000
    hopeless = Fake(["not json", "still not json"])
    r7b = ReviewerAgent(hopeless).review(REQ, DESIGN, broken, "gpt", evaluation=real_eval)
    assert not r7b.reliable and r7b.verdict is None and not r7b.passed and r7b.blockers and "objective findings only" in r7b.warnings[0]
    r7c = ReviewerAgent(Fake(["not json", "still not json"])).review(REQ, DESIGN, repo, "gpt", evaluation=good_eval)
    assert not r7c.reliable and r7c.passed and "unusable" in r7c.summary_line()           # nothing objective is wrong
    r7d = ReviewerAgent(Fake(["not json", TransientModelError("down")])).review(REQ, DESIGN, repo, "gpt")
    assert not r7d.reliable and any("repair call failed" in w for w in r7d.warnings)
    for err in (FatalModelError("bad key", model_id="gpt"), TransientModelError("down", model_id="gpt")):
        try:
            ReviewerAgent(Fake([err])).review(REQ, DESIGN, repo, "gpt")
            raise AssertionError("expected ModelError")
        except (FatalModelError, TransientModelError) as exc:
            assert exc is err
    for bad_reply in ('[1, 2]', '{"unrelated": true}', ""):
        assert not ReviewerAgent(Fake([bad_reply, bad_reply])).review(REQ, DESIGN, repo, "gpt").reliable
    print("PASS 7: invalid JSON -> one repair call, then objective-only unreliable result; first-call errors propagate")

    # 8. Trend across rounds, stuck detection, improvement test
    round1 = ReviewerAgent(Fake([{"verdict": "fail", "summary": "bugs", "requirements": [], "findings": [
        finding(severity="blocker", title="Crash when the file name has no extension"),
        finding(severity="major", file="app/watcher.py", line=4, title="scan ignores subfolders"),
        finding(severity="minor", title="Case sensitive extension match")]}])).review(REQ, DESIGN, repo, "gpt")
    round2 = ReviewerAgent(Fake([{"verdict": "fail", "summary": "better", "requirements": [], "findings": [
        finding(severity="blocker", title="Crash if file name has no extension"),                     # reworded, persists
        finding(severity="minor", file="main.py", line=3, title="Prints instead of logging")]}])).review(
        REQ, DESIGN, repo, "gpt", previous=round1, iteration=2)
    pf = next(f for f in round2.findings if f.severity == "blocker")
    assert pf.status == "persisting" and pf.rounds == 2 and round2.stuck and not round2.passed
    assert sorted(f.title for f in round2.resolved) == ["Case sensitive extension match", "scan ignores subfolders"]
    assert next(f for f in round2.findings if f.severity == "minor").status == "new" and round2.findings[0].id == "R2-1"
    assert round2.improved_over(round1) and not round1.improved_over(round2) and round1.severity_score == 12.0
    pr = Fake([PASS_REPLY])
    ReviewerAgent(pr).review(REQ, DESIGN, repo, "gpt", previous=round1.to_dict()["findings"], iteration=2,
                             diff_text="--- a/app/rules.py\n+++ b/app/rules.py\n@@ -1 +1 @@\n-x\n+y\n", changed_files=["main.py"])
    pp = pr.calls[0][1].prompt
    assert "PREVIOUS REVIEW FINDINGS" in pp and "scan ignores subfolders" in pp and "CHANGES SINCE THE LAST REVIEW" in pp
    print(f"PASS 8: findings tracked across rounds (1 persisting, {len(round2.resolved)} resolved, stuck={round2.stuck}); prompts carry history")

    # 9. Outputs for the other agents and the state manager; JSON round trip
    text = round2.to_revision_prompt()
    assert text.startswith("Review findings to fix") and "[main.py:3]" in text and "reported in 1 earlier review" in text
    assert text.index("blocker") < text.index("minor") and len(round2.to_revision_prompt(max_chars=120)) <= 120
    assert round2.architecture_feedback().startswith("- (blocker) Crash")                 # stuck blockers go to the architect
    design_review = ReviewerAgent(Fake([{"verdict": "fail", "summary": "", "requirements": [], "findings": [
        finding(severity="major", category="design", file="", title="Watcher and rules are tightly coupled",
                suggestion="introduce an interface")]}])).review(REQ, DESIGN, repo, "gpt")
    assert "tightly coupled" in design_review.architecture_feedback() and "interface" in design_review.architecture_feedback()
    assert ReviewerAgent(Fake([PASS_REPLY])).review(REQ, DESIGN, repo, "gpt").architecture_feedback() == ""
    st = round2.for_state()
    assert st[0]["severity"] == "blocker" and set(st[0]) >= {"title", "file", "line", "source", "rounds"}
    back = ReviewResult.from_dict(json.loads(json.dumps(round2.to_dict())))
    assert back.passed == round2.passed and back.stuck and [f.title for f in back.findings] == [f.title for f in round2.findings]
    assert back.findings[0].rounds == 2 and len(back.resolved) == 2
    assert Finding("t", "minor").to_dict()["status"] == "new" and ReviewResult(passed=True).to_revision_prompt() == ""
    print("PASS 9: revision prompt, architecture feedback, state records and JSON round trip")

    # 10. Same-model warning; empty workspace / requirement; context budget respected on a big repo
    same = ReviewerAgent(Fake([PASS_REPLY])).review(REQ, DESIGN, repo, "claude", coder_model="claude")
    assert any("not independent" in w for w in same.warnings) and same.passed
    for fn in (lambda: ReviewerAgent(Fake([])).review(REQ, DESIGN, Repository(Path(tmp.name) / "empty"), "m"),
               lambda: ReviewerAgent(Fake([])).review("  ", DESIGN, repo, "m")):
        try:
            fn()
            raise AssertionError("expected ReviewError")
        except ReviewError:
            pass
    big = new_repo({f"pkg/mod{i}.py": "\n".join(f"def f{i}_{j}(x):\n    return x + {j}" for j in range(120)) for i in range(25)})
    bf = Fake([PASS_REPLY])
    ReviewerAgent(bf, context_budget_tokens=3000).review(REQ, None, big, "gpt")
    code_part = bf.calls[0][1].prompt.split("CODE UNDER REVIEW\n", 1)[1].split("\n\nReply with ONE JSON", 1)[0]
    assert len(code_part) // 4 <= 3000 and "=== pkg/" in code_part and "ARCHITECTURE" not in bf.calls[0][1].prompt
    print(f"PASS 10: same-model warning, misuse raises ReviewError, big repo reviewed within budget (~{len(code_part)//4} tokens)")

    tmp.cleanup()