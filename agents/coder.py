"""
agents/coder.py

The implementation agent. Three jobs, one class:

    from agents.coder import CoderAgent, make_lite_spec

    coder = CoderAgent(registry.generate, max_output_tokens=registry.config(model_id).max_output_tokens)

    # 1. lite benchmark ("lite_generation" stage, then one candidate per model)
    spec = make_lite_spec(requirement, analysis, generate=registry.generate, model_id=analyzer_id)   # model optional
    result = coder.implement_lite(spec, Repository(candidate_dir), model_id)          # candidate_dir per model
    spec.acceptance_tests          # {"test_acceptance.py": source} for Evaluator.evaluate_many(...) (may be empty)

    # 2. full build from the architecture (file by file in dependency order, in token-sized batches)
    result = coder.implement(design, requirement, analysis, repo, model_id, on_batch=checkpoint)
    result.complete, result.missing, result.error, result.total_tokens

    # 3. revision from reviewer / evaluator findings (only changed files come back)
    result = coder.revise(repo, findings_text, model_id, requirement=requirement, design=design)
    result.snapshot                # repo.restore(result.snapshot) rolls the revision back
    result.diff_summary            # "1 added, 2 modified, 0 deleted"

How a build survives the real world:
  * Replies are parsed with context.repository (<file path="..."> blocks). A file cut off by the token limit is NEVER
    written. Files that are missing, cut off, or full of "rest of code unchanged" placeholders get ONE retry round,
    in groups of at most two files, with a prompt that says exactly what was wrong.
  * Files are written in the architecture's build order; each batch sees the files it depends on (ContextSelector),
    never the whole repository and never a conversation history.
  * Existing files are protected: a batch may not overwrite a file it was not asked to write.
  * A model failure does NOT lose progress. Everything already written stays; the result carries `error`
    (the exception: FatalModelError, RateLimitError..., or CallBudgetExceeded), `missing` and `remaining()`.
    Call implement() again (with the same or a fallback model): files that already exist are skipped (`resume=True`).
  * `max_calls` caps model calls per result so a misbehaving model cannot burn a quota in a retry loop.
  * `on_batch(batch, result)` is called after every batch so the orchestrator can checkpoint.

Agents never choose models or talk to providers: `generate(model_id, ModelRequest) -> ModelResponse`
(registry.generate) and the model id come from the scheduler.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from agents.hld import HLDResult, PlannedFile
from context.indexer import RepoIndex
from context.repository import FILE_FORMAT_INSTRUCTIONS, RepoError, parse_model_output
from context.selector import ContextSelector
from core.logger import get_logger
from core.task_analyzer import extract_json_object
from models.base import ModelRequest, ModelResponse

logger = get_logger("agents.coder")

GenerateFn = Callable[[str, ModelRequest], ModelResponse]

KIND_TOKENS = {"source": 700, "test": 500, "doc": 250}     # rough output tokens per planned file
BATCH_FILL = 0.75                                          # a batch is planned to use this share of the output limit
RETRY_GROUP = 2                                            # files per retry call
DEFAULT_MAX_OUTPUT = 4096
DEFAULT_MAX_CALLS = 60
ARCHITECTURE_CHARS = 12_000
CONTEXT_BUDGET_TOKENS = 3000
MAX_ACCEPTANCE_CHARS = 20_000
GENERIC_STEMS = {"__init__"}


class CoderError(Exception):
    """Misuse (e.g. a design with no files). Operational failures are reported in CodeResult.error instead."""


class CallBudgetExceeded(Exception):
    """The per-result model call cap was reached; progress so far is kept."""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class BatchResult:
    files: list = field(default_factory=list)            # paths requested ([] = open-ended: lite / revision)
    written: dict = field(default_factory=dict)          # path -> created | updated | unchanged
    missing: list = field(default_factory=list)          # requested but not produced (after the retry round)
    incomplete: list = field(default_factory=list)       # cut off by the token limit: not written
    placeholders: dict = field(default_factory=dict)     # path -> [(line, text)] still present after retries
    ignored: list = field(default_factory=list)          # [(path, why)] valid files the batch was not allowed to write
    rejected: list = field(default_factory=list)         # [(path, why)] unsafe / invalid
    warnings: list = field(default_factory=list)
    calls: int = 0
    tokens: int = 0
    retried: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CodeResult:
    model_id: str
    mode: str                                            # lite | full | revise
    written: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)          # planned files that still do not exist
    incomplete: list = field(default_factory=list)
    placeholders: dict = field(default_factory=dict)
    ignored: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    skipped_existing: list = field(default_factory=list) # resume: files that were already there
    warnings: list = field(default_factory=list)
    batches: list = field(default_factory=list)          # [BatchResult]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    calls: int = 0
    error: Optional[Exception] = None                    # the failure that interrupted the work, if any
    snapshot: Any = None                                 # revise: repository snapshot taken before the change
    diff_summary: str = ""                               # revise: what changed

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def changed(self) -> list:
        return sorted(p for p, status in self.written.items() if status != "unchanged")

    @property
    def complete(self) -> bool:
        """No error, nothing missing or cut off (and, for a revision, something actually changed)."""
        if self.error is not None or self.missing or self.incomplete:
            return False
        if self.mode == "revise":
            return bool(self.changed)
        return bool(self.written or self.skipped_existing)

    @property
    def clean(self) -> bool:
        """complete AND no placeholder comments left in the code."""
        return self.complete and not self.placeholders

    def remaining(self) -> list:
        return list(self.missing)

    def summary(self) -> str:
        parts = [f"{self.mode} by {self.model_id}: {len(self.written)} written"]
        if self.skipped_existing:
            parts.append(f"{len(self.skipped_existing)} already present")
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.incomplete:
            parts.append(f"{len(self.incomplete)} cut off")
        if self.placeholders:
            parts.append(f"{len(self.placeholders)} with placeholders")
        parts.append(f"{self.calls} call(s), {self.total_tokens} tokens")
        if self.error is not None:
            parts.append(f"INTERRUPTED: {type(self.error).__name__}: {self.error}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if k not in ("error", "snapshot", "batches")}
        data["error"] = f"{type(self.error).__name__}: {self.error}" if self.error is not None else None
        data["batches"] = [b.to_dict() for b in self.batches]
        data["complete"], data["clean"] = self.complete, self.clean
        return data


@dataclass
class LiteSpec:
    """The reduced task every benchmark candidate receives (plus, optionally, independent acceptance tests)."""
    title: str
    task: str
    language: str = "Python"
    interface: list = field(default_factory=list)        # [{"path", "signature", "behaviour"}]
    acceptance_tests: dict = field(default_factory=dict) # test file name -> source (Python, unittest-style)
    token_budget: int = 8000
    source: str = "heuristic"                            # heuristic | llm
    notes: list = field(default_factory=list)
    tokens_used: int = 0

    def prompt(self) -> str:
        text = self.task.strip()
        if self.interface:
            lines = ["", "REQUIRED INTERFACE (use exactly these paths and names; hidden acceptance tests import them):"]
            for item in self.interface:
                lines.append(f"- {item.get('path', '')}: {item.get('signature', '')}"
                             + (f"  -- {item['behaviour']}" if item.get("behaviour") else ""))
            text += "\n" + "\n".join(lines)
        return text

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, raw: dict) -> "LiteSpec":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Lite spec
# --------------------------------------------------------------------------- #

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def heuristic_lite_spec(requirement: str, analysis: Any = None, token_budget: int = 8000) -> LiteSpec:
    """A reduced task built without any model call: core structure + ONE feature + tests, headless and offline."""
    language = _get(analysis, "primary_language", None) or "Python"
    features = [str(f) for f in (_get(analysis, "key_features", None) or []) if str(f).strip()]
    feature = features[0] if features else "the single most important behaviour in the requirement"
    lines = max(80, min(500, int(token_budget * 0.6 / 10)))
    persistence = ("A minimal persistence layer (a single file or sqlite3 database in a temp-friendly location)."
                   if _get(analysis, "needs_database", False) else "No persistence beyond what the feature needs.")
    headless = ["Headless: no GUI toolkit, no network access, no dependence on real user folders. Put the logic in "
                "importable functions/classes; a minimal command-line entry point is enough."]
    if _get(analysis, "needs_ui", False):
        headless.append("The real project has a UI: build only the UI-independent core it would call.")
    task = (
        "Implement a SMALL, runnable version of the project below. This is a benchmark: a reduced scope, not the "
        "full product.\n\n"
        f"FULL PROJECT (context only): {requirement.strip()}\n\n"
        "BUILD ONLY:\n"
        "1. The core structure: a few modules with clearly separated responsibilities.\n"
        f"2. ONE representative feature: {feature}.\n"
        f"3. {persistence}\n"
        "4. Real tests in a tests/ folder (python unittest style so they run with unittest and pytest) covering "
        "the core behaviour.\n"
        "5. A README.md with how to run and test it.\n\n"
        "CONSTRAINTS\n"
        f"- Language: {language}. Standard library only.\n"
        "- At most 6 files in total, including tests and the README.\n"
        f"- {' '.join(headless)}\n"
        "- It must run offline and finish in a few seconds.\n"
        f"- Keep it compact: roughly {lines} lines of code in total. No placeholders or stubs.")
    return LiteSpec(title="Lite version", task=task, language=str(language), token_budget=token_budget,
                    source="heuristic")


LITE_PROMPT = """You are preparing a SMALL benchmark task used to compare coding models. All candidates will get the
same task and be scored by tests, so it must be precise.

FULL REQUIREMENT
{requirement}

REDUCED SCOPE (already decided)
{scope}

Reply with JSON only (no markdown fence):
{{"title": "short title",
  "task": "instructions for the coder, 150-300 words: the reduced scope, constraints, what to deliver",
  "interface": [{{"path": "pkg/module.py", "signature": "def f(x: str) -> str", "behaviour": "one line"}}],
  "acceptance_tests": {{"test_acceptance.py": "<complete python source>"}}}}

Rules:
- Language: {language}. Headless, offline, standard library only; finishes in seconds.
- "interface" lists the 2-5 functions/classes the acceptance tests use, with exact paths and signatures, so that
  independent implementations expose the same names.
- acceptance_tests: ONE file, python unittest.TestCase classes (they run under unittest and pytest), importing only
  names from "interface", using tempfile for any files, deterministic, under 5 seconds, testing real behaviour
  (not just that imports work). Use [] / {{}} for acceptance_tests if the language is not Python."""


def _clean_acceptance(raw: Any, notes: list) -> dict:
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for name, source in raw.items():
        base = posixpath.basename(str(name).replace("\\", "/"))
        if not re.fullmatch(r"test_[\w\-]+\.py", base):
            notes.append(f"acceptance test {name!r} dropped: file name must look like test_*.py")
            continue
        if not isinstance(source, str) or not source.strip() or len(source) > MAX_ACCEPTANCE_CHARS:
            notes.append(f"acceptance test {base} dropped: empty or too large")
            continue
        try:
            ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            notes.append(f"acceptance test {base} dropped: does not parse ({exc})")
            continue
        out[base] = source
    return out


def make_lite_spec(requirement: str, analysis: Any = None, *, generate: Optional[GenerateFn] = None,
                   model_id: Optional[str] = None, token_budget: int = 8000) -> LiteSpec:
    """
    Build the lite benchmark task. Always returns a usable spec: the heuristic one, upgraded by a model call
    (interface contract + independent acceptance tests) when `generate` and `model_id` are given and the reply is valid.
    Use a cheap model for this (the analyzer model): it is a small call.
    """
    if not (requirement or "").strip():
        raise CoderError("empty requirement")
    base = heuristic_lite_spec(requirement, analysis, token_budget)
    if generate is None or not model_id:
        return base
    language = base.language
    prompt = LITE_PROMPT.format(requirement=requirement.strip(), scope=base.task, language=language)
    try:
        response = generate(model_id, ModelRequest(
            prompt=prompt, system="You design fair, precise coding benchmarks. Output valid JSON only.",
            max_tokens=2500, temperature=0.2, purpose="lite_benchmark", metadata={"stage": "lite_generation"}))
    except Exception as exc:
        base.notes.append(f"model-written lite spec unavailable ({type(exc).__name__}: {exc}); using the built-in one")
        return base
    base.tokens_used = int(getattr(response, "input_tokens", 0) or 0) + int(getattr(response, "output_tokens", 0) or 0)
    data = extract_json_object(getattr(response, "text", "") or "")
    if not isinstance(data, dict) or not str(data.get("task", "")).strip():
        base.notes.append("model reply for the lite spec was not usable; using the built-in one")
        return base

    interface = []
    for item in data.get("interface") or []:
        if isinstance(item, dict) and str(item.get("path", "")).strip() and str(item.get("signature", "")).strip():
            interface.append({"path": str(item["path"]).strip(), "signature": str(item["signature"]).strip(),
                              "behaviour": str(item.get("behaviour", "")).strip()})
    notes = list(base.notes)
    tests = _clean_acceptance(data.get("acceptance_tests"), notes) if language.lower() == "python" else {}
    if tests and not interface:
        notes.append("acceptance tests dropped: no interface contract to hold candidates to")
        tests = {}
    return LiteSpec(title=str(data.get("title") or base.title)[:80], task=str(data["task"]).strip(), language=language,
                    interface=interface, acceptance_tests=tests, token_budget=token_budget, source="llm",
                    notes=notes, tokens_used=base.tokens_used)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

def estimate_file_tokens(planned: Optional[PlannedFile]) -> int:
    if planned is None:
        return KIND_TOKENS["source"]
    return KIND_TOKENS["test" if planned.is_test else "doc" if planned.is_doc else "source"]


def plan_batches(paths: list, files: Mapping, max_output_tokens: int) -> list:
    """Group `paths` (already in build order) into batches that should fit in one reply. Never empty batches."""
    cap = max(1, int(max_output_tokens * BATCH_FILL))
    batches, current, used = [], [], 0
    for path in paths:
        cost = estimate_file_tokens(files.get(path))
        if current and used + cost > cap:
            batches.append(current)
            current, used = [], 0
        current.append(path)
        used += cost
    if current:
        batches.append(current)
    return batches


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SYSTEM_TEMPLATE = (
    "You are an expert {language} software engineer. You write complete, correct, runnable code.\n"
    "- Follow the architecture's file layout and interfaces EXACTLY: names, signatures, data shapes. Other files are "
    "written separately and must fit together.\n"
    "- No placeholders, stubs, TODOs standing in for logic, or comments like 'rest of the code unchanged'.\n"
    "- Prefer the standard library; use third-party packages only if the architecture lists them.\n"
    "- Handle errors explicitly, write short docstrings, keep functions small, separate logic from I/O so it is testable.\n"
    "- Output only the requested files in the requested format; explanations go outside the file blocks.")


def retry_note(batch: BatchResult, group: list) -> str:
    lines = ["RETRY: your previous reply had problems with these files:"]
    for path in group:
        if path in batch.incomplete:
            lines.append(f"- {path} was cut off (the reply hit the length limit), so it was discarded")
        elif path in batch.placeholders:
            where = ", ".join(str(n) for n, _t in batch.placeholders[path][:3])
            lines.append(f"- {path} contained placeholder comments (lines {where}): write the real code")
        else:
            lines.append(f"- {path} was missing from your reply")
    lines.append("Write ONLY the files listed below, each COMPLETE. Be compact (short comments, no unnecessary code) "
                 "so each file fits in one reply.")
    return "\n".join(lines)


def _file_lines(paths: list, files: Mapping) -> str:
    out = []
    for path in paths:
        planned = files.get(path)
        purpose = f": {planned.purpose}" if planned is not None and planned.purpose else ":"
        uses = f" [uses: {', '.join(planned.depends_on)}]" if planned is not None and planned.depends_on else ""
        out.append(f"- {path}{purpose}{uses}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

@dataclass
class _Outcome:
    written: dict = field(default_factory=dict)
    ignored: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    incomplete: list = field(default_factory=list)
    placeholders: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


class CoderAgent:
    def __init__(self, generate: GenerateFn, *, max_output_tokens: int = DEFAULT_MAX_OUTPUT, temperature: float = 0.2,
                 max_calls: int = DEFAULT_MAX_CALLS, architecture_chars: int = ARCHITECTURE_CHARS,
                 context_budget_tokens: int = CONTEXT_BUDGET_TOKENS):
        """
        max_output_tokens: the coding model's reply limit (registry.config(id).max_output_tokens); batches are sized to fit.
        max_calls:         hard cap on model calls per returned result (a runaway guard, not a target).
        """
        self.generate = generate
        self.max_output_tokens = max(500, int(max_output_tokens))
        self.temperature = temperature
        self.max_calls = max(1, int(max_calls))
        self.architecture_chars = architecture_chars
        self.context_budget_tokens = context_budget_tokens

    # ---- 1. lite benchmark ----------------------------------------------------------- #

    def implement_lite(self, spec: LiteSpec, repo: Any, model_id: str, *, max_tokens: Optional[int] = None) -> CodeResult:
        """Build the lite task into `repo` (one workspace per candidate). Never raises for model failures."""
        result = CodeResult(model_id=model_id, mode="lite")
        system = SYSTEM_TEMPLATE.format(language=spec.language)
        budget = max_tokens or max(1500, min(self.max_output_tokens, int(spec.token_budget * 0.75)))

        def make_prompt(group: list, note: str) -> str:
            parts = [spec.prompt()]
            if note:
                parts.append(note)
            if group:
                parts.append("WRITE THESE FILES NOW\n" + "\n".join(f"- {p}:" for p in group))
            parts.append(FILE_FORMAT_INSTRUCTIONS)
            return "\n\n".join(parts)

        batch = self._build_batch(repo, model_id, result, [], make_prompt, system=system, purpose="lite_benchmark",
                                  max_tokens=budget, protect_existing=False)
        self._merge(result, batch)
        if not result.written and not result.error:
            result.warnings.append("the model produced no usable files")
        logger.info(result.summary())
        return result

    # ---- 2. full build --------------------------------------------------------------- #

    def implement(self, design: HLDResult, requirement: str, analysis: Any, repo: Any, model_id: str, *,
                  resume: bool = True, redo: Optional[list] = None, max_tokens: Optional[int] = None,
                  on_batch: Optional[Callable[[BatchResult, CodeResult], None]] = None) -> CodeResult:
        """
        Build every planned file in the design's build order. resume=True skips files that already exist
        (so a crashed / quota-interrupted build continues where it stopped); `redo` forces specific files to be rewritten.
        Model failures are returned in result.error with progress kept; only misuse raises CoderError.
        """
        order = list(design.build_order())
        if not order:
            raise CoderError("the design lists no files to build")
        by_path = {f.path: f for f in design.files}
        redo_set = set(redo or [])
        result = CodeResult(model_id=model_id, mode="full")
        language = _get(analysis, "primary_language", None) or "Python"
        system = SYSTEM_TEMPLATE.format(language=language)
        budget = min(max_tokens or self.max_output_tokens, self.max_output_tokens)

        todo = []
        for path in order:
            if resume and path not in redo_set and self._exists_ok(repo, path):
                result.skipped_existing.append(path)
            else:
                todo.append(path)
        architecture = (design.markdown if len(design.markdown) <= self.architecture_chars
                        else design.digest(self.architecture_chars))

        for paths in plan_batches(todo, by_path, budget):
            def make_prompt(group: list, note: str, _paths=paths) -> str:
                wanted = group or _paths
                focus = [d for p in wanted for d in (by_path[p].depends_on if p in by_path else [])]
                query = " ".join(f"{p} {by_path[p].purpose}" if p in by_path else p for p in wanted)
                context = self._context(repo, query, focus, include_tests=any(
                    by_path[p].is_test for p in wanted if p in by_path))
                parts = [f"PROJECT REQUIREMENT\n{requirement.strip()}", f"ARCHITECTURE\n{architecture.strip()}",
                         "ALREADY IMPLEMENTED (read-only context: import from these, do not rewrite them)\n"
                         + (context or "(nothing yet)")]
                if note:
                    parts.append(note)
                parts.append("WRITE THESE FILES NOW\n" + _file_lines(wanted, by_path))
                parts.append(FILE_FORMAT_INSTRUCTIONS + "\nWrite ONLY the files listed above (plus empty package "
                             "__init__.py files if the language needs them).")
                return "\n\n".join(parts)

            batch = self._build_batch(repo, model_id, result, paths, make_prompt, system=system, purpose="coding",
                                      max_tokens=budget, protect_existing=True)
            self._merge(result, batch)
            if on_batch is not None:
                try:
                    on_batch(batch, result)
                except Exception as exc:                          # bookkeeping must never crash a build
                    logger.warning("on_batch callback failed: %s", exc)
            if result.error is not None:
                break
        result.missing = [p for p in order if not self._exists_ok(repo, p)]
        logger.info(result.summary())
        return result

    # ---- 3. revision ----------------------------------------------------------------- #

    def revise(self, repo: Any, findings: str, model_id: str, *, requirement: str = "", design: Optional[HLDResult] = None,
               analysis: Any = None, focus_files: Optional[list] = None, max_tokens: Optional[int] = None) -> CodeResult:
        """
        Fix the problems in `findings` (reviewer text, Evaluation.findings(), test failures). Only changed or new files
        come back. result.snapshot rolls the workspace back; result.diff_summary says what changed.
        """
        result = CodeResult(model_id=model_id, mode="revise")
        if not (findings or "").strip():
            raise CoderError("no findings to revise from")
        result.snapshot = repo.snapshot("before-revision")
        language = _get(analysis, "primary_language", None) or "Python"
        system = SYSTEM_TEMPLATE.format(language=language)
        focus = [p for p in (focus_files or []) if repo.exists(p)] or self.files_mentioned(repo, findings)
        architecture = ""
        if design is not None:
            architecture = (design.markdown if len(design.markdown) <= self.architecture_chars
                            else design.digest(self.architecture_chars))
        budget = min(max_tokens or self.max_output_tokens, self.max_output_tokens)

        def make_prompt(group: list, note: str) -> str:
            wanted_focus = group or focus
            context = self._context(repo, findings[:500], wanted_focus, include_tests=True,
                                    budget=self.context_budget_tokens * 2)
            parts = []
            if requirement.strip():
                parts.append(f"PROJECT REQUIREMENT\n{requirement.strip()}")
            if architecture:
                parts.append(f"ARCHITECTURE\n{architecture.strip()}")
            parts.append(f"FINDINGS TO FIX\n{findings.strip()[:8000]}")
            parts.append("CURRENT CODE (relevant files)\n" + (context or "(workspace is empty)"))
            if note:
                parts.append(note)
            parts.append("Fix every finding with the smallest change that is correct. Do not change public interfaces "
                         "defined in the architecture unless a finding requires it, and do not touch unrelated files. "
                         "Return ONLY the files you change or add, each COMPLETE.")
            if group:
                parts.append("WRITE THESE FILES NOW\n" + "\n".join(f"- {p}:" for p in group))
            parts.append(FILE_FORMAT_INSTRUCTIONS)
            return "\n\n".join(parts)

        batch = self._build_batch(repo, model_id, result, [], make_prompt, system=system, purpose="coding",
                                  max_tokens=budget, protect_existing=False)
        self._merge(result, batch)
        try:
            result.diff_summary = repo.diff(result.snapshot).summary()
        except Exception as exc:
            logger.warning("Could not summarise the revision diff: %s", exc)
        if not result.error:
            if not result.written:
                result.warnings.append("the model returned no usable files: nothing was changed")
            elif not result.changed:
                result.warnings.append("the model returned the files unchanged: nothing was fixed")
        logger.info(result.summary())
        return result

    @staticmethod
    def files_mentioned(repo: Any, text: str, limit: int = 8) -> list:
        """Workspace files named in `text` (full path, file name, or a distinctive module name), in order of first mention."""
        hits: list = []
        for path in repo.list_files():
            name = posixpath.basename(path)
            stem = posixpath.splitext(name)[0]
            positions = [text.find(path)] if path in text else []
            if name in text:
                positions.append(text.find(name))
            if stem not in GENERIC_STEMS and len(stem) >= 3:
                m = re.search(rf"(?<![\w]){re.escape(stem)}(?![\w])", text)
                if m:
                    positions.append(m.start())
            positions = [p for p in positions if p >= 0]
            if positions:
                hits.append((min(positions), path))
        return [p for _pos, p in sorted(hits)][:limit]

    # ---- batch machinery ------------------------------------------------------------- #

    def _build_batch(self, repo: Any, model_id: str, result: CodeResult, paths: list, make_prompt: Callable,
                     *, system: str, purpose: str, max_tokens: int, protect_existing: bool) -> BatchResult:
        batch = BatchResult(files=list(paths))
        tokens_before, calls_before = result.total_tokens, result.calls
        allowed = set(paths) if paths else None

        def finish() -> BatchResult:
            batch.calls = result.calls - calls_before
            batch.tokens = result.total_tokens - tokens_before
            if result.error is not None:
                batch.error = f"{type(result.error).__name__}: {result.error}"
            batch.missing = [p for p in paths if p not in batch.written or p in batch.incomplete]
            return batch

        text = self._call(model_id, result, make_prompt([], ""), system, max_tokens, purpose)
        if text is None:
            return finish()
        self._absorb(batch, self._apply(repo, text, allowed, protect_existing, allow_extra=True))

        todo = [p for p in paths if p not in batch.written]
        todo += [p for p in list(batch.incomplete) + list(batch.placeholders) if p not in todo]
        for start in range(0, len(todo), RETRY_GROUP):
            group = todo[start:start + RETRY_GROUP]
            reply = self._call(model_id, result, make_prompt(group, retry_note(batch, group)), system, max_tokens, purpose)
            if reply is None:
                break
            batch.retried += [p for p in group if p not in batch.retried]
            self._absorb(batch, self._apply(repo, reply, set(group), False, allow_extra=False), group=group)
        return finish()

    def _call(self, model_id: str, result: CodeResult, prompt: str, system: str, max_tokens: int,
              purpose: str) -> Optional[str]:
        if result.calls >= self.max_calls:
            result.error = CallBudgetExceeded(f"stopped after {self.max_calls} model calls")
            return None
        request = ModelRequest(prompt=prompt, system=system, max_tokens=max_tokens, temperature=self.temperature,
                               purpose=purpose, metadata={"stage": purpose, "call": result.calls + 1})
        try:
            response = self.generate(model_id, request)
        except Exception as exc:                                  # progress is kept; the orchestrator decides what next
            logger.warning("Model call to %s failed: %s: %s", model_id, type(exc).__name__, exc)
            result.error = exc
            return None
        result.input_tokens += int(getattr(response, "input_tokens", 0) or 0)
        result.output_tokens += int(getattr(response, "output_tokens", 0) or 0)
        result.cost_usd += float(getattr(response, "cost_usd", 0.0) or 0.0)
        result.latency_seconds += float(getattr(response, "latency_seconds", 0.0) or 0.0)
        result.calls += 1
        return getattr(response, "text", "") or ""

    @staticmethod
    def _apply(repo: Any, text: str, allowed: Optional[set], protect_existing: bool, allow_extra: bool) -> _Outcome:
        """
        Write the valid files of a reply.
          allowed=None            any path may be written (lite, revision); existing files are protected only if asked
          allowed={...}           those paths are written; other paths only if allow_extra and they do not exist yet
        """
        parsed = parse_model_output(text)
        out = _Outcome(rejected=list(parsed.rejected), incomplete=list(parsed.incomplete), warnings=list(parsed.warnings))
        for path, content in parsed.files.items():
            if allowed is not None and path not in allowed:
                if not allow_extra:
                    out.ignored.append((path, "not one of the requested files"))
                    continue
                if repo.exists(path):
                    out.ignored.append((path, "exists already and was not requested: not overwritten"))
                    continue
            elif allowed is None and protect_existing and repo.exists(path):
                out.ignored.append((path, "exists already: not overwritten"))
                continue
            try:
                out.written[path] = repo.write_file(path, content)
            except RepoError as exc:
                out.rejected.append((path, str(exc)))
        out.placeholders = {p: v for p, v in parsed.placeholders.items() if p in out.written}
        return out

    @staticmethod
    def _absorb(batch: BatchResult, out: _Outcome, group: Optional[list] = None) -> None:
        batch.written.update(out.written)
        batch.ignored += out.ignored
        batch.rejected += out.rejected
        batch.warnings += out.warnings
        if group is None:
            batch.incomplete = list(out.incomplete)
            batch.placeholders = dict(out.placeholders)
            return
        for path in group:                                        # a retry replaces what we knew about these files
            batch.incomplete = [p for p in batch.incomplete if p != path]
            batch.placeholders.pop(path, None)
        batch.incomplete += [p for p in out.incomplete if p in group and p not in batch.incomplete]
        batch.placeholders.update({p: v for p, v in out.placeholders.items() if p in group})

    @staticmethod
    def _merge(result: CodeResult, batch: BatchResult) -> None:
        result.batches.append(batch)
        result.written.update(batch.written)
        result.ignored += batch.ignored
        result.rejected += batch.rejected
        result.warnings += batch.warnings
        for path in batch.written:                                # a later success clears an earlier problem
            result.incomplete = [p for p in result.incomplete if p != path]
        result.incomplete += [p for p in batch.incomplete if p not in result.incomplete]
        for path in batch.written:
            result.placeholders.pop(path, None)
        result.placeholders.update(batch.placeholders)
        result.missing = [p for p in result.missing if p not in batch.written] + \
            [p for p in batch.missing if p not in result.missing]

    @staticmethod
    def _exists_ok(repo: Any, path: str) -> bool:
        try:
            if not repo.exists(path):
                return False
            return bool(repo.read_file(path, errors="replace").strip()) or path.endswith("__init__.py")
        except Exception:
            return False

    def _context(self, repo: Any, query: str, focus: list, include_tests: bool = False,
                 budget: Optional[int] = None) -> str:
        """Relevant already-written code for a prompt (never the whole repository). Empty string if there is none."""
        try:
            if not repo.list_files():
                return ""
            index = RepoIndex()
            index.update(repo)
            selected = ContextSelector(repo, index).select(
                query, budget_tokens=budget or self.context_budget_tokens, focus_files=focus,
                include_tests=include_tests, include_map=True)
            return selected.render()
        except Exception as exc:
            logger.warning("Context selection failed (%s: %s); continuing without it", type(exc).__name__, exc)
            return ""


# --------------------------------------------------------------------------- #
# Self-test:  python -m agents.coder     (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from agents.hld import HLDResult as _HLD, PlannedFile as _PF
    from context.repository import Repository
    from core.logger import setup_logging
    from models.base import FatalModelError, TransientModelError

    setup_logging("ERROR")
    tmp = tempfile.TemporaryDirectory()
    counter = [0]

    def new_repo(files=None):
        counter[0] += 1
        repo = Repository(Path(tmp.name) / f"w{counter[0]}")
        if files:
            repo.write_files(files)
        return repo

    def requested(prompt):
        """Paths listed under 'WRITE THESE FILES NOW' in a prompt."""
        if "WRITE THESE FILES NOW" not in prompt:
            return []
        block = prompt.split("WRITE THESE FILES NOW\n", 1)[1].split("\n\n", 1)[0]
        return re.findall(r"^- ([\w./\-]+\.\w+):", block, re.M)

    PLACEHOLDER = "def categorise(name):\n    # ... rest of the code unchanged\n    pass\n"
    CODE = {"app/rules.py": "def categorise(name):\n    return 'docs' if name.endswith('.pdf') else 'other'\n",
            "app/watcher.py": "from app.rules import categorise\n\n\ndef scan(names):\n    return [categorise(n) for n in names]\n",
            "main.py": "from app.watcher import scan\n\nprint(scan(['a.pdf']))\n",
            "tests/test_rules.py": "import unittest\nfrom app.rules import categorise\n\n\nclass T(unittest.TestCase):\n"
                                   "    def test_pdf(self):\n        self.assertEqual(categorise('a.pdf'), 'docs')\n",
            "README.md": "# Sorter\n\nRun `python main.py`.\n"}

    class Oracle:
        """Fake model: scripted replies first, then it writes whatever files the prompt asks for."""
        def __init__(self, files=None, script=None):
            self.files, self.script, self.calls = dict(CODE, **(files or {})), list(script or []), []

        def __call__(self, model_id, request):
            self.calls.append((model_id, request))
            item = self.script.pop(0) if self.script else "\n".join(
                f'<file path="{p}">\n{self.files.get(p, "VALUE = 1" + chr(10))}</file>' for p in requested(request.prompt))
            if isinstance(item, Exception):
                raise item
            return ModelResponse(text=item, model_id=model_id, input_tokens=800, output_tokens=400, cost_usd=0.005,
                                 latency_seconds=1.0, attempts=1, purpose=request.purpose)

    def block(path, content):
        return f'<file path="{path}">\n{content}</file>\n'

    DESIGN = _HLD(markdown="# Sorter\n\n## Overview\n\nSorts downloads.\n\n## Interfaces\n\n```python\ndef categorise(name: str) -> str\n```\n",
                  files=[_PF("main.py", "entry point", ["app/watcher.py"]), _PF("app/watcher.py", "scan names", ["app/rules.py"]),
                         _PF("app/rules.py", "categorise one name"), _PF("tests/test_rules.py", "rule tests", ["app/rules.py"]),
                         _PF("README.md", "usage")], entry_point="main.py", run_command="python main.py")
    ORDER = ["app/rules.py", "app/watcher.py", "main.py", "tests/test_rules.py", "README.md"]
    ANALYSIS = {"primary_language": "Python", "key_features": ["watch downloads", "categorise"], "needs_database": False,
                "needs_ui": True}
    REQ = "Build a desktop app that monitors my downloads and automatically categorizes them"

    # 1. Heuristic lite spec
    spec = make_lite_spec(REQ, ANALYSIS)
    assert spec.source == "heuristic" and "watch downloads" in spec.task and REQ in spec.task and "Headless" in spec.task
    assert "UI-independent core" in spec.task and "roughly 480 lines" in spec.task and not spec.acceptance_tests
    assert "sqlite3" not in spec.task and "sqlite3" in heuristic_lite_spec("x", {"needs_database": True}).task
    try:
        make_lite_spec("  ")
        raise AssertionError("expected CoderError")
    except CoderError:
        pass
    print("PASS 1: heuristic lite spec (reduced scope, one feature, headless, size hint) without any model call")

    # 2. Model-written lite spec: interface + acceptance tests validated; bad parts dropped; fallbacks
    good_json = json.dumps({"title": "Sorter lite", "task": "Build categorise(name).",
                            "interface": [{"path": "app/rules.py", "signature": "def categorise(name: str) -> str",
                                           "behaviour": "pdf -> docs"}, {"path": "", "signature": "x"}],
                            "acceptance_tests": {"sub/test_acc.py": "import unittest\nclass A(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(True)\n",
                                                 "notatest.py": "x = 1\n", "test_bad.py": "def (:\n"}})
    ora = Oracle(script=["Here you go:\n```json\n" + good_json + "\n```"])
    s2 = make_lite_spec(REQ, ANALYSIS, generate=ora, model_id="local")
    assert s2.source == "llm" and list(s2.acceptance_tests) == ["test_acc.py"] and len(s2.interface) == 1
    assert "app/rules.py: def categorise" in s2.prompt() and s2.tokens_used == 1200 and len(s2.notes) == 2
    assert ora.calls[0][1].purpose == "lite_benchmark"
    s2b = make_lite_spec(REQ, {"primary_language": "Go"}, generate=Oracle(script=[good_json]), model_id="local")
    assert s2b.acceptance_tests == {} and s2b.source == "llm"                        # non-Python: no Python tests
    for bad in (TransientModelError("down"), "not json at all", json.dumps({"task": ""})):
        s = make_lite_spec(REQ, ANALYSIS, generate=Oracle(script=[bad]), model_id="local")
        assert s.source == "heuristic" and s.notes and not s.acceptance_tests
    no_iface = json.dumps({"task": "t", "acceptance_tests": {"test_x.py": "x = 1\n"}})
    assert make_lite_spec(REQ, ANALYSIS, generate=Oracle(script=[no_iface]), model_id="l").acceptance_tests == {}
    assert LiteSpec.from_dict(json.loads(json.dumps(s2.to_dict()))).acceptance_tests == s2.acceptance_tests
    print("PASS 2: model lite spec validated (bad tests/interface dropped, fallbacks to heuristic, JSON round trip)")

    # 3. implement_lite: happy path, prompt contents, token budget
    ora = Oracle(script=[block("calc.py", "def f():\n    return 1\n") + block("tests/test_calc.py", "import calc\n") + "Done."])
    repo = new_repo()
    r3 = CoderAgent(ora).implement_lite(spec, repo, "gpt")
    assert r3.complete and r3.clean and sorted(r3.written) == ["calc.py", "tests/test_calc.py"] and r3.calls == 1
    req3 = ora.calls[0][1]
    assert req3.purpose == "lite_benchmark" and req3.max_tokens == 4096 and spec.task in req3.prompt
    assert "<file path=" in req3.prompt and "Language" not in req3.system and "Python" in req3.system
    assert r3.total_tokens == 1200 and r3.cost_usd == 0.005 and r3.summary().startswith("lite by gpt")
    print(f"PASS 3: lite candidate built -> {r3.summary()}")

    # 4. Truncated / placeholder files in a lite reply -> one retry round naming exactly what was wrong
    cut = block("a.py", "x = 1\n") + '<file path="b.py">\ndef broken(:\n    ret'
    ora = Oracle(script=[cut], files={"b.py": "def ok():\n    return 2\n"})
    r4 = CoderAgent(ora).implement_lite(spec, new_repo(), "gpt")
    assert r4.complete and sorted(r4.written) == ["a.py", "b.py"] and r4.calls == 2 and r4.batches[0].retried == ["b.py"]
    retry_prompt = ora.calls[1][1].prompt
    assert "b.py was cut off" in retry_prompt and "WRITE THESE FILES NOW\n- b.py:" in retry_prompt
    ora = Oracle(script=[block("a.py", PLACEHOLDER)], files={"a.py": PLACEHOLDER})
    r4b = CoderAgent(ora).implement_lite(spec, new_repo(), "gpt")
    assert r4b.complete and not r4b.clean and "a.py" in r4b.placeholders and r4b.calls == 2        # retried once, still flagged
    assert "placeholder comments" in ora.calls[1][1].prompt
    print("PASS 4: cut-off and placeholder files retried once with a precise prompt; leftovers are reported, not hidden")

    # 5. Batch planning
    files = {f.path: f for f in DESIGN.files}
    assert sum(map(len, plan_batches(ORDER, files, 4096))) == 5
    tiny = plan_batches(ORDER, files, 1400)
    assert tiny == [["app/rules.py"], ["app/watcher.py"], ["main.py"], ["tests/test_rules.py", "README.md"]]
    assert plan_batches(["a.py"], {}, 100) == [["a.py"]] and plan_batches([], {}, 4096) == []
    assert sum(len(b) for b in plan_batches(ORDER, files, 99999)) == 5 and len(plan_batches(ORDER, files, 99999)) == 1
    print("PASS 5: batches sized to the output limit, order preserved, never empty")

    # 6. Full build: dependency order, context from earlier batches, protection of existing files
    ora = Oracle()
    repo = new_repo()
    coder = CoderAgent(ora, max_output_tokens=1400)
    seen = []
    r6 = coder.implement(DESIGN, REQ, ANALYSIS, repo, "claude", on_batch=lambda b, r: seen.append(list(b.files)))
    assert r6.complete and r6.clean and not r6.missing and sorted(r6.written) == sorted(ORDER) and r6.calls == 4
    assert seen == tiny and sorted(repo.list_files()) == sorted(ORDER)
    p2 = ora.calls[1][1].prompt                                       # the watcher batch must see rules.py
    assert "ALREADY IMPLEMENTED" in p2 and "def categorise" in p2 and "## Interfaces" in p2 and REQ in p2
    assert "- app/watcher.py: scan names [uses: app/rules.py]" in p2 and ora.calls[0][1].purpose == "coding"
    assert "(nothing yet)" in ora.calls[0][1].prompt
    # protection: a batch that rewrites an existing file and adds an unrequested one
    repo7 = new_repo({"app/rules.py": "ORIGINAL = True\n"})
    ora7 = Oracle(script=[block("app/watcher.py", "W = 1\n") + block("app/rules.py", "CLOBBERED = True\n") +
                          block("app/__init__.py", "") + block("../evil.py", "x = 1\n")])
    d7 = _HLD(markdown=DESIGN.markdown, files=[_PF("app/watcher.py", "w")])
    r7 = CoderAgent(ora7).implement(d7, REQ, ANALYSIS, repo7, "m", resume=False)
    assert repo7.read_file("app/rules.py") == "ORIGINAL = True\n" and repo7.exists("app/__init__.py")
    assert ("app/rules.py" in dict(r7.ignored)) and any("evil" in p for p, _ in r7.rejected) and r7.complete
    print(f"PASS 6: build in dependency order with context from earlier files; existing files protected ({r6.calls} calls)")

    # 7. Missing + cut-off + placeholder files in a full batch -> grouped retries
    first = block("app/rules.py", PLACEHOLDER) + block("app/watcher.py", CODE["app/watcher.py"]) + \
        '<file path="main.py">\nfrom app.watcher import sc'
    ora = Oracle(script=[first], files={"app/rules.py": PLACEHOLDER})
    r8 = CoderAgent(ora, max_output_tokens=8000).implement(DESIGN, REQ, ANALYSIS, new_repo(), "m")
    assert len(r8.batches) == 1 and r8.calls == 3 and r8.complete and not r8.clean and set(r8.placeholders) == {"app/rules.py"}
    assert [requested(c[1].prompt) for c in ora.calls[1:]] == [["main.py", "tests/test_rules.py"], ["README.md", "app/rules.py"]]
    assert "main.py was cut off" in ora.calls[1][1].prompt and "tests/test_rules.py was missing" in ora.calls[1][1].prompt
    print(f"PASS 7: missing/cut-off/placeholder files retried in groups of 2 ({r8.calls} calls); leftover placeholders flagged")

    # 8. Interruption keeps progress; resume continues only what is missing
    repo = new_repo()
    ora = Oracle(script=[block("app/rules.py", CODE["app/rules.py"]), FatalModelError("quota gone", model_id="claude")])
    r9 = CoderAgent(ora, max_output_tokens=1400).implement(DESIGN, REQ, ANALYSIS, repo, "claude")
    assert isinstance(r9.error, FatalModelError) and not r9.complete and r9.written == {"app/rules.py": "created"}
    assert r9.missing == ORDER[1:] and r9.remaining() == ORDER[1:] and "INTERRUPTED" in r9.summary()
    ora2 = Oracle()
    r10 = CoderAgent(ora2, max_output_tokens=1400).implement(DESIGN, REQ, ANALYSIS, repo, "gpt")
    assert r10.complete and r10.skipped_existing == ["app/rules.py"] and r10.calls == 3 and not r10.missing
    assert all("app/rules.py:" not in c[1].prompt.split("WRITE THESE FILES NOW")[1] for c in ora2.calls)
    r11 = CoderAgent(Oracle(), max_output_tokens=1400).implement(DESIGN, REQ, ANALYSIS, repo, "gpt", redo=["app/rules.py"])
    assert r11.calls == 1 and r11.batches[0].files == ["app/rules.py"] and r11.complete
    r12 = CoderAgent(Oracle(), max_output_tokens=1400).implement(DESIGN, REQ, ANALYSIS, repo, "gpt")
    assert r12.calls == 0 and r12.complete and len(r12.skipped_existing) == 5
    print(f"PASS 8: interrupted build kept its progress; resume wrote only the {r10.calls} remaining batches; redo/no-op work")

    # 9. Call cap protects the quota
    ora = Oracle()
    r13 = CoderAgent(ora, max_output_tokens=1400, max_calls=2).implement(DESIGN, REQ, ANALYSIS, new_repo(), "m")
    assert isinstance(r13.error, CallBudgetExceeded) and r13.calls == 2 and len(r13.written) == 2 and len(r13.missing) == 3
    print("PASS 9: max_calls stops a runaway build and reports what is missing")

    # 10. Revision: focus from findings, only changed files, snapshot rollback, diff summary
    repo = new_repo(CODE)
    findings = "Tests:\n  - failure: test_pdf (tests/test_rules.py): 'docs' != 'documents'\nrules.py returns the wrong label."
    ora = Oracle(script=[block("app/rules.py", "def categorise(name):\n    return 'documents'\n") + block("app/extra.py", "E = 1\n")])
    r14 = CoderAgent(ora).revise(repo, findings, "gpt", requirement=REQ, design=DESIGN, analysis=ANALYSIS)
    prompt = ora.calls[0][1].prompt
    assert r14.complete and r14.changed == ["app/extra.py", "app/rules.py"] and "1 added, 1 modified" in r14.diff_summary
    assert "FINDINGS TO FIX" in prompt and "'docs' != 'documents'" in prompt and "return 'docs' if name.endswith" in prompt
    assert "## Interfaces" in prompt and "tests/test_rules.py" in prompt and ora.calls[0][1].purpose == "coding"
    assert "documents" in repo.read_file("app/rules.py")
    repo.restore(r14.snapshot)
    assert repo.read_file("app/rules.py") == CODE["app/rules.py"] and not repo.exists("app/extra.py")
    assert CoderAgent.files_mentioned(repo, findings)[:2] == ["tests/test_rules.py", "app/rules.py"]
    print(f"PASS 10: revision touched only what it returned ({r14.diff_summary}); snapshot rolled it back")

    # 11. Revision edge cases: no files, identical files, truncated file, errors, misuse
    r15 = CoderAgent(Oracle(script=["I could not find the bug."])).revise(new_repo(CODE), "bug in rules", "m")
    assert not r15.complete and any("no usable files" in w for w in r15.warnings)
    same = new_repo(CODE)
    r16 = CoderAgent(Oracle(script=[block("app/rules.py", CODE["app/rules.py"])])).revise(same, "bug in rules", "m")
    assert not r16.complete and r16.written == {"app/rules.py": "unchanged"} and any("unchanged" in w for w in r16.warnings)
    r17 = CoderAgent(Oracle(script=[TransientModelError("down", model_id="m")])).revise(new_repo(CODE), "bug", "m")
    assert isinstance(r17.error, TransientModelError) and not r17.complete and r17.snapshot is not None
    for fn in (lambda: CoderAgent(Oracle()).revise(new_repo(CODE), "  ", "m"),
               lambda: CoderAgent(Oracle()).implement(_HLD(markdown="# x\n"), REQ, ANALYSIS, new_repo(), "m")):
        try:
            fn()
            raise AssertionError("expected CoderError")
        except CoderError:
            pass
    print("PASS 11: empty/identical/failed revisions are reported honestly; misuse raises CoderError")

    # 12. Results are JSON-safe; fenced-format replies still work; callback errors do not break a build
    d = r9.to_dict()
    assert json.dumps(d) and d["error"].startswith("FatalModelError") and d["complete"] is False and len(d["batches"]) == 2
    fenced = "Here are the files:\n\n**File: calc.py**\n```python\ndef f():\n    return 1\n```\n"
    r18 = CoderAgent(Oracle(script=[fenced])).implement_lite(spec, new_repo(), "m")
    assert r18.complete and "calc.py" in r18.written and any("did not use the <file" in w for w in r18.warnings)

    def boom(batch, res):
        raise RuntimeError("checkpoint failed")
    r19 = CoderAgent(Oracle(), max_output_tokens=99999).implement(DESIGN, REQ, ANALYSIS, new_repo(), "m", on_batch=boom)
    assert r19.complete and r19.calls == 1
    print("PASS 12: results serialise to JSON; fenced replies accepted with a warning; checkpoint-callback errors contained")

    tmp.cleanup()