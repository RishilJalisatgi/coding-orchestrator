"""
agents/hld.py

The architecture (high-level design) agent. It turns a requirement + task analysis into `architecture.md`
plus a machine-readable file plan the coder can work from, and it can REVISE the design from reviewer
findings by replacing only the sections that must change.

    from agents.hld import HLDAgent, parse_architecture, HLDError

    agent = HLDAgent(registry.generate)                       # any callable (model_id, ModelRequest) -> ModelResponse
    design = agent.design(requirement, analysis, model_id=plan.assignments["hld"])
    state_manager.write_artifact(state, "architecture.md", design.markdown)

    design.files                 # [PlannedFile(path, purpose, depends_on)]  (from the design's JSON block)
    design.build_order()         # dependency-first order, tests last
    design.entry_point, design.run_command, design.dependencies
    design.digest(max_chars=2500)    # compact summary for per-file coding prompts
    design.warnings              # anything the parser had to work around

    design = agent.revise(design, reviewer_findings, model_id=...)    # returns the updated design
    design = parse_architecture(state_manager.read_artifact(state, "architecture.md"))   # resume after a crash

Contract with the orchestrator:
  * The model is chosen by the scheduler and passed in; this agent never picks or calls providers itself.
  * ModelError / ModelUnavailable from the callable propagate untouched, so the orchestrator can ask
    scheduler.fallback_model(). The one exception: if the FIRST reply was received and only the optional
    repair call fails, the partial design is kept and judged on its own merits.
  * HLDError means the reply was unusable (empty, or core sections still missing after one repair call).
    `HLDError.result` carries the partial design, if any.

Required sections (case/wording tolerant): Overview, Components, File Layout (a ```json file plan),
Interfaces, Key Decisions, Testing Strategy, Risks. Core = the first four; without them a coder has nothing
to build from. A truncated reply triggers ONE repair call that asks only for the missing/broken sections.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from context.repository import InvalidPath, normalize_path
from core.logger import get_logger
from models.base import ModelRequest, ModelResponse

logger = get_logger("agents.hld")

GenerateFn = Callable[[str, ModelRequest], ModelResponse]

REQUIRED = ("overview", "components", "file_layout", "interfaces", "decisions", "testing", "risks")
CORE = ("overview", "components", "file_layout", "interfaces")
HEADINGS = {"overview": "Overview", "components": "Components", "file_layout": "File Layout",
            "interfaces": "Interfaces", "decisions": "Key Decisions", "testing": "Testing Strategy",
            "risks": "Risks"}
KEYWORDS = (("changes", ("change",)), ("file_layout", ("file", "layout", "structure")),
            ("interfaces", ("interface", "api", "contract")), ("decisions", ("decision",)),
            ("testing", ("test",)), ("risks", ("risk",)), ("components", ("component", "module")),
            ("overview", ("overview", "summary", "introduction", "goal")))

DEFAULT_MAX_TOKENS = 4000
MIN_MAX_TOKENS = 1500
MAX_MAX_TOKENS = 6000

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
H1_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$")
H2_RE = re.compile(r"^##\s+(.+?)\s*#*\s*$")
JSON_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.S)
BACKTICK_PATH_RE = re.compile(r"`([\w./\\-]+\.[A-Za-z0-9]{1,6})`")


class HLDError(Exception):
    """The design could not be produced or is unusable. `result` holds the partial design, if there is one."""

    def __init__(self, message: str, result: Optional["HLDResult"] = None):
        super().__init__(message)
        self.result = result


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class PlannedFile:
    path: str
    purpose: str = ""
    depends_on: list = field(default_factory=list)

    @property
    def is_test(self) -> bool:
        name = self.path.rsplit("/", 1)[-1]
        parts = self.path.split("/")[:-1]
        return name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts")) \
            or any(p in ("tests", "test", "__tests__") for p in parts)

    @property
    def is_doc(self) -> bool:
        return self.path.lower().endswith((".md", ".rst", ".txt")) and not self.is_test


@dataclass
class HLDResult:
    markdown: str
    title: str = ""
    model_id: str = ""
    files: list = field(default_factory=list)            # [PlannedFile]
    entry_point: str = ""
    run_command: str = ""
    dependencies: list = field(default_factory=list)     # third-party packages the design needs
    warnings: list = field(default_factory=list)
    missing: list = field(default_factory=list)          # required sections absent or unusable
    revisions: list = field(default_factory=list)        # change notes from revise() calls
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    calls: int = 0

    @property
    def complete(self) -> bool:
        return not any(k in CORE for k in self.missing)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def sections(self) -> dict:
        """{canonical key: body} for the recognised sections."""
        return {s[0]: s[2] for s in split_sections(self.markdown)[2] if not s[0].startswith("x:")}

    def section(self, key: str, default: str = "") -> str:
        return self.sections().get(key, default)

    def build_order(self) -> list:
        """File paths, dependencies first (stable for ties): code, then tests, then docs. Cycles are tolerated."""
        sources = [f for f in self.files if not f.is_test and not f.is_doc]
        tests = [f for f in self.files if f.is_test]
        docs = [f for f in self.files if f.is_doc]
        return _topological(sources) + _topological(tests) + _topological(docs)

    def digest(self, max_chars: int = 2500) -> str:
        """Compact summary for coding prompts: overview, file plan, then as much of the interfaces as fits."""
        lines = ["ARCHITECTURE DIGEST"]
        overview = self.section("overview").strip()
        if overview:
            lines += ["Overview: " + " ".join(overview.split())[:600]]
        if self.entry_point or self.run_command:
            lines.append(f"Entry point: {self.entry_point or '?'}  Run: {self.run_command or '?'}")
        if self.dependencies:
            lines.append("Third-party packages: " + ", ".join(self.dependencies))
        if self.files:
            lines.append("Files (build order):")
            by_path = {f.path: f for f in self.files}
            for path in self.build_order():
                f = by_path[path]
                dep = f" [uses: {', '.join(f.depends_on)}]" if f.depends_on else ""
                lines.append(f"  - {path}: {f.purpose}{dep}".rstrip())
        text = "\n".join(lines)
        room = max_chars - len(text) - len("\nInterfaces:\n")
        interfaces = self.section("interfaces").strip()
        if interfaces and room > 120:
            text += "\nInterfaces:\n" + (interfaces if len(interfaces) <= room else interfaces[: room - 15] + "\n...(truncated)")
        return text if len(text) <= max_chars else text[: max_chars - 15] + "\n...(truncated)"

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if k != "files"}
        data["files"] = [dict(f.__dict__) for f in self.files]
        return data

    @classmethod
    def from_dict(cls, raw: dict) -> "HLDResult":
        known = set(cls.__dataclass_fields__)
        data = {k: v for k, v in raw.items() if k in known and k != "files"}
        data["files"] = [PlannedFile(**{k: v for k, v in f.items() if k in PlannedFile.__dataclass_fields__})
                         for f in raw.get("files", [])]
        return cls(**data)


def _topological(files: list) -> list:
    paths = [f.path for f in files]
    known = set(paths)
    deps = {f.path: [d for d in f.depends_on if d in known and d != f.path] for f in files}
    done: list = []
    placed: set = set()
    remaining = list(paths)
    while remaining:
        ready = [p for p in remaining if all(d in placed for d in deps[p])]
        if not ready:                                   # a cycle: keep the original order for what is left
            done += remaining
            break
        for p in ready:
            done.append(p)
            placed.add(p)
        remaining = [p for p in remaining if p not in placed]
    return done


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def section_key(heading: str) -> str:
    """Canonical key for a heading ('Data & Interfaces' -> 'interfaces'); unknown headings keep their own key."""
    text = re.sub(r"[^a-z ]", " ", heading.lower())
    for key, words in KEYWORDS:
        if any(w in text for w in words):
            return key
    return "x:" + " ".join(text.split())


def clean_reply(text: str) -> str:
    """Strip think blocks, a wrapping ```markdown fence and chatter before the first heading."""
    text = THINK_RE.sub("", text or "").strip()
    wrapped = (re.search(r"```(?:markdown|md)[ \t]*\n(.*)\n```\s*$", text, re.S)
               or re.match(r"^```[ \t]*\n(.*)\n```\s*$", text, re.S))
    if wrapped:
        text = wrapped.group(1).strip()
    fence = False
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            fence = not fence
        elif not fence and re.match(r"^#{1,2}\s", line):
            return "\n".join(lines[i:]).strip()
    return text


def split_sections(text: str) -> tuple:
    """(title, preamble, [(key, heading, body)]). Headings inside code fences are ignored."""
    title, pre, sections, current, fence = "", [], [], None, False
    seen: set = set()
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fence = not fence
        elif not fence:
            m2 = H2_RE.match(line)
            if m2:
                key = section_key(m2.group(1))
                if key in seen:
                    key = "x:" + " ".join(re.sub(r"[^a-z ]", " ", m2.group(1).lower()).split()) + " (again)"
                seen.add(key)
                current = [key, m2.group(1).strip(), []]
                sections.append(current)
                continue
            m1 = H1_RE.match(line)
            if m1 and not title and current is None:
                title = m1.group(1).strip()
                continue
        (current[2] if current else pre).append(line)
    return title, "\n".join(pre).strip(), [(k, h, "\n".join(b).strip("\n")) for k, h, b in sections]


def assemble(title: str, preamble: str, sections: list) -> str:
    """Rebuild a document with the standard sections in their standard order (unknown ones after)."""
    order = {k: i for i, k in enumerate(REQUIRED)}
    ordered = sorted(sections, key=lambda s: order.get(s[0], 100))
    parts = [f"# {title or 'Architecture'}"]
    if preamble:
        parts.append(preamble)
    for _key, heading, body in ordered:
        parts.append(f"## {heading}\n\n{body.strip()}")
    return "\n\n".join(parts).rstrip() + "\n"


def parse_file_plan(body: str) -> tuple:
    """(files, extras, warnings) from a File Layout section: a JSON block, or backticked paths as a fallback."""
    warnings: list = []
    data: Any = None
    for match in JSON_FENCE_RE.finditer(body):
        try:
            data = json.loads(match.group(1))
            break
        except ValueError:
            continue
    entries: list = []
    extras: dict = {}
    if isinstance(data, dict):
        entries = data.get("files") or []
        extras = {k: data.get(k) for k in ("entry_point", "run_command", "dependencies")}
    elif isinstance(data, list):
        entries = data
    if not entries and data is None:
        found = list(dict.fromkeys(BACKTICK_PATH_RE.findall(body)))
        if found:
            warnings.append("file plan had no valid JSON block: file list recovered from backticked paths")
            entries = found

    files: list = []
    seen: set = set()
    for entry in entries:
        raw, purpose, deps = (entry, "", []) if isinstance(entry, str) else (
            entry.get("path") if isinstance(entry, dict) else None,
            str(entry.get("purpose", "")) if isinstance(entry, dict) else "",
            entry.get("depends_on") if isinstance(entry, dict) else [])
        try:
            path = normalize_path(raw)
        except InvalidPath as exc:
            warnings.append(f"file plan entry {raw!r} dropped: {exc}")
            continue
        if path in seen:
            warnings.append(f"file plan lists {path} twice: duplicate dropped")
            continue
        seen.add(path)
        clean_deps = []
        for dep in deps if isinstance(deps, list) else []:
            try:
                clean_deps.append(normalize_path(dep))
            except InvalidPath:
                continue
        files.append(PlannedFile(path=path, purpose=purpose.strip(), depends_on=clean_deps))
    known = {f.path for f in files}
    for f in files:
        f.depends_on = [d for d in dict.fromkeys(f.depends_on) if d in known and d != f.path]
    return files, extras, warnings


def _truncated(text: str) -> bool:
    return sum(1 for line in text.splitlines() if line.strip().startswith("```")) % 2 == 1


def parse_architecture(markdown: str, model_id: str = "") -> HLDResult:
    """Parse an architecture document (a model reply or a saved architecture.md) into an HLDResult."""
    text = clean_reply(markdown)
    title, _pre, sections = split_sections(text)
    keys = {s[0]: s for s in sections}
    result = HLDResult(markdown=text.rstrip() + "\n", title=title, model_id=model_id)
    missing = [k for k in REQUIRED if k not in keys or not keys[k][2].strip()]
    if "file_layout" in keys:
        files, extras, warnings = parse_file_plan(keys["file_layout"][2])
        result.files, result.warnings = files, warnings
        result.entry_point = str(extras.get("entry_point") or "")
        result.run_command = str(extras.get("run_command") or "")
        deps = extras.get("dependencies")
        result.dependencies = [str(d) for d in deps if isinstance(d, (str, int, float))] if isinstance(deps, list) else []
        if not files and "file_layout" not in missing:
            missing.append("file_layout")
        if result.entry_point and result.files and result.entry_point not in {f.path for f in files}:
            try:
                ep = normalize_path(result.entry_point)
            except InvalidPath:
                ep = ""
            if ep not in {f.path for f in files}:
                result.warnings.append(f"entry_point {result.entry_point!r} is not in the file list")
            result.entry_point = ep or result.entry_point
    result.missing = missing
    return result


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a senior software architect. You write concrete, right-sized high-level designs that a different "
    "engineer (another AI model) will implement file by file without being able to ask you questions. "
    "Prefer the standard library and boring, proven choices. Do not write implementation code: signatures, "
    "data shapes and short examples only. Never leave placeholders such as 'TBD'."
)

TEMPLATE = """Design the architecture for this requirement.

REQUIREMENT
{requirement}

ANALYSIS
{analysis}
{extras}
{sizing}

Answer with ONE markdown document and nothing else, using exactly these sections in this order:

# <short project title>
## Overview
What it does, who uses it, the main flow in 3-6 sentences.
## Components
Each component: its responsibility, what it owns, what it must NOT do.
## File Layout
A single ```json block and nothing else in this section:
{{"entry_point": "main.py", "run_command": "python main.py", "dependencies": ["third-party packages, or empty"],
 "files": [{{"path": "pkg/module.py", "purpose": "one line", "depends_on": ["other/file.py"]}}]}}
List EVERY file to create (source, tests, README.md, requirements if needed), repo-relative paths, forward slashes.
## Interfaces
Signatures (with types, return values and errors raised) of every function/class used ACROSS files, plus the
shape of persisted/transferred data (tables, JSON). This is the contract the coder follows.
## Key Decisions
The 3-6 decisions that matter, each with the reason and the alternative rejected.
## Testing Strategy
What is tested and how (unit/integration), which files, how to run them.
## Risks
Concrete technical risks and the mitigation for each.
"""

REPAIR_TEMPLATE = """Your architecture document below is incomplete or has unusable parts.

CURRENT DOCUMENT
{document}

SECTIONS TO PROVIDE: {sections}
Output ONLY those sections, each starting with its exact heading (## {example}), complete and consistent with the
document above. For "File Layout" output the entire JSON block again. No preamble."""

REVISE_TEMPLATE = """A reviewer found problems with the implementation that may require changing the architecture.

REQUIREMENT
{requirement}

CURRENT ARCHITECTURE
{document}

REVIEW FINDINGS
{findings}

Decide which sections of the architecture must change to resolve the findings. Output ONLY those sections, each
with its exact heading and its COMPLETE new text (if you change File Layout, output the whole JSON block), followed by
## Changes
a bullet list of what you changed and why. Do not output unchanged sections. If the findings are implementation
bugs that the architecture already covers, output only "## Changes" saying so."""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def describe_analysis(analysis: Any) -> str:
    if analysis is None:
        return "(no analysis available: infer reasonable assumptions and state them in the Overview)"
    lines = [f"Type: {_get(analysis, 'task_type', 'unknown')}   Complexity: {_get(analysis, 'complexity', '?')}/10   "
             f"Language: {_get(analysis, 'primary_language', 'unspecified')}"]
    needs = [n for n, flag in (("a user interface", "needs_ui"), ("a database", "needs_database"),
                               ("network access", "needs_network")) if _get(analysis, flag, False)]
    lines.append("Needs: " + (", ".join(needs) if needs else "no UI, database or network"))
    for label, name in (("Key features", "key_features"), ("Known risks", "risks"), ("Assumptions", "assumptions")):
        values = _get(analysis, name) or []
        if values:
            lines.append(f"{label}: " + "; ".join(str(v) for v in values[:10]))
    return "\n".join(lines)


def sizing_guidance(complexity: Optional[float]) -> str:
    c = complexity if isinstance(complexity, (int, float)) else 5.0
    if c < 4:
        return "SIZE: keep it small: at most ~6 source files, no layers or abstractions the requirement does not need. About 500 words."
    if c < 7:
        return "SIZE: a modest design: a few cohesive modules with clear boundaries. About 900 words."
    return ("SIZE: this is a complex system: separate concerns cleanly, define module boundaries and the data flow "
            "carefully, but do not invent components the requirement does not need. About 1500 words.")


def output_budget(analysis: Any, override: Optional[int] = None) -> int:
    if override:
        return int(override)
    plan = _get(analysis, "token_plan", None)
    hld = plan.get("hld") if isinstance(plan, dict) else None
    base = int(hld * 0.8) if isinstance(hld, (int, float)) and hld > 0 else DEFAULT_MAX_TOKENS
    return max(MIN_MAX_TOKENS, min(MAX_MAX_TOKENS, base))


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class HLDAgent:
    def __init__(self, generate: GenerateFn, temperature: float = 0.3, max_repair_calls: int = 1):
        self.generate = generate
        self.temperature = temperature
        self.max_repair_calls = max(0, max_repair_calls)

    # ---- design -------------------------------------------------------------------- #

    def design(self, requirement: str, analysis: Any, model_id: str, *, repo_context: str = "",
               constraints: str = "", max_tokens: Optional[int] = None) -> HLDResult:
        """
        Produce a new design. repo_context: a rendered ContextSelector result when extending an existing
        codebase; constraints: extra rules (e.g. 'must run offline'). Raises HLDError / propagates ModelError.
        """
        if not (requirement or "").strip():
            raise HLDError("empty requirement")
        extras = ""
        if constraints.strip():
            extras += f"\nCONSTRAINTS\n{constraints.strip()}\n"
        if repo_context.strip():
            extras += f"\nEXISTING CODEBASE (extend it; do not redesign what already works)\n{repo_context.strip()}\n"
        prompt = TEMPLATE.format(requirement=requirement.strip(), analysis=describe_analysis(analysis), extras=extras,
                                 sizing=sizing_guidance(_get(analysis, "complexity")))
        budget = output_budget(analysis, max_tokens)
        response = self._ask(model_id, prompt, budget, attempt=1)
        text = clean_reply(response.text)
        if not text.strip():
            raise HLDError(f"{model_id} returned an empty design")
        result = parse_architecture(text, model_id)
        self._add_usage(result, response)

        repairs = 0
        while repairs < self.max_repair_calls and self._needs_repair(result):
            repairs += 1
            result = self._repair(result, model_id, budget, repairs)
        if not result.complete:
            raise HLDError(f"design from {model_id} is missing core sections: "
                           f"{', '.join(k for k in result.missing if k in CORE)}", result)
        if result.missing:
            result.warnings.append("optional sections missing: " + ", ".join(result.missing))
        logger.info("HLD by %s: %d files, %d tokens, %d call(s)%s", model_id, len(result.files), result.total_tokens,
                    result.calls, f", warnings: {len(result.warnings)}" if result.warnings else "")
        return result

    def _needs_repair(self, result: HLDResult) -> bool:
        if not result.missing:
            return False
        return (not result.complete) or _truncated(result.markdown)

    def _repair(self, result: HLDResult, model_id: str, budget: int, attempt: int) -> HLDResult:
        wanted = [k for k in REQUIRED if k in result.missing]
        document = result.markdown
        # a truncated final section is replaced wholesale by the repair, so cut it out of the document first
        title, pre, sections = split_sections(document)
        if _truncated(document) and sections:
            last = sections[-1]
            if _truncated(last[2]) or last[0] in wanted:
                sections = sections[:-1]
                if last[0] in REQUIRED and last[0] not in wanted:
                    wanted.append(last[0])
        document = assemble(title, pre, sections)
        prompt = REPAIR_TEMPLATE.format(document=document, sections=", ".join(HEADINGS[k] for k in wanted),
                                        example=HEADINGS[wanted[0]] if wanted else "Risks")
        try:
            response = self._ask(model_id, prompt, budget, attempt=attempt + 1)
        except Exception as exc:                             # keep what we already have; the caller judges it
            logger.warning("HLD repair call to %s failed (%s); using the partial design", model_id, exc)
            result.warnings.append(f"repair call failed: {exc}")
            return result
        fix_title, _p, fixed = split_sections(clean_reply(response.text))
        by_key = {s[0]: s for s in sections}
        for key, heading, body in fixed:
            if key in ("changes",) or not body.strip():
                continue
            by_key[key] = (key, heading, body)
        merged = parse_architecture(assemble(title or fix_title, pre, list(by_key.values())), model_id)
        merged.warnings = result.warnings + merged.warnings + [f"repaired by a second call: {', '.join(wanted)}"]
        merged.input_tokens, merged.output_tokens = result.input_tokens, result.output_tokens
        merged.cost_usd, merged.latency_seconds, merged.calls = result.cost_usd, result.latency_seconds, result.calls
        merged.revisions = list(result.revisions)
        self._add_usage(merged, response)
        return merged

    # ---- revise -------------------------------------------------------------------- #

    def revise(self, current: HLDResult, findings: str, model_id: str, *, requirement: str = "",
               max_tokens: Optional[int] = None) -> HLDResult:
        """
        Update a design from review findings by replacing only the sections the model returns. The returned
        HLDResult is new; `current` is untouched. Raises HLDError if the reply changes nothing usable.
        """
        if not (findings or "").strip():
            raise HLDError("no findings to revise the architecture from", current)
        prompt = REVISE_TEMPLATE.format(requirement=(requirement or "(see architecture)").strip(),
                                        document=current.markdown, findings=findings.strip()[:6000])
        response = self._ask(model_id, prompt, output_budget(None, max_tokens), attempt=1, purpose="hld")
        _t, _p, returned = split_sections(clean_reply(response.text))
        notes = next((body for key, _h, body in returned if key == "changes"), "").strip()
        replacements = [s for s in returned if s[0] != "changes" and s[2].strip()]

        revisions = list(current.revisions)
        if notes:
            revisions.append(notes)
        if not replacements:
            revised = HLDResult.from_dict(current.to_dict())
            revised.revisions = revisions
            self._reset_usage(revised, response)
            revised.warnings = [w for w in current.warnings]
            revised.warnings.append("revision changed no sections" + (f": {' '.join(notes.split())[:200]}" if notes else ""))
            return revised

        title, pre, sections = split_sections(current.markdown)
        by_key = {s[0]: s for s in sections}
        for key, heading, body in replacements:
            by_key[key] = (key, by_key[key][1] if key in by_key else heading, body)
        revised = parse_architecture(assemble(title, pre, list(by_key.values())), model_id)
        if not revised.complete:
            raise HLDError("revised design lost core sections: " +
                           ", ".join(k for k in revised.missing if k in CORE), revised)
        revised.revisions = revisions
        revised.warnings.append("revised sections: " + ", ".join(s[0] for s in replacements))
        self._reset_usage(revised, response)
        logger.info("HLD revised by %s: sections %s", model_id, [s[0] for s in replacements])
        return revised

    # ---- helpers ------------------------------------------------------------------- #

    def _ask(self, model_id: str, prompt: str, max_tokens: int, attempt: int, purpose: str = "hld") -> ModelResponse:
        request = ModelRequest(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens,
                               temperature=self.temperature, purpose=purpose,
                               metadata={"stage": "hld", "attempt": attempt})
        return self.generate(model_id, request)

    @staticmethod
    def _add_usage(result: HLDResult, response: ModelResponse) -> None:
        result.input_tokens += int(getattr(response, "input_tokens", 0) or 0)
        result.output_tokens += int(getattr(response, "output_tokens", 0) or 0)
        result.cost_usd += float(getattr(response, "cost_usd", 0.0) or 0.0)
        result.latency_seconds += float(getattr(response, "latency_seconds", 0.0) or 0.0)
        result.calls += 1

    @classmethod
    def _reset_usage(cls, result: HLDResult, response: ModelResponse) -> None:
        result.input_tokens = result.output_tokens = result.calls = 0
        result.cost_usd = result.latency_seconds = 0.0
        cls._add_usage(result, response)


# --------------------------------------------------------------------------- #
# Self-test:  python -m agents.hld     (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from core.logger import setup_logging
    from models.base import FatalModelError, TransientModelError

    setup_logging("ERROR")

    PLAN = {"entry_point": "main.py", "run_command": "python main.py", "dependencies": ["watchdog"],
            "files": [{"path": "main.py", "purpose": "entry point", "depends_on": ["app/watcher.py", "app/rules.py"]},
                      {"path": "app/watcher.py", "purpose": "watch the folder", "depends_on": ["app/rules.py"]},
                      {"path": "app/rules.py", "purpose": "categorise a file", "depends_on": []},
                      {"path": "tests/test_rules.py", "purpose": "rule tests", "depends_on": ["app/rules.py"]},
                      {"path": "README.md", "purpose": "usage"}]}

    def doc(skip=(), plan=PLAN, cut_in=None, title="Downloads Sorter"):
        bodies = {
            "overview": "Watches the downloads folder and moves files into category folders.",
            "components": "Watcher: detects new files. Rules: decides the category. Mover: moves files.",
            "file_layout": "```json\n" + json.dumps(plan, indent=1) + "\n```",
            "interfaces": "```python\ndef categorise(name: str) -> str: ...\nclass Watcher:\n    def start(self) -> None: ...\n```",
            "decisions": "Polling vs events: events, because polling wastes CPU.",
            "testing": "pytest tests/ covers rules.",
            "risks": "Half-downloaded files: wait until the size is stable.",
        }
        parts = [f"# {title}"]
        for key in REQUIRED:
            if key in skip:
                continue
            parts.append(f"## {HEADINGS[key]}\n\n{bodies[key]}")
        text = "\n\n".join(parts)
        return text[:cut_in] if cut_in else text

    class Fake:
        """Scripted model: returns the next reply; records every request."""
        def __init__(self, replies):
            self.replies, self.calls = list(replies), []

        def __call__(self, model_id, request):
            self.calls.append((model_id, request))
            item = self.replies.pop(0)
            if isinstance(item, Exception):
                raise item
            return ModelResponse(text=item, model_id=model_id, input_tokens=1000, output_tokens=500,
                                 cost_usd=0.01, latency_seconds=1.5, attempts=1, purpose=request.purpose)

    ANALYSIS = {"task_type": "desktop_application", "complexity": 7.5, "primary_language": "Python",
                "needs_ui": True, "needs_database": False, "needs_network": False,
                "key_features": ["watch downloads", "categorise by type"], "risks": ["partial downloads"],
                "assumptions": ["Windows or Linux"], "token_plan": {"hld": 5100}}
    REQ = "Build a desktop app that monitors my downloads and automatically categorizes them"

    # 1. A good reply: sections, file plan, build order, extras
    fake = Fake([doc()])
    d = HLDAgent(fake).design(REQ, ANALYSIS, "claude")
    assert d.complete and not d.missing and d.calls == 1 and d.model_id == "claude"
    assert [f.path for f in d.files][:3] == ["main.py", "app/watcher.py", "app/rules.py"]
    order = d.build_order()
    assert order.index("app/rules.py") < order.index("app/watcher.py") < order.index("main.py")
    assert order.index("main.py") < order.index("tests/test_rules.py") < order.index("README.md") == len(order) - 1
    assert d.entry_point == "main.py" and d.run_command == "python main.py" and d.dependencies == ["watchdog"]
    assert d.total_tokens == 1500 and d.cost_usd == 0.01
    print(f"PASS 1: design parsed -> {len(d.files)} files, build order {order[:3]}..., {d.total_tokens} tokens")

    # 2. The request: purpose, system prompt, requirement, analysis, sizing and token budget
    model_id, req = fake.calls[0]
    assert model_id == "claude" and req.purpose == "hld" and req.system and req.max_tokens == 4080
    assert REQ in req.prompt and "watch downloads" in req.prompt and "complexity" in req.prompt.lower() and "a user interface" in req.prompt
    assert "complex system" in req.prompt and "## File Layout" in req.prompt
    small = Fake([doc()])
    HLDAgent(small).design("A script that renames files", {"complexity": 2.0}, "m", constraints="must run offline",
                           repo_context="src/a.py: def f()", max_tokens=1800)
    sreq = small.calls[0][1]
    assert "keep it small" in sreq.prompt and "must run offline" in sreq.prompt and "src/a.py" in sreq.prompt
    assert "EXISTING CODEBASE" in sreq.prompt and sreq.max_tokens == 1800
    assert output_budget({"token_plan": {"hld": 100}}) == MIN_MAX_TOKENS and output_budget({"token_plan": {"hld": 99999}}) == MAX_MAX_TOKENS
    assert output_budget(None) == DEFAULT_MAX_TOKENS
    print("PASS 2: prompt carries requirement, analysis, sizing, constraints, repo context; token budget clamped")

    # 3. Chatter, think blocks and a wrapping fence are stripped
    noisy = "<think>hmm</think>Sure! Here is the design:\n\n```markdown\n" + doc() + "\n```"
    d3 = HLDAgent(Fake([noisy])).design(REQ, ANALYSIS, "m")
    assert d3.markdown.startswith("# Downloads Sorter") and d3.complete and len(d3.files) == 5
    assert parse_architecture("intro line\n\n" + doc()).title == "Downloads Sorter"
    print("PASS 3: think blocks, chatter and wrapping fences removed")

    # 4. Wording-tolerant headings; no JSON -> file list recovered from backticked paths (with a warning)
    loose = ("# X\n\n## Summary\nA tool.\n\n## Module Design\nParts.\n\n## Project Structure\n"
             "- `main.py` entry\n- `core/engine.py` logic\n- `core/engine.py` again\n\n## Data & Interfaces\nf(x)\n\n"
             "## Design Decisions\nd\n\n## Tests\nt\n\n## Risks and Mitigations\nr\n")
    d4 = parse_architecture(loose)
    assert d4.complete and not d4.missing and [f.path for f in d4.files] == ["main.py", "core/engine.py"]
    assert any("backticked" in w for w in d4.warnings)
    print("PASS 4: loose headings recognised; file list recovered from backticks with a warning")

    # 5. Unsafe paths, duplicates, unknown dependencies, cycles
    messy = {"files": [{"path": "../evil.py"}, {"path": "/abs.py"}, {"path": "a.py", "depends_on": ["b.py", "ghost.py"]},
                       {"path": "b.py", "depends_on": ["a.py"]}, {"path": "a.py"}, 7]}
    d5 = parse_architecture(doc(plan=messy))
    assert [f.path for f in d5.files] == ["a.py", "b.py"] and d5.files[0].depends_on == ["b.py"]
    assert sum("dropped" in w for w in d5.warnings) >= 3 and sorted(d5.build_order()) == ["a.py", "b.py"]    # cycle tolerated
    print(f"PASS 5: unsafe/duplicate/junk entries dropped ({len(d5.warnings)} warnings), cycle tolerated")

    # 6. Truncated reply -> ONE repair call asks only for the missing/broken sections, then merges
    cut = doc(cut_in=doc().index("## Interfaces") + 60)             # cuts inside Interfaces (unclosed fence)
    repair_reply = ("## Interfaces\n\n```python\ndef categorise(name: str) -> str: ...\n```\n\n## Key Decisions\nevents\n\n"
                    "## Testing Strategy\npytest\n\n## Risks\npartial files\n")
    fake6 = Fake([cut, repair_reply])
    d6 = HLDAgent(fake6).design(REQ, ANALYSIS, "gpt")
    assert len(fake6.calls) == 2 and d6.calls == 2 and d6.complete and not d6.missing
    r_prompt = fake6.calls[1][1].prompt
    assert "Interfaces" in r_prompt and "Key Decisions" in r_prompt and "Overview" not in r_prompt.split("SECTIONS TO PROVIDE:")[1].splitlines()[0]
    assert d6.total_tokens == 3000 and any("repaired" in w for w in d6.warnings)
    assert d6.markdown.index("## Interfaces") < d6.markdown.index("## Key Decisions") < d6.markdown.index("## Risks")
    print(f"PASS 6: truncated reply repaired with one extra call ({d6.total_tokens} tokens total), sections in order")

    # 7. Core still missing after the repair -> HLDError with the partial design; no endless retrying
    fake7 = Fake([doc(skip=("components", "interfaces")), "## Components\nonly this"])
    try:
        HLDAgent(fake7).design(REQ, ANALYSIS, "m")
        raise AssertionError("expected HLDError")
    except HLDError as exc:
        assert "interfaces" in str(exc) and exc.result is not None and exc.result.files and len(fake7.calls) == 2
    # a single missing optional section is only a warning (no extra call)
    fake7b = Fake([doc(skip=("risks",))])
    d7 = HLDAgent(fake7b).design(REQ, ANALYSIS, "m")
    assert len(fake7b.calls) == 1 and d7.complete and d7.missing == ["risks"] and any("optional" in w for w in d7.warnings)
    print("PASS 7: missing core -> HLDError after one repair; missing optional section -> warning, no extra call")

    # 8. Failures: empty reply, model errors propagate, a failing repair keeps the partial design
    for bad in ("", "   <think>x</think> "):
        try:
            HLDAgent(Fake([bad])).design(REQ, ANALYSIS, "m")
            raise AssertionError("expected HLDError")
        except HLDError as exc:
            assert "empty" in str(exc)
    for err in (FatalModelError("bad key", model_id="m"), TransientModelError("down", model_id="m")):
        try:
            HLDAgent(Fake([err])).design(REQ, ANALYSIS, "m")
            raise AssertionError("expected ModelError")
        except (FatalModelError, TransientModelError) as exc:
            assert exc is err
    try:
        HLDAgent(Fake([doc(skip=("interfaces",)), TransientModelError("down")])).design(REQ, ANALYSIS, "m")
        raise AssertionError("expected HLDError")
    except HLDError as exc:
        assert exc.result is not None and any("repair call failed" in w for w in exc.result.warnings)
    try:
        HLDAgent(Fake([])).design("   ", ANALYSIS, "m")
        raise AssertionError("expected HLDError")
    except HLDError:
        pass
    print("PASS 8: empty reply -> HLDError; ModelError propagates; failed repair keeps the partial design")

    # 9. revise(): only returned sections change, file plan re-parsed, notes kept, original untouched
    new_plan = dict(PLAN, files=PLAN["files"] + [{"path": "app/mover.py", "purpose": "move files safely",
                                                   "depends_on": ["app/rules.py"]}])
    rev_reply = ("## File Layout\n\n```json\n" + json.dumps(new_plan) + "\n```\n\n"
                 "## Risks\nMoves can fail across drives: copy then delete.\n\n## Changes\n- added app/mover.py\n- risk about drives\n")
    fake9 = Fake([rev_reply])
    d9 = HLDAgent(fake9).revise(d, "Moving across drives loses files; no mover module.", "claude", requirement=REQ)
    assert "app/mover.py" in [f.path for f in d9.files] and len(d.files) == 5 and len(d9.files) == 6
    assert d9.section("overview") == d.section("overview") and "copy then delete" in d9.section("risks")
    assert "copy then delete" not in d.section("risks") and d9.revisions and "mover" in d9.revisions[0]
    assert d9.calls == 1 and d9.total_tokens == 1500 and "## Changes" not in d9.markdown
    assert "REVIEW FINDINGS" in fake9.calls[0][1].prompt and "Moving across drives" in fake9.calls[0][1].prompt
    print(f"PASS 9: revision replaced 2 sections ({len(d.files)} -> {len(d9.files)} files); unchanged sections intact")

    # 10. revise() edge cases: nothing changed, empty findings, a revision that breaks the plan
    d10 = HLDAgent(Fake(["## Changes\nThe architecture already covers this; it is an implementation bug."])).revise(d, "bug", "m")
    assert d10.markdown == d.markdown and any("changed no sections" in w for w in d10.warnings) and d10 is not d
    try:
        HLDAgent(Fake([])).revise(d, "  ", "m")
        raise AssertionError("expected HLDError")
    except HLDError:
        pass
    try:
        HLDAgent(Fake(["## File Layout\n\n```json\n{\"files\": []}\n```\n"])).revise(d, "x", "m")
        raise AssertionError("expected HLDError")
    except HLDError as exc:
        assert "file_layout" in str(exc)
    print("PASS 10: no-op revision, empty findings and a plan-breaking revision handled")

    # 11. Digest respects its budget; round trips for resume (markdown and dict)
    for limit in (3000, 600, 250):
        text = d.digest(limit)
        assert len(text) <= limit and "ARCHITECTURE DIGEST" in text, (limit, len(text))
    assert "app/rules.py" in d.digest(600) and "def categorise" in d.digest(3000)
    again = parse_architecture(d.markdown, "claude")
    assert [f.path for f in again.files] == [f.path for f in d.files] and again.complete and again.build_order() == d.build_order()
    back = HLDResult.from_dict(json.loads(json.dumps(d9.to_dict())))
    assert back.build_order() == d9.build_order() and back.revisions == d9.revisions and back.files[0].depends_on == d9.files[0].depends_on
    print("PASS 11: digest within budget; architecture.md re-parsed on resume; to_dict/from_dict round trip")

    # 12. Misc: sections inside code fences are not headings; PlannedFile.is_test; describe_analysis handles objects
    tricky = doc().replace("def categorise(name: str) -> str: ...", "## not a heading\ndef categorise(): ...")
    assert [k for k, _h, _b in split_sections(tricky)[2]] == list(REQUIRED)
    assert PlannedFile("tests/test_a.py").is_test and PlannedFile("pkg/a_test.py").is_test and not PlannedFile("pkg/a.py").is_test

    class Obj:
        task_type, complexity, primary_language = "backend_api", 5, "Go"
        needs_ui = needs_database = needs_network = False
        key_features, risks, assumptions, token_plan = ["x"], [], [], {}
    assert "Go" in describe_analysis(Obj()) and "no UI" in describe_analysis(Obj()) and "no analysis" in describe_analysis(None)
    print("PASS 12: fenced '##' lines ignored, test-file detection, analysis objects and None handled")