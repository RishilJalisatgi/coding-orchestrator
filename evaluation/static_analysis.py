"""
evaluation/static_analysis.py

Static checks on a candidate workspace: no candidate code is executed here.

    from context.repository import Repository
    from evaluation.static_analysis import StaticAnalyzer

    report = StaticAnalyzer(settings.orchestrator).analyze(Repository(candidate_dir))
    report.metrics            # {"compiles": 1.0, "code_quality": 0.82, "security": 1.0, "documentation": 0.6}
    report.tools              # per tool: ran / skipped (and why) / issue counts
    report.issues             # [Issue(tool, path, line, code, message, severity)]

What is checked, and what happens when a tool is missing:
  * compile check   (always, stdlib)   share of .py files that parse                  -> `compiles`
  * ruff            (optional)         lint issues per 100 lines                       -> part of `code_quality`
  * mypy            (optional)         type errors per 100 lines                       -> part of `code_quality`
  * complexity      (always, stdlib)   average cyclomatic complexity (own AST count)   -> part of `code_quality`
  * bandit          (optional)         security findings by severity                   -> `security`
                    fallback: a small built-in AST scan for obvious problems (eval, shell=True, pickle,
                    verify=False, hard-coded secrets). `report.sources["security"]` says which one ran.
  * documentation   (always, stdlib)   docstring coverage + README + usage section    -> `documentation`

A missing tool is recorded as skipped (never an error) and its metric is None, so scoring.py drops it
instead of guessing. `code_quality` needs at least one of lint / type-check results; complexity alone is
not enough to call code "good", so without them code_quality is None.

Safety: tools run in subprocesses with a timeout, in their own process group, with a scrubbed
environment (no API keys), and with project config files disabled (`--isolated`, empty mypy config) so a
candidate cannot make a tool load its code. Only Python is analysed in depth; other languages get
line counts only.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from config.settings import OrchestratorSettings
from core.logger import get_logger
from evaluation.scoring import (score_complexity, score_documentation, score_lint, score_quality,
                                score_security)

logger = get_logger("evaluation.static_analysis")

MAX_OUTPUT_CHARS = 1_000_000
MAX_ISSUES_KEPT = 200
TEST_DIR_NAMES = ("tests", "test", "__tests__")
ENV_WHITELIST = ("PATH", "SYSTEMROOT", "COMSPEC", "PATHEXT", "LANG", "LC_ALL", "TZ")


# --------------------------------------------------------------------------- #
# Running commands safely (shared helper; evaluation/tests.py and runtime.py reuse it)
# --------------------------------------------------------------------------- #

@dataclass
class CommandResult:
    returncode: Optional[int]          # None if the process never started or was killed by timeout
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    seconds: float = 0.0
    error: str = ""                    # could not start (tool not found, permission...)


def safe_env(extra: Optional[dict] = None) -> dict:
    """A scrubbed environment: only PATH-like basics. API keys and tokens are NOT passed on."""
    env = {k: os.environ[k] for k in ENV_WHITELIST if k in os.environ}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"})
    if extra:
        env.update(extra)
    return env


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_command(cmd: list, cwd: str, timeout: float, env: Optional[dict] = None) -> CommandResult:
    """Run a command with a hard timeout, in its own process group; output is size-capped. Never raises."""
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env if env is not None else safe_env(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            start_new_session=(os.name == "posix"))
    except (FileNotFoundError, PermissionError, OSError) as e:
        return CommandResult(None, error=f"{type(e).__name__}: {e}")
    try:
        out, err = proc.communicate(timeout=timeout)
        return CommandResult(proc.returncode, out[:MAX_OUTPUT_CHARS], err[:MAX_OUTPUT_CHARS],
                             False, time.monotonic() - started)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            out, err = "", ""
        return CommandResult(None, (out or "")[:MAX_OUTPUT_CHARS], (err or "")[:MAX_OUTPUT_CHARS],
                             True, time.monotonic() - started)


Runner = Callable[[list, str, float], CommandResult]


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class Issue:
    tool: str
    path: str
    line: int
    code: str
    message: str
    severity: str = "low"              # low | medium | high  (lint/type issues are 'low')


@dataclass
class ToolReport:
    name: str
    ran: bool = False
    skipped_reason: str = ""
    issue_count: int = 0
    by_severity: dict = field(default_factory=dict)
    seconds: float = 0.0
    note: str = ""


@dataclass
class StaticReport:
    metrics: dict = field(default_factory=dict)        # compiles, code_quality, security, documentation (0..1 or None)
    tools: dict = field(default_factory=dict)          # name -> ToolReport
    issues: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)        # metric -> what produced it ("ruff+mypy", "bandit", "builtin"...)
    lines: int = 0
    python_files: int = 0
    avg_complexity: Optional[float] = None
    docstring_ratio: Optional[float] = None
    compile_errors: list = field(default_factory=list)  # [(path, message)]
    notes: list = field(default_factory=list)

    def skipped_tools(self) -> list:
        return sorted(n for n, t in self.tools.items() if not t.ran)

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if k not in ("tools", "issues")}
        data["tools"] = {n: t.__dict__ for n, t in self.tools.items()}
        data["issues"] = [i.__dict__ for i in self.issues[:MAX_ISSUES_KEPT]]
        return data

    def summary(self) -> str:
        parts = [f"{k}={v:.2f}" for k, v in self.metrics.items() if v is not None]
        text = ", ".join(parts) or "no metrics"
        if self.skipped_tools():
            text += f" (skipped: {', '.join(self.skipped_tools())})"
        return text


# --------------------------------------------------------------------------- #
# Stdlib analyses (always available)
# --------------------------------------------------------------------------- #

def is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (any(p in TEST_DIR_NAMES for p in parts[:-1]) or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


def check_compiles(sources: dict) -> tuple:
    """sources {path: text} -> (ratio or None, [(path, error)])."""
    if not sources:
        return None, []
    errors = []
    for path, text in sources.items():
        try:
            compile(text, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            errors.append((path, f"{type(e).__name__}: {e}"[:200]))
    return (len(sources) - len(errors)) / len(sources), errors


_BRANCH_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp, ast.Assert)


def _function_complexity(fn: ast.AST) -> int:
    """McCabe-style: 1 + decision points. Nested function bodies are counted with their own function."""
    total = 1
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, _BRANCH_NODES):
            total += 1
        elif isinstance(node, ast.BoolOp):
            total += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            total += 1 + len(node.ifs)
        elif isinstance(node, ast.Match if hasattr(ast, "Match") else ()):      # pragma: no cover
            total += len(node.cases)
        stack.extend(ast.iter_child_nodes(node))
    return total


def analyse_python_source(text: str) -> Optional[dict]:
    """Per-file facts from the AST: complexities, docstring coverage. None if the file does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    complexities, documented, documentable = [], 0, 0
    if tree.body:                                         # module docstring counts once per non-empty module
        documentable += 1
        documented += 1 if ast.get_docstring(tree) else 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexities.append(_function_complexity(node))
            if not node.name.startswith("_"):
                documentable += 1
                documented += 1 if ast.get_docstring(node) else 0
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            documentable += 1
            documented += 1 if ast.get_docstring(node) else 0
    return {"complexities": complexities, "documented": documented, "documentable": documentable}


_SECURITY_PATTERNS = [
    # (regex on a source line, code, message, severity)
    (re.compile(r"\b(?:eval|exec)\s*\("), "B-EVAL", "use of eval/exec", "high"),
    (re.compile(r"shell\s*=\s*True"), "B-SHELL", "subprocess with shell=True", "high"),
    (re.compile(r"\bos\.system\s*\("), "B-OSSYSTEM", "os.system call", "medium"),
    (re.compile(r"\bpickle\.loads?\s*\("), "B-PICKLE", "pickle deserialisation", "medium"),
    (re.compile(r"\byaml\.load\s*\((?![^)]*Loader)"), "B-YAML", "yaml.load without a safe Loader", "medium"),
    (re.compile(r"verify\s*=\s*False"), "B-TLS", "TLS verification disabled", "medium"),
    (re.compile(r"""(?i)\b(?:password|passwd|secret|api_?key|token)\w*\s*=\s*["'][^"']{6,}["']"""),
     "B-SECRET", "hard-coded secret", "high"),
    (re.compile(r"\bhashlib\.(?:md5|sha1)\s*\("), "B-WEAKHASH", "weak hash (md5/sha1)", "low"),
    (re.compile(r"\bdebug\s*=\s*True"), "B-DEBUG", "debug mode enabled", "low"),
    (re.compile(r"""(?:execute|executemany)\s*\(\s*f?["'][^"']*(?:%s|\{|\+)"""), "B-SQL",
     "SQL built from strings", "medium"),
]


def builtin_security_scan(sources: dict) -> list:
    """A deliberately small fallback for when bandit is not installed. Comments are skipped."""
    issues = []
    for path, text in sources.items():
        if is_test_path(path):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern, code, message, severity in _SECURITY_PATTERNS:
                if pattern.search(line):
                    issues.append(Issue("builtin", path, number, code, message, severity))
    return issues


# --------------------------------------------------------------------------- #
# Tool output parsers
# --------------------------------------------------------------------------- #

def parse_ruff_json(text: str) -> list:
    issues = []
    for item in json.loads(text or "[]"):
        loc = item.get("location") or {}
        issues.append(Issue("ruff", item.get("filename", ""), int(loc.get("row") or 0),
                            str(item.get("code") or "ruff"), str(item.get("message", "")), "low"))
    return issues


_MYPY_LINE = re.compile(r"^(?P<path>.+?):(?P<line>\d+)(?::\d+)?: (?P<level>error|warning|note): (?P<msg>.*?)"
                        r"(?:\s+\[(?P<code>[\w-]+)\])?$")


def parse_mypy_text(text: str) -> list:
    issues = []
    for raw in (text or "").splitlines():
        m = _MYPY_LINE.match(raw.strip())
        if m and m.group("level") == "error":
            issues.append(Issue("mypy", m.group("path"), int(m.group("line")), m.group("code") or "mypy",
                                m.group("msg"), "low"))
    return issues


def parse_bandit_json(text: str) -> list:
    data = json.loads(text or "{}")
    issues = []
    for item in data.get("results", []):
        severity = str(item.get("issue_severity", "LOW")).lower()
        issues.append(Issue("bandit", item.get("filename", ""), int(item.get("line_number") or 0),
                            str(item.get("test_id", "bandit")), str(item.get("issue_text", "")),
                            severity if severity in ("low", "medium", "high") else "low"))
    return issues


def _relative(path: str, root: str) -> str:
    path = path.replace("\\", "/")
    root = root.replace("\\", "/").rstrip("/") + "/"
    if path.startswith(root):
        return path[len(root):]
    return path[2:] if path.startswith("./") else path


# --------------------------------------------------------------------------- #
# The analyzer
# --------------------------------------------------------------------------- #

class StaticAnalyzer:
    def __init__(self, settings: Optional[OrchestratorSettings] = None, runner: Optional[Runner] = None,
                 commands: Optional[dict] = None):
        """
        runner:   how commands are executed (tests inject a fake).
        commands: {"ruff": [argv...] | None, ...} to bypass tool discovery (tests); None = tool missing.
        """
        self.settings = settings or OrchestratorSettings()
        self.runner: Runner = runner or run_command
        self._commands = commands

    # ---- discovery ----------------------------------------------------------- #

    def command_for(self, tool: str) -> Optional[list]:
        if self._commands is not None:
            return self._commands.get(tool)
        exe = shutil.which(tool)
        if exe:
            return [exe]
        try:
            import importlib.util
            if importlib.util.find_spec(tool):
                return [sys.executable, "-m", tool]
        except (ImportError, ValueError):
            pass
        return None

    # ---- main entry ---------------------------------------------------------- #

    def analyze(self, repo: Any) -> StaticReport:
        """Analyse a Repository (or anything with list_files/read_file/root). Never raises."""
        report = StaticReport()
        try:
            self._analyze(repo, report)
        except Exception as e:                            # bookkeeping must not crash a run
            logger.error("Static analysis failed unexpectedly: %s: %s", type(e).__name__, e)
            logger.debug("details", exc_info=True)
            report.notes.append(f"static analysis aborted: {type(e).__name__}: {e}")
        return report

    def _analyze(self, repo: Any, report: StaticReport) -> None:
        files = repo.list_files(text_only=True)
        py_files = [p for p in files if p.endswith(".py")]
        sources = {}
        for path in files:
            try:
                sources[path] = repo.read_file(path, errors="replace")
            except Exception:
                continue
        py_sources = {p: sources[p] for p in py_files if p in sources}
        report.python_files = len(py_sources)
        report.lines = sum(t.count("\n") + 1 for t in sources.values() if t)
        py_lines = sum(t.count("\n") + 1 for t in py_sources.values() if t)

        if not py_sources:
            report.notes.append("no Python files: only line counts available for this language")
            for tool in ("ruff", "mypy", "bandit"):
                report.tools[tool] = ToolReport(tool, skipped_reason="no Python files")
            report.metrics = {"compiles": None, "code_quality": None, "security": None,
                              "documentation": self._documentation(sources, None)}
            return

        # 1. compile check
        ratio, errors = check_compiles(py_sources)
        report.metrics["compiles"] = ratio
        report.compile_errors = errors
        parsed = {p: analyse_python_source(t) for p, t in py_sources.items()}

        # 2. complexity + docstrings (non-test files only)
        complexities, documented, documentable = [], 0, 0
        for path, facts in parsed.items():
            if facts is None or is_test_path(path):
                continue
            complexities += facts["complexities"]
            documented += facts["documented"]
            documentable += facts["documentable"]
        report.avg_complexity = sum(complexities) / len(complexities) if complexities else None
        report.docstring_ratio = documented / documentable if documentable else None

        # 3. external tools
        lint = self._run_ruff(repo, report, py_lines)
        types = self._run_mypy(repo, report, py_lines)
        security = self._security(repo, report, py_sources)

        quality_sources = [n for n, v in (("ruff", lint), ("mypy", types)) if v is not None]
        if quality_sources:
            report.metrics["code_quality"] = score_quality(lint, types, score_complexity(report.avg_complexity))
            report.sources["code_quality"] = "+".join(quality_sources + (["complexity"] if complexities else []))
        else:
            report.metrics["code_quality"] = None
            report.notes.append("code_quality not measured: neither ruff nor mypy ran "
                                "(complexity alone is not a quality score)")
        report.metrics["security"] = security
        report.metrics["documentation"] = self._documentation(sources, report.docstring_ratio)
        report.sources["documentation"] = "docstrings+readme"

    # ---- documentation ------------------------------------------------------ #

    @staticmethod
    def _documentation(sources: dict, docstring_ratio: Optional[float]) -> Optional[float]:
        readmes = [p for p in sources if p.lower().split("/")[-1].startswith("readme") and "/" not in p]
        has_readme = bool(readmes)
        has_usage = None
        if has_readme:
            text = sources[readmes[0]].lower()
            has_usage = bool(re.search(r"\b(usage|install|run|getting started|how to)\b", text))
        return score_documentation(docstring_ratio, has_readme, has_usage if has_readme else False)

    # ---- tools ---------------------------------------------------------------- #

    def _execute(self, name: str, argv: list, repo: Any, report: StaticReport) -> Optional[CommandResult]:
        tool = report.tools.setdefault(name, ToolReport(name))
        timeout = float(self.settings.sandbox_timeout_seconds)
        result = self.runner(argv, str(repo.root), timeout)
        tool.seconds = round(result.seconds, 3)
        if result.error:
            tool.skipped_reason = f"could not start: {result.error}"
        elif result.timed_out:
            tool.skipped_reason = f"timed out after {timeout:.0f}s"
        else:
            tool.ran = True
            return result
        logger.warning("%s skipped: %s", name, tool.skipped_reason)
        return None

    def _missing(self, name: str, report: StaticReport) -> None:
        report.tools[name] = ToolReport(name, skipped_reason="not installed")
        logger.info("%s not installed; metric skipped", name)

    def _record(self, report: StaticReport, name: str, issues: list, root: str) -> None:
        for issue in issues:
            issue.path = _relative(issue.path, root)
        tool = report.tools[name]
        tool.issue_count = len(issues)
        for issue in issues:
            tool.by_severity[issue.severity] = tool.by_severity.get(issue.severity, 0) + 1
        report.issues.extend(issues[:MAX_ISSUES_KEPT])

    def _run_ruff(self, repo: Any, report: StaticReport, py_lines: int) -> Optional[float]:
        base = self.command_for("ruff")
        if not base:
            self._missing("ruff", report)
            return None
        result = self._execute("ruff", base + ["check", "--isolated", "--no-cache", "--output-format", "json", "."],
                               repo, report)
        if result is None:
            return None
        if result.returncode not in (0, 1):                # 2 = ruff itself failed
            report.tools["ruff"].ran = False
            report.tools["ruff"].skipped_reason = f"ruff error (exit {result.returncode}): {result.stderr.strip()[:150]}"
            return None
        try:
            issues = parse_ruff_json(result.stdout)
        except (ValueError, TypeError, AttributeError) as e:
            report.tools["ruff"].ran = False
            report.tools["ruff"].skipped_reason = f"unreadable ruff output: {e}"
            return None
        self._record(report, "ruff", issues, str(repo.root))
        return score_lint(len(issues), py_lines)

    def _run_mypy(self, repo: Any, report: StaticReport, py_lines: int) -> Optional[float]:
        base = self.command_for("mypy")
        if not base:
            self._missing("mypy", report)
            return None
        cache = os.path.join(tempfile.gettempdir(), "orchestrator-mypy-cache")
        argv = base + ["--config-file", os.devnull, "--ignore-missing-imports", "--no-error-summary",
                       "--no-color-output", "--show-column-numbers", "--cache-dir", cache, "."]
        result = self._execute("mypy", argv, repo, report)
        if result is None:
            return None
        if result.returncode not in (0, 1):                # 2 = mypy crashed / bad usage
            report.tools["mypy"].ran = False
            report.tools["mypy"].skipped_reason = (
                f"mypy error (exit {result.returncode}): {(result.stderr or result.stdout).strip()[:150]}")
            return None
        issues = parse_mypy_text(result.stdout)
        self._record(report, "mypy", issues, str(repo.root))
        return score_lint(len(issues), py_lines)

    def _security(self, repo: Any, report: StaticReport, py_sources: dict) -> Optional[float]:
        base = self.command_for("bandit")
        issues = None
        if base:
            excludes = ",".join(f"./{d}" for d in TEST_DIR_NAMES)
            result = self._execute("bandit", base + ["-r", ".", "-f", "json", "-q", "-x", excludes], repo, report)
            if result is not None:
                try:
                    if result.returncode not in (0, 1):
                        raise ValueError(f"exit {result.returncode}: {result.stderr.strip()[:120]}")
                    issues = parse_bandit_json(result.stdout)
                    self._record(report, "bandit", issues, str(repo.root))
                    report.sources["security"] = "bandit"
                except (ValueError, TypeError, AttributeError) as e:
                    report.tools["bandit"].ran = False
                    report.tools["bandit"].skipped_reason = f"unusable bandit output: {e}"
        else:
            self._missing("bandit", report)
        if issues is None:
            issues = builtin_security_scan(py_sources)
            report.tools["builtin_security"] = ToolReport("builtin_security", ran=True, issue_count=len(issues))
            self._record(report, "builtin_security", issues, str(repo.root))
            report.sources["security"] = "builtin"
            report.notes.append("security measured by the small built-in scan (bandit unavailable)")
        counts = {s: sum(1 for i in issues if i.severity == s) for s in ("high", "medium", "low")}
        return score_security(counts["high"], counts["medium"], counts["low"])


# --------------------------------------------------------------------------- #
# Self-test:  python -m evaluation.static_analysis    (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from context.repository import Repository
    from core.logger import setup_logging

    setup_logging("ERROR")

    def close(a, b, tol=1e-6):
        return abs(a - b) <= tol

    GOOD = '"""Greeter."""\n\n\ndef greet(name: str) -> str:\n    """Say hello."""\n    return f"Hello {name}"\n'
    RISKY = ('import os, pickle, subprocess\npassword = "hunter2hunter2"\n\n\ndef run(cmd):\n'
             '    subprocess.call(cmd, shell=True)\n    os.system(cmd)\n    # eval("ignored: comment")\n'
             '    return eval(cmd)\n')
    BRANCHY = ("def f(a, b, c):\n    if a and b or c:\n        for x in range(3):\n            if x:\n                pass\n"
               "    try:\n        pass\n    except ValueError:\n        pass\n    return [i for i in range(3) if i]\n")

    tmp = tempfile.TemporaryDirectory()

    def make_repo(name, files):
        repo = Repository(Path(tmp.name) / name)
        repo.write_files(files)
        return repo

    # 1. Stdlib pieces: compile check, complexity, docstrings, security scan
    ratio, errs = check_compiles({"a.py": GOOD, "b.py": "def (:\n"})
    assert ratio == 0.5 and errs[0][0] == "b.py" and "SyntaxError" in errs[0][1] and check_compiles({}) == (None, [])
    facts = analyse_python_source(BRANCHY)
    assert facts["complexities"] == [9], facts           # 1 + if + (and, or) 2 + for + if + except + comprehension(1+1 if) 2 = 9
    assert analyse_python_source("def (:") is None
    facts = analyse_python_source(GOOD)
    assert (facts["documented"], facts["documentable"]) == (2, 2)
    found = builtin_security_scan({"app.py": RISKY, "tests/test_app.py": RISKY})
    assert {i.code for i in found} == {"B-EVAL", "B-SHELL", "B-OSSYSTEM", "B-SECRET"}, {i.code for i in found}
    assert all(i.path == "app.py" for i in found) and not any(i.line == 8 for i in found)   # comment + tests skipped
    print("PASS 1: compile check, complexity, docstring coverage, built-in security scan")

    # 2. Parsers
    ruff_out = json.dumps([{"filename": "/w/app.py", "code": "F401", "message": "unused import",
                            "location": {"row": 3, "column": 1}}])
    assert parse_ruff_json(ruff_out)[0].code == "F401" and parse_ruff_json("")[0:1] == []
    mypy_out = ('app.py:4:12: error: Incompatible return value type  [return-value]\n'
                'app.py:4:12: note: something\nSuccess\napp.py:9: error: Name "x" is not defined  [name-defined]\n')
    parsed = parse_mypy_text(mypy_out)
    assert [(i.line, i.code) for i in parsed] == [(4, "return-value"), (9, "name-defined")]
    bandit_out = json.dumps({"results": [{"filename": "./a.py", "line_number": 2, "test_id": "B602",
                                          "issue_text": "shell", "issue_severity": "HIGH"}]})
    assert parse_bandit_json(bandit_out)[0].severity == "high"
    assert _relative("/w/app.py", "/w") == "app.py" and _relative("./a.py", "/w") == "a.py"
    print("PASS 2: ruff / mypy / bandit output parsers")

    # 3. All tools "installed": metrics computed from their output (fake runner, no real tools needed)
    calls = []

    def fake_runner(argv, cwd, timeout):
        calls.append(argv)
        tool = os.path.basename(argv[0])
        if tool == "ruff":
            return CommandResult(1, ruff_out, "", False, 0.1)
        if tool == "mypy":
            return CommandResult(1, mypy_out, "", False, 0.2)
        if tool == "bandit":
            return CommandResult(1, bandit_out, "", False, 0.1)
        return CommandResult(None, error="unknown")

    repo = make_repo("full", {"app.py": GOOD, "README.md": "# App\n\n## Usage\nrun it\n",
                              "tests/test_app.py": "def test_x():\n    assert True\n"})
    az = StaticAnalyzer(commands={"ruff": ["/bin/ruff"], "mypy": ["/bin/mypy"], "bandit": ["/bin/bandit"]},
                        runner=fake_runner)
    rep = az.analyze(repo)
    py_lines = sum(t.count("\n") + 1 for p, t in repo.read_text_files(["app.py", "tests/test_app.py"]).items())
    assert rep.metrics["compiles"] == 1.0 and rep.python_files == 2
    assert close(rep.metrics["security"], 0.6)                                  # one HIGH finding
    lint, typ = score_lint(1, py_lines), score_lint(2, py_lines)
    assert close(rep.metrics["code_quality"], score_quality(lint, typ, score_complexity(1.0)))
    assert rep.sources["code_quality"] == "ruff+mypy+complexity" and rep.sources["security"] == "bandit"
    assert rep.metrics["documentation"] == 1.0 and rep.skipped_tools() == []
    ruff_cmd = next(c for c in calls if c[0] == "/bin/ruff")
    assert "--isolated" in ruff_cmd and "json" in ruff_cmd                      # project config cannot be loaded
    mypy_cmd = next(c for c in calls if c[0] == "/bin/mypy")
    assert mypy_cmd[mypy_cmd.index("--config-file") + 1] == os.devnull
    assert rep.issues[0].path == "app.py" or rep.issues[0].path.endswith("app.py")
    print("PASS 3: metrics from ruff + mypy + bandit ->", rep.summary())

    # 4. No tools installed: skipped, not failed; security falls back to the built-in scan
    az = StaticAnalyzer(commands={"ruff": None, "mypy": None, "bandit": None}, runner=fake_runner)
    rep = az.analyze(make_repo("bare", {"app.py": RISKY, "util.py": GOOD}))
    assert rep.metrics["code_quality"] is None and rep.skipped_tools() == ["bandit", "mypy", "ruff"]
    assert rep.tools["ruff"].skipped_reason == "not installed"
    assert rep.sources["security"] == "builtin" and rep.metrics["security"] == 0.0     # 3 high + 1 medium
    assert any("code_quality not measured" in n for n in rep.notes)
    assert rep.metrics["compiles"] == 1.0 and rep.avg_complexity is not None
    print("PASS 4: missing tools skipped; code_quality None; security via built-in scan ->", rep.metrics["security"])

    # 5. Tool failures: timeout, crash, garbage output, start error -> skipped with a reason
    def bad_runner(argv, cwd, timeout):
        tool = os.path.basename(argv[0])
        if tool == "ruff":
            return CommandResult(None, "", "", True, timeout)
        if tool == "mypy":
            return CommandResult(2, "", "internal error", False, 0.1)
        return CommandResult(1, "<<not json>>", "", False, 0.1)

    az = StaticAnalyzer(commands={"ruff": ["ruff"], "mypy": ["mypy"], "bandit": ["bandit"]}, runner=bad_runner)
    rep = az.analyze(make_repo("flaky", {"a.py": GOOD}))
    assert "timed out" in rep.tools["ruff"].skipped_reason and "exit 2" in rep.tools["mypy"].skipped_reason
    assert "unusable" in rep.tools["bandit"].skipped_reason and rep.sources["security"] == "builtin"
    assert rep.metrics["code_quality"] is None and rep.metrics["security"] == 1.0
    print("PASS 5: timeouts / crashes / garbage output become skips, with reasons")

    # 6. Syntax-error project, empty project, non-Python project
    rep = StaticAnalyzer(commands={"ruff": None, "mypy": None, "bandit": None}).analyze(
        make_repo("syn", {"ok.py": GOOD, "bad.py": "def broken(:\n    pass\n"}))
    assert rep.metrics["compiles"] == 0.5 and rep.compile_errors[0][0] == "bad.py"
    rep = StaticAnalyzer(commands={}).analyze(make_repo("js", {"index.js": "console.log(1)\n"}))
    assert rep.metrics["compiles"] is None and rep.metrics["security"] is None and rep.python_files == 0
    assert rep.tools["ruff"].skipped_reason == "no Python files" and rep.lines == 2
    rep = StaticAnalyzer(commands={}).analyze(make_repo("empty", {}))
    assert rep.python_files == 0 and rep.metrics["compiles"] is None
    print("PASS 6: syntax errors counted, non-Python and empty projects handled")

    # 7. Documentation: docstring ratio + README + usage
    undocumented = "def a():\n    pass\n\n\nclass B:\n    pass\n"
    az = StaticAnalyzer(commands={})
    low = az.analyze(make_repo("nodoc", {"m.py": undocumented})).metrics["documentation"]
    high = az.analyze(make_repo("doc", {"m.py": GOOD, "README.md": "Install it, then run it. Usage below."})).metrics["documentation"]
    assert low == 0.0 and high == 1.0 and 0.0 <= az.analyze(make_repo("mid", {"m.py": GOOD, "README.md": "hello"})
                                                           ).metrics["documentation"] < 1.0
    print(f"PASS 7: documentation score {low} (none) -> {high} (docstrings + README + usage)")

    # 8. run_command: real subprocess, timeout kills the process group, missing binary is an error not an exception
    r = run_command([sys.executable, "-c", "print('hi')"], tmp.name, 20)
    assert r.returncode == 0 and r.stdout.strip() == "hi"
    t0 = time.monotonic()
    r = run_command([sys.executable, "-c", "import time; time.sleep(30)"], tmp.name, 1)
    assert r.timed_out and r.returncode is None and time.monotonic() - t0 < 10
    r = run_command(["definitely-not-a-real-binary-xyz"], tmp.name, 5)
    assert r.error and r.returncode is None
    os.environ["ANTHROPIC_API_KEY"] = "sk-secret"
    try:
        r = run_command([sys.executable, "-c", "import os; print(os.environ.get('ANTHROPIC_API_KEY', 'absent'))"], tmp.name, 20)
        assert r.stdout.strip() == "absent"                                    # API keys never reach tools
    finally:
        del os.environ["ANTHROPIC_API_KEY"]
    print("PASS 8: run_command: timeout kills the process, errors are values, API keys are scrubbed")

    # 9. Real tools, if present on this machine (ruff / mypy / bandit): skipped otherwise
    real = StaticAnalyzer()
    available = [t for t in ("ruff", "mypy", "bandit") if real.command_for(t)]
    if available:
        rep = real.analyze(make_repo("real", {"app.py": "import os\n\n\ndef f(x: int) -> str:\n    return x\n"}))
        if real.command_for("ruff"):
            assert rep.tools["ruff"].ran and rep.tools["ruff"].issue_count >= 1, rep.tools["ruff"]
        if real.command_for("mypy"):
            assert rep.tools["mypy"].ran and rep.tools["mypy"].issue_count >= 1, rep.tools["mypy"]
        print("PASS 9: real tools worked ->", ", ".join(available), "|", rep.summary())
    else:
        print("PASS 9: no real tools installed here; real-tool check skipped (install ruff/mypy/bandit to enable)")

    # 10. Report is JSON-serialisable; analyze() never raises
    json.dumps(rep.to_dict())

    class Broken:
        root = Path(tmp.name)

        def list_files(self, **kw):
            raise RuntimeError("disk on fire")

    assert any("aborted" in n for n in StaticAnalyzer(commands={}).analyze(Broken()).notes)
    print("PASS 10: report serialises to JSON; unexpected failures are contained")