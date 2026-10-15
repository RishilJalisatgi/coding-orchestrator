"""
evaluation/tests.py

Runs a candidate's tests (and the orchestrator's own acceptance tests) in a sandboxed subprocess.

    from context.repository import Repository
    from evaluation.tests import TestRunner

    runner = TestRunner(settings.orchestrator)
    report = runner.run(Repository(candidate_dir),
                        acceptance_tests={"test_acceptance.py": ACCEPTANCE_SOURCE})   # acceptance is optional
    report.metrics()                    # {"test_pass_rate": 0.9, "checks": 1.0}  (None = nothing measured)
    report.candidate                    # SuiteResult: passed/failed/errors/skipped, failures, seconds...
    report.acceptance
    print(report.failure_summary())     # short text to hand to the reviewer / coder for a revision

Two kinds of suites, deliberately separate:
  * candidate tests   -> `test_pass_rate`   (the model wrote them, so they only show it is self-consistent)
  * acceptance tests  -> `checks`           (written by the orchestrator from the requirement; the stronger
                                             signal, and what `scoring.score_correctness(checks=...)` takes)

Isolation:
  * every run happens in a throw-away COPY of the workspace; the original is never touched
  * hard timeout, own process group (killed on timeout), scrubbed environment (no API keys),
    HOME/TMPDIR pointed at the temp directory, pytest config files of the candidate ignored
  * acceptance tests are written into the copy after the candidate's files, in their own directory
This limits accidents, not a determined attacker; run the whole orchestrator in a container if you
evaluate code you do not trust.

pytest is used when available (results read from its JUnit XML, which is exact). Otherwise it falls back
to `python -m unittest`. If neither can run, the suite is reported as skipped and the metric is None.
A suite that times out scores 0.0 (hanging code is wrong code); "no tests found" is None, not 0 --
callers that REQUIRE tests (e.g. the lite benchmark asks for them) can use `report.candidate.no_tests`.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from config.settings import OrchestratorSettings
from core.logger import get_logger
from evaluation.scoring import score_tests
from evaluation.static_analysis import CommandResult, Runner, run_command, safe_env

logger = get_logger("evaluation.tests")

MAX_JUNIT_BYTES = 5_000_000
MAX_FAILURES_KEPT = 25
OUTPUT_TAIL_CHARS = 3000
ACCEPTANCE_DIR = "_acceptance_tests"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class Failure:
    name: str
    kind: str                      # failure | error
    message: str = ""


@dataclass
class SuiteResult:
    label: str                     # "candidate" | "acceptance"
    tool: str = ""                 # pytest | unittest | ""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    seconds: float = 0.0
    timed_out: bool = False
    no_tests: bool = False
    exit_code: Optional[int] = None
    failures: list = field(default_factory=list)
    output_tail: str = ""
    skipped_reason: str = ""       # the suite could not be run at all (tool missing, could not start)

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def ran(self) -> bool:
        return not self.skipped_reason

    @property
    def pass_rate(self) -> Optional[float]:
        """0..1, or None when nothing was measured. A timeout is 0.0."""
        if self.skipped_reason:
            return None
        if self.timed_out:
            return 0.0
        if self.no_tests:
            return None
        return score_tests(self.passed, self.failed, self.errors)

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if k != "failures"}
        data["failures"] = [f.__dict__ for f in self.failures]
        data["pass_rate"] = self.pass_rate
        return data

    def line(self) -> str:
        if self.skipped_reason:
            return f"{self.label}: not run ({self.skipped_reason})"
        if self.timed_out:
            return f"{self.label}: TIMED OUT after {self.seconds:.0f}s ({self.tool})"
        if self.no_tests:
            return f"{self.label}: no tests found ({self.tool})"
        return (f"{self.label}: {self.passed} passed, {self.failed} failed, {self.errors} errors, "
                f"{self.skipped} skipped ({self.tool}, {self.seconds:.1f}s)")


@dataclass
class TestReport:
    __test__ = False                                  # keep pytest from collecting this class by name
    candidate: Optional[SuiteResult] = None
    acceptance: Optional[SuiteResult] = None

    def metrics(self) -> dict:
        return {"test_pass_rate": self.candidate.pass_rate if self.candidate else None,
                "checks": self.acceptance.pass_rate if self.acceptance else None}

    def suites(self) -> list:
        return [s for s in (self.candidate, self.acceptance) if s]

    def failure_summary(self, max_items: int = 10, max_chars: int = 3000) -> str:
        """Compact text for a revision prompt: what failed and why (acceptance first)."""
        lines = []
        for suite in (self.acceptance, self.candidate):
            if not suite:
                continue
            lines.append(suite.line())
            for f in suite.failures[:max_items]:
                lines.append(f"  - {f.kind}: {f.name}: {f.message}" if f.message else f"  - {f.kind}: {f.name}")
            if len(suite.failures) > max_items:
                lines.append(f"  ... and {len(suite.failures) - max_items} more")
            if suite.timed_out and suite.output_tail:
                lines.append("  last output:\n    " + suite.output_tail[-500:].replace("\n", "\n    "))
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 15] + "\n...(truncated)"

    def to_dict(self) -> dict:
        return {"candidate": self.candidate.to_dict() if self.candidate else None,
                "acceptance": self.acceptance.to_dict() if self.acceptance else None,
                "metrics": self.metrics()}


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #

def parse_junit(path: Path) -> Optional[dict]:
    """Counts and failures from a pytest JUnit XML file. None if missing/unreadable/too big."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_JUNIT_BYTES:
            return None
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "failures": []}
    for case in root.iter("testcase"):
        name = ((case.get("classname") or "") + "::" + (case.get("name") or "")).strip(":")
        bad = case.find("failure")
        kind = "failure"
        if bad is None:
            bad = case.find("error")
            kind = "error"
        if bad is not None:
            counts["failed" if kind == "failure" else "errors"] += 1
            message = (bad.get("message") or (bad.text or "")).strip().splitlines()
            counts["failures"].append(Failure(name, kind, (message[0] if message else "")[:300]))
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return counts


_UT_RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
_UT_RESULT = re.compile(r"^(OK|FAILED)(?: \((.*?)\))?\s*$", re.M)
_UT_PROBLEM = re.compile(r"^(FAIL|ERROR): (.+)$", re.M)


def parse_unittest(output: str) -> Optional[dict]:
    """Counts from `python -m unittest` output (it writes to stderr). None if no summary line was found."""
    ran = _UT_RAN.search(output or "")
    if not ran:
        return None
    total = int(ran.group(1))
    failures = errors = skipped = 0
    result = _UT_RESULT.search(output)
    if result and result.group(2):
        for key, value in re.findall(r"(\w+)=(\d+)", result.group(2)):
            if key == "failures":
                failures = int(value)
            elif key == "errors":
                errors = int(value)
            elif key == "skipped":
                skipped = int(value)
    problems = [Failure(m.group(2).strip(), "failure" if m.group(1) == "FAIL" else "error")
                for m in _UT_PROBLEM.finditer(output)]
    passed = max(0, total - failures - errors - skipped)
    return {"passed": passed, "failed": failures, "errors": errors, "skipped": skipped, "failures": problems,
            "total": total}


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #

class TestRunner:
    __test__ = False

    def __init__(self, settings: Optional[OrchestratorSettings] = None, runner: Optional[Runner] = None,
                 commands: Optional[dict] = None):
        """
        runner:   how commands are executed (tests inject a fake).
        commands: {"pytest": [argv...] | None} to bypass discovery (tests); None = pytest missing.
        """
        self.settings = settings or OrchestratorSettings()
        self.runner: Runner = runner or run_command
        self._commands = commands

    def pytest_command(self) -> Optional[list]:
        if self._commands is not None:
            return self._commands.get("pytest")
        try:
            if importlib.util.find_spec("pytest"):
                return [sys.executable, "-m", "pytest"]
        except (ImportError, ValueError):
            pass
        exe = shutil.which("pytest")
        return [exe] if exe else None

    # ---- public API ----------------------------------------------------------- #

    def run(self, repo: Any, acceptance_tests: Optional[Mapping] = None, run_candidate_tests: bool = True,
            timeout: Optional[float] = None) -> TestReport:
        """Run the candidate's own tests and/or the orchestrator's acceptance tests. Never raises."""
        report = TestReport()
        if run_candidate_tests:
            report.candidate = self._suite(repo, "candidate", None, timeout)
        if acceptance_tests:
            report.acceptance = self._suite(repo, "acceptance", acceptance_tests, timeout)
        return report

    # ---- one suite ------------------------------------------------------------- #

    def _suite(self, repo: Any, label: str, acceptance: Optional[Mapping], timeout: Optional[float]) -> SuiteResult:
        result = SuiteResult(label=label)
        try:
            with tempfile.TemporaryDirectory(prefix="orch-tests-") as tmp:
                work = Path(tmp) / "work"
                scratch = Path(tmp) / "scratch"
                scratch.mkdir()
                copy = repo.copy_to(work)
                target = "."
                if acceptance:
                    target = ACCEPTANCE_DIR
                    self._write_acceptance(Path(copy.root), acceptance)
                self._execute(result, Path(copy.root), scratch, target, float(timeout or self.settings.sandbox_timeout_seconds))
        except Exception as e:                           # a harness problem must not crash the orchestrator
            logger.error("Running %s tests failed: %s: %s", label, type(e).__name__, e)
            result.skipped_reason = f"harness error: {type(e).__name__}: {e}"[:200]
        return result

    @staticmethod
    def _write_acceptance(root: Path, tests: Mapping) -> None:
        folder = root / ACCEPTANCE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("", encoding="utf-8")
        for name, source in tests.items():
            safe = Path(str(name).replace("\\", "/")).name            # basename only: no path tricks
            if not safe.endswith(".py"):
                safe += ".py"
            (folder / safe).write_text(str(source), encoding="utf-8")

    @staticmethod
    def _ensure_packages(root: Path) -> None:
        """
        unittest only descends into packages, but model-written code usually has a tests/ folder without __init__.py.
        In the (throwaway) copy, make every folder that holds a test*.py importable so discovery finds them.
        """
        skip = {".git", "__pycache__", "node_modules", "venv", ".venv", "env", "site-packages"}
        for test_file in root.rglob("test*.py"):
            rel = test_file.relative_to(root)
            if any(part in skip for part in rel.parts):
                continue
            folder = test_file.parent
            while folder != root:
                marker = folder / "__init__.py"
                if not marker.exists():
                    marker.write_text("", encoding="utf-8")
                folder = folder.parent

    def _env(self, root: Path, scratch: Path) -> dict:
        paths = [str(root)] + ([str(root / "src")] if (root / "src").is_dir() else [])
        return safe_env({"PYTHONPATH": os.pathsep.join(paths), "HOME": str(scratch), "TMPDIR": str(scratch),
                         "TEMP": str(scratch), "TMP": str(scratch)})

    def _execute(self, result: SuiteResult, root: Path, scratch: Path, target: str, timeout: float) -> None:
        env = self._env(root, scratch)
        pytest = self.pytest_command()
        junit = scratch / "junit.xml"
        if pytest:
            result.tool = "pytest"
            argv = pytest + ["-q", "-p", "no:cacheprovider", "-c", os.devnull, "--rootdir", str(root),
                             "--tb=short", "--no-header", "-o", "junit_family=xunit2",
                             f"--junitxml={junit}", target]
        else:
            result.tool = "unittest"
            self._ensure_packages(root)
            argv = [sys.executable, "-m", "unittest", "discover", "-s", target, "-t", ".", "-p", "test*.py"]

        run = _run_with_env(self.runner, argv, str(root), timeout, env)
        result.seconds = round(run.seconds, 3)
        result.exit_code = run.returncode
        result.output_tail = ((run.stdout or "") + ("\n" if run.stdout and run.stderr else "") + (run.stderr or ""))[-OUTPUT_TAIL_CHARS:]

        if run.error:
            result.skipped_reason = f"could not start {result.tool}: {run.error}"[:200]
            return
        if run.timed_out:
            result.timed_out = True
            return

        counts = parse_junit(junit) if pytest else parse_unittest((run.stderr or "") + "\n" + (run.stdout or ""))
        if counts is None:
            if pytest and run.returncode == 5:                # pytest: "no tests collected"
                result.no_tests = True
            elif pytest:
                result.skipped_reason = f"pytest produced no readable results (exit {run.returncode})"
            else:
                result.no_tests = run.returncode in (0, 5)
                if not result.no_tests:
                    result.skipped_reason = f"unittest produced no readable results (exit {run.returncode})"
            return
        result.passed, result.failed = counts["passed"], counts["failed"]
        result.errors, result.skipped = counts["errors"], counts["skipped"]
        result.failures = counts["failures"][:MAX_FAILURES_KEPT]
        if result.total == 0 and result.skipped == 0:
            result.no_tests = True
        elif result.total == 0:                               # everything skipped: nothing was verified
            result.no_tests = True


def _run_with_env(runner: Runner, argv: list, cwd: str, timeout: float, env: dict) -> CommandResult:
    """Call the runner with an environment when it supports one (the default does; simple fakes may not)."""
    try:
        return runner(argv, cwd, timeout, env)                # type: ignore[call-arg]
    except TypeError:
        return runner(argv, cwd, timeout)


# --------------------------------------------------------------------------- #
# Self-test:  python -m evaluation.tests      (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json
    import time

    from context.repository import Repository
    from core.logger import setup_logging

    setup_logging("ERROR")

    tmp = tempfile.TemporaryDirectory()

    def make_repo(name, files):
        repo = Repository(Path(tmp.name) / name)
        if files:
            repo.write_files(files)
        return repo

    CALC = "def add(a, b):\n    return a + b\n\n\ndef div(a, b):\n    return a // b\n"
    PYTEST_MIXED = ("from calc import add, div\nimport pytest\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\n"
                    "def test_add_neg():\n    assert add(-1, -1) == -2\n\n\ndef test_div_wrong():\n    assert div(7, 2) == 3.5\n\n\n"
                    "def test_boom():\n    raise RuntimeError('kaboom')\n\n\n"
                    "@pytest.mark.skip(reason='later')\ndef test_skipped():\n    pass\n")

    # 1. Parsers (no processes involved)
    xml_path = Path(tmp.name) / "j.xml"
    xml_path.write_text(
        '<testsuites><testsuite tests="4"><testcase classname="t" name="ok"/>'
        '<testcase classname="t" name="bad"><failure message="assert 1 == 2">trace</failure></testcase>'
        '<testcase classname="t" name="err"><error message="RuntimeError: x"/></testcase>'
        '<testcase classname="t" name="skip"><skipped/></testcase></testsuite></testsuites>', encoding="utf-8")
    c = parse_junit(xml_path)
    assert (c["passed"], c["failed"], c["errors"], c["skipped"]) == (1, 1, 1, 1)
    assert c["failures"][0].name == "t::bad" and c["failures"][0].message == "assert 1 == 2"
    assert parse_junit(Path(tmp.name) / "missing.xml") is None
    xml_path.write_text("<not xml", encoding="utf-8")
    assert parse_junit(xml_path) is None
    ut = parse_unittest("....F.E.s\n======\nFAIL: test_a (m.T)\nERROR: test_b (m.T)\n\nRan 9 tests in 0.1s\n\n"
                        "FAILED (failures=1, errors=1, skipped=1)\n")
    assert (ut["passed"], ut["failed"], ut["errors"], ut["skipped"]) == (6, 1, 1, 1) and len(ut["failures"]) == 2
    assert parse_unittest("Ran 3 tests in 0.0s\n\nOK\n")["passed"] == 3 and parse_unittest("garbage") is None
    print("PASS 1: JUnit XML and unittest output parsers")

    # 2. Metrics semantics
    assert SuiteResult("candidate", passed=9, failed=1).pass_rate == 0.9
    assert SuiteResult("candidate", timed_out=True).pass_rate == 0.0
    assert SuiteResult("candidate", no_tests=True).pass_rate is None
    assert SuiteResult("candidate", skipped_reason="x").pass_rate is None
    assert SuiteResult("candidate", passed=3, skipped=5).pass_rate == 1.0
    print("PASS 2: pass_rate: normal, timeout = 0.0, no tests / not run = None")

    # 3. Real pytest, mixed results; original workspace untouched; failure details captured
    pytest_here = TestRunner().pytest_command()
    if pytest_here:
        repo = make_repo("mixed", {"calc.py": CALC, "test_calc.py": PYTEST_MIXED})
        before = repo.snapshot()
        rep = TestRunner().run(repo)
        s = rep.candidate
        assert s.tool == "pytest" and (s.passed, s.failed, s.errors, s.skipped) == (2, 2, 0, 1), s.to_dict()      # an exception inside a test is a failure for pytest
        assert abs(s.pass_rate - 0.5) < 1e-9 and rep.metrics() == {"test_pass_rate": 0.5, "checks": None}
        names = {f.name.split("::")[-1]: f.kind for f in s.failures}
        assert names == {"test_div_wrong": "failure", "test_boom": "failure"}, names
        assert not repo.diff(before).changed                                     # runs in a copy
        assert "test_div_wrong" in rep.failure_summary()
        print("PASS 3: real pytest ->", s.line())
    else:
        print("PASS 3: pytest not installed here; real-pytest check skipped")

    # 4. Acceptance tests are injected into a copy, run separately, and cannot be shadowed by candidate files
    if pytest_here:
        repo = make_repo("accept", {"calc.py": CALC, "test_calc.py": "def test_own():\n    assert True\n"})
        acceptance = {"test_requirements.py": "from calc import add\n\n\ndef test_req_add():\n    assert add(2, 2) == 4\n\n\n"
                                               "def test_req_div_float():\n    from calc import div\n    assert div(1, 2) == 0.5\n",
                      "../evil/test_escape.py": "def test_escape():\n    assert True\n"}
        rep = TestRunner().run(repo, acceptance_tests=acceptance)
        assert rep.candidate.pass_rate == 1.0
        assert (rep.acceptance.passed, rep.acceptance.failed) == (2, 1), rep.acceptance.to_dict()   # escape file ran as a plain test
        assert abs(rep.metrics()["checks"] - 2 / 3) < 1e-9
        assert not repo.exists(ACCEPTANCE_DIR + "/test_requirements.py")                          # never written to the original
        assert "acceptance" in rep.failure_summary().splitlines()[0]
        print("PASS 4: acceptance suite separate from candidate suite ->", rep.acceptance.line())
    else:
        print("PASS 4: skipped (no pytest)")

    # 5. Timeout: infinite loop is killed and scores 0.0; collection error / no tests handled
    if pytest_here:
        repo = make_repo("hang", {"test_hang.py": "import time\n\n\ndef test_hang():\n    time.sleep(60)\n"})
        t0 = time.monotonic()
        rep = TestRunner().run(repo, timeout=3)
        assert rep.candidate.timed_out and rep.candidate.pass_rate == 0.0 and time.monotonic() - t0 < 30
        broken = TestRunner().run(make_repo("collect", {"test_x.py": "import does_not_exist\n\n\ndef test_a():\n    pass\n"}))
        assert broken.candidate.errors == 1 and broken.candidate.pass_rate == 0.0, broken.candidate.to_dict()
        none = TestRunner().run(make_repo("notests", {"lib.py": "x = 1\n"}))
        assert none.candidate.no_tests and none.metrics()["test_pass_rate"] is None
        print("PASS 5: hang killed (0.0), collection error counted, 'no tests' is None")
    else:
        print("PASS 5: skipped (no pytest)")

    # 6. unittest fallback when pytest is missing (real subprocess)
    UT = ("import unittest\nfrom calc import add\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n"
          "        self.assertEqual(add(1, 1), 2)\n\n    def test_bad(self):\n        self.assertEqual(add(1, 1), 3)\n")
    rep = TestRunner(commands={"pytest": None}).run(make_repo("ut", {"calc.py": CALC, "test_ut.py": UT}))
    s = rep.candidate
    assert s.tool == "unittest" and (s.passed, s.failed) == (1, 1) and s.pass_rate == 0.5, s.to_dict()
    ok_rep = TestRunner(commands={"pytest": None}).run(make_repo("ut_none", {"calc.py": CALC}))
    assert ok_rep.candidate.no_tests
    print("PASS 6: unittest fallback ->", s.line())

    # 7. Tool cannot start / unreadable output -> suite skipped with a reason (metric None), never an exception
    def cannot_start(argv, cwd, timeout, env=None):
        return CommandResult(None, error="FileNotFoundError: nope")

    rep = TestRunner(commands={"pytest": ["/no/such/pytest"]}, runner=cannot_start).run(make_repo("nostart", {"t.py": "x=1\n"}))
    assert rep.candidate.skipped_reason.startswith("could not start") and rep.metrics()["test_pass_rate"] is None

    def silent(argv, cwd, timeout, env=None):
        return CommandResult(3, "", "internal error", False, 0.1)

    rep = TestRunner(commands={"pytest": ["pytest"]}, runner=silent).run(make_repo("silent", {"t.py": "x=1\n"}))
    assert "no readable results" in rep.candidate.skipped_reason
    print("PASS 7: start failures and unreadable output become skips with a reason")

    # 8. Environment: API keys scrubbed, HOME/TMPDIR redirected, PYTHONPATH includes root and src/
    seen = {}

    def spy(argv, cwd, timeout, env=None):
        seen.update(env or {})
        seen["cwd"] = cwd
        return CommandResult(5, "", "", False, 0.0)

    os.environ["OPENAI_API_KEY"] = "sk-secret"
    try:
        TestRunner(commands={"pytest": ["pytest"]}, runner=spy).run(make_repo("envcheck", {"src/m.py": "x = 1\n"}))
    finally:
        del os.environ["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY" not in seen and seen["HOME"] == seen["TMPDIR"] and "orch-tests-" in seen["HOME"]
    assert seen["PYTHONPATH"].endswith("src") and seen["cwd"].endswith("work")
    print("PASS 8: sandbox environment (no API keys, temp HOME, PYTHONPATH with src/)")

    # 9. Harness failure is contained; temp copies are cleaned up; report serialises
    class BadRepo:
        root = Path(tmp.name)

        def copy_to(self, dest):
            raise OSError("disk full")

    rep = TestRunner(commands={"pytest": None}).run(BadRepo())
    assert "harness error" in rep.candidate.skipped_reason
    leftovers = [p for p in Path(tempfile.gettempdir()).glob("orch-tests-*")]
    assert not leftovers, leftovers
    json.dumps(rep.to_dict())
    print("PASS 9: harness errors contained, temp directories cleaned, report is JSON")

    # 10. unittest fallback finds tests in a tests/ folder WITHOUT __init__.py (the usual model-written layout),
    #     and the candidate's workspace is left untouched
    pkg_repo = make_repo("pkgless", {"app/rules.py": "def f():\n    return 1\n",
                                     "tests/test_rules.py": "import unittest\nfrom app.rules import f\n\n\nclass T(unittest.TestCase):\n"
                                                            "    def test_f(self):\n        self.assertEqual(f(), 1)\n"})
    s10 = TestRunner(commands={"pytest": None}).run(pkg_repo).candidate
    assert s10.tool == "unittest" and (s10.passed, s10.failed, s10.no_tests) == (1, 0, False), s10.to_dict()
    assert not pkg_repo.exists("tests/__init__.py") and not pkg_repo.exists("app/__init__.py")
    print("PASS 10: unittest fallback discovers tests in folders without __init__.py; workspace untouched")