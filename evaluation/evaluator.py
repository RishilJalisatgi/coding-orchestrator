"""
evaluation/evaluator.py

The full evaluation pipeline for one candidate workspace (or several, side by side). It only COORDINATES
the other evaluation modules and turns their reports into one 0..1 score; it never executes candidate code
itself (that happens in the sandboxed runners).

    from context.repository import Repository
    from evaluation.evaluator import Evaluator

    ev = Evaluator(settings.orchestrator)

    # one candidate
    result = ev.evaluate(Repository(candidate_dir), task_type="backend_api", label="gpt",
                         acceptance_tests={"test_acceptance.py": SOURCE},      # optional, written by the orchestrator
                         run_specs=[RunSpec(command=["python", "main.py"], expected_stdout="ok\\n")])  # optional
    result.final_score          # 0..1
    result.metrics              # {"correctness": 0.9, "test_pass_rate": 1.0, "runtime": None, ...}
    result.summary()            # one line for logs
    result.findings()           # text for a revision prompt / reviewer: what is broken and why
    result.save(path)           # JSON report (atomic write)

    # a lite-benchmark round: same workload for every candidate; runtime/memory are scored RELATIVE to the best
    results = ev.evaluate_many({"gpt": repo_a, "gemini": repo_b, "local": None}, task_type="desktop_application")
    rank_candidates({m: r.score for m, r in results.items()})

    # optional LLM judge on top (only the subjective metrics move, and by at most `judge_share`)
    result = ev.apply_judge(result, {"architecture": 0.8, "code_quality": 0.7})

Pipeline (each stage is optional and failure-contained; a stage that cannot run is recorded, never guessed):
    static analysis  -> compiles, code_quality, security, documentation
    import check     -> "runs" (does every module import?)
    tests            -> test_pass_rate (the candidate's own tests) and checks (the orchestrator's acceptance tests)
    run specs        -> runtime / memory (+ whether the workload produced the expected output)
    scoring          -> correctness = f(compiles, runs, checks); final score with the task type's weights
`architecture` is not measurable by tools: it stays unmeasured until a judge supplies it.

Never raises: an unexpected error in a stage is stored in `stage_errors` and the stage counts as not measured.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from config.settings import OrchestratorSettings
from core.logger import get_logger
from evaluation.runtime import ImportCheck, RunResult, RunSpec, RuntimeRunner
from evaluation.scoring import (ScoreResult, blend_judge, compute_score, failure_score, rank_candidates,
                                score_correctness)
from evaluation.static_analysis import StaticAnalyzer, StaticReport
from evaluation.tests import TestReport, TestRunner

logger = get_logger("evaluation.evaluator")

DEFAULT_FINDINGS_CHARS = 3000
MAX_STATIC_FINDINGS = 8
MAX_FAILED_IMPORTS = 5
MIN_RELATIVE_SECONDS = 0.25     # relative runtime/memory never use a "best" below these floors:
MIN_RELATIVE_MB = 32.0          # differences that small are interpreter start-up noise, not quality


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class Evaluation:
    label: str
    task_type: str
    score: ScoreResult
    static: Optional[StaticReport] = None
    tests: Optional[TestReport] = None
    imports: Optional[ImportCheck] = None
    runs: list = field(default_factory=list)            # [RunResult]
    stage_errors: dict = field(default_factory=dict)    # stage -> "ErrorType: message" (stage counted as not measured)
    files: int = 0
    seconds: float = 0.0
    empty: bool = False                                 # the candidate produced no files at all
    notes: list = field(default_factory=list)

    # ---- convenience ------------------------------------------------------------ #

    @property
    def final_score(self) -> float:
        return self.score.final_score

    @property
    def metrics(self) -> dict:
        return dict(self.score.metrics)

    @property
    def reliable(self) -> bool:
        return self.score.reliable

    def summary(self) -> str:
        """One line for logs and decision reasoning."""
        if self.empty:
            return f"{self.label}: 0.000 (no files produced)"
        shown = ", ".join(f"{k}={v:.2f}" for k, v in self.score.metrics.items() if v is not None)
        text = f"{self.label}: {self.final_score:.3f} [{shown or 'nothing measured'}]"
        text += f" coverage {self.score.coverage:.0%}"
        if self.score.gate_factor < 1.0:
            text += f", gated x{self.score.gate_factor:.2f}"
        if self.score.skipped:
            text += f", not measured: {', '.join(self.score.skipped)}"
        if self.stage_errors:
            text += f", stage errors: {', '.join(sorted(self.stage_errors))}"
        return text

    def findings(self, max_chars: int = DEFAULT_FINDINGS_CHARS) -> str:
        """
        What is wrong with the candidate, as compact text for a revision prompt or reviewer. Objective facts
        only (compile errors, import failures, failing tests, failed runs, serious static findings).
        """
        lines: list = []
        if self.empty:
            lines.append("No files were produced.")
        if self.static and self.static.compile_errors:
            lines.append("Files that do not compile:")
            lines += [f"  - {p}: {m}" for p, m in self.static.compile_errors[:MAX_FAILED_IMPORTS]]
        if self.imports:
            if self.imports.failed:
                lines.append("Modules that fail to import:")
                lines += [f"  - {m}: {err}" for m, err in self.imports.failed[:MAX_FAILED_IMPORTS]]
            if self.imports.timed_out:
                lines.append("Modules that hang on import: " + ", ".join(self.imports.timed_out[:MAX_FAILED_IMPORTS]))
        if self.tests:
            text = self.tests.failure_summary()
            if text:
                lines.append("Tests:")
                lines.append(text)
        for run in self.runs:
            if run.error and not run.runs:
                continue                                          # could not be started here: an environment issue
            if not run.completed:
                why = "timed out" if run.timed_out else (
                    "wrong output" if run.output_ok is False else f"exit code {run.exit_code}")
                tail = (run.stderr_tail or run.stdout_tail or "").strip()[-400:]
                lines.append(f"Run '{run.name}' failed ({why})" + (f":\n    {tail}" if tail else ""))
        if self.static:
            serious = [i for i in self.static.issues if i.severity in ("high", "medium")]
            rest = [i for i in self.static.issues if i.severity not in ("high", "medium") and i.tool == "mypy"]
            shown = (serious + rest)[:MAX_STATIC_FINDINGS]
            if shown:
                lines.append("Static analysis:")
                lines += [f"  - [{i.tool}/{i.severity}] {i.path}:{i.line} {i.code} {i.message}".rstrip() for i in shown]
        if not lines:
            lines.append("No objective problems found.")
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 15] + "\n...(truncated)"

    def to_dict(self) -> dict:
        return {
            "label": self.label, "task_type": self.task_type, "final_score": self.final_score,
            "score": self.score.to_dict(),
            "static": self.static.to_dict() if self.static else None,
            "tests": self.tests.to_dict() if self.tests else None,
            "imports": self.imports.to_dict() if self.imports else None,
            "runs": [r.to_dict() for r in self.runs],
            "stage_errors": dict(self.stage_errors), "files": self.files,
            "seconds": round(self.seconds, 3), "empty": self.empty, "notes": list(self.notes),
        }

    def save(self, path: "str | Path") -> bool:
        """Write the JSON report atomically. Returns False (and logs) instead of raising."""
        target = Path(path)
        tmp = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, default=str)
            os.replace(tmp, target)
            return True
        except Exception as exc:                                     # bookkeeping must never crash a run
            logger.warning("Could not save evaluation report %s: %s", target, exc)
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return False


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #

class Evaluator:
    def __init__(self, settings: Optional[OrchestratorSettings] = None,
                 static: Optional[StaticAnalyzer] = None, tests: Optional[TestRunner] = None,
                 runtime: Optional[RuntimeRunner] = None, run_repeats: int = 1,
                 require_tests: bool = False, judge_share: float = 0.25):
        """
        run_repeats:   fresh runs per RunSpec (median is used). 1 keeps benchmarks cheap; raise it for timing you trust.
        require_tests: a candidate that was asked for tests but wrote none gets test_pass_rate 0.0 instead of
                       "not measured" (use for the lite benchmark, where tests are part of the assignment).
        judge_share:   maximum influence of the LLM judge on each subjective metric (capped at 0.5 by scoring).
        The three runners are injectable (tests pass fakes).
        """
        self.settings = settings or OrchestratorSettings()
        self.static = static or StaticAnalyzer(self.settings)
        self.tests = tests or TestRunner(self.settings)
        self.runtime = runtime or RuntimeRunner(self.settings)
        self.run_repeats = max(1, int(run_repeats))
        self.require_tests = require_tests
        self.judge_share = judge_share

    # ---- public API --------------------------------------------------------------- #

    def evaluate(self, repo: Any, task_type: Optional[str] = None, *, label: str = "candidate",
                 acceptance_tests: Optional[Mapping] = None, run_specs: Optional[list] = None,
                 reference_seconds: Optional[float] = None, reference_mb: Optional[float] = None,
                 require_tests: Optional[bool] = None, run_candidate_tests: bool = True) -> Evaluation:
        """
        Evaluate one workspace. reference_seconds / reference_mb are absolute budgets for runtime / memory
        (1.0 at or under, 1/ratio above); without them runtime and memory stay unmeasured, except that a
        run which failed or timed out scores 0.0. Use evaluate_many() to compare candidates relatively.
        """
        ev = self._collect(repo, task_type, label, acceptance_tests, run_specs, run_candidate_tests)
        refs = {None: (reference_seconds, reference_mb)}
        self._score(ev, refs, self.require_tests if require_tests is None else require_tests)
        logger.info("Evaluated %s", ev.summary())
        return ev

    def evaluate_many(self, candidates: Mapping, task_type: Optional[str] = None, *,
                      acceptance_tests: Optional[Mapping] = None, run_specs: Optional[list] = None,
                      reference_seconds: Optional[float] = None, reference_mb: Optional[float] = None,
                      require_tests: Optional[bool] = None, run_candidate_tests: bool = True) -> dict:
        """
        Evaluate {model_id: Repository | None} with identical settings. A None / empty workspace is a failed
        candidate (score 0, with a reason). Unless absolute references are given, each run spec's runtime and
        memory are scored relative to the best candidate that completed it (best = 1.0).
        """
        collected: dict = {}
        for model_id, repo in candidates.items():
            if repo is None:
                collected[model_id] = self._empty(model_id, task_type, "model produced no workspace")
            else:
                collected[model_id] = self._collect(repo, task_type, str(model_id), acceptance_tests,
                                                    run_specs, run_candidate_tests)
        refs: dict = {None: (reference_seconds, reference_mb)}
        if reference_seconds is None and reference_mb is None:
            refs = self._relative_references(collected.values())
        strict = self.require_tests if require_tests is None else require_tests
        for ev in collected.values():
            self._score(ev, refs, strict)
            logger.info("Evaluated %s", ev.summary())
        return collected

    def apply_judge(self, evaluation: Evaluation, judge_scores: Mapping,
                    judge_share: Optional[float] = None) -> Evaluation:
        """
        Blend an LLM judge's opinion into a COPY of the evaluation. Only code_quality / architecture /
        documentation can change, each by at most `judge_share`; everything objective is untouched.
        """
        if evaluation.empty or not judge_scores:
            return evaluation
        out = copy.copy(evaluation)
        share = self.judge_share if judge_share is None else judge_share
        weights = self.settings.weights_for(evaluation.task_type)
        try:
            out.score = blend_judge(evaluation.score, judge_scores, weights, share)
        except Exception as exc:
            logger.warning("Judge blending failed (%s); keeping the objective score", exc)
            out.stage_errors = dict(evaluation.stage_errors, judge=f"{type(exc).__name__}: {exc}")
        return out

    @staticmethod
    def rank(evaluations: Mapping) -> list:
        """[(model_id, final_score)] best first, deterministic tie-breaks (coverage, correctness, id)."""
        return rank_candidates({m: e.score for m, e in evaluations.items()})

    @staticmethod
    def scores(evaluations: Mapping) -> dict:
        """{model_id: final_score}: the shape Scheduler.select_coder() expects."""
        return {m: e.final_score for m, e in evaluations.items()}

    # ---- stages --------------------------------------------------------------------- #

    def _empty(self, label: str, task_type: Optional[str], reason: str) -> Evaluation:
        return Evaluation(label=label, task_type=task_type or "", score=failure_score(reason),
                          empty=True, notes=[reason])

    def _stage(self, ev: Evaluation, name: str, fn):
        try:
            return fn()
        except Exception as exc:
            logger.error("Evaluation stage '%s' failed for %s: %s: %s", name, ev.label, type(exc).__name__, exc)
            ev.stage_errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    def _collect(self, repo: Any, task_type: Optional[str], label: str, acceptance_tests: Optional[Mapping],
                 run_specs: Optional[list], run_candidate_tests: bool) -> Evaluation:
        started = time.monotonic()
        ev = Evaluation(label=label, task_type=task_type or "", score=failure_score("not scored yet"))
        files = self._stage(ev, "list_files", repo.list_files)
        if not files:
            empty = self._empty(label, task_type, "no files produced" if files is not None
                                else "workspace could not be read")
            empty.stage_errors = ev.stage_errors
            return empty
        ev.files = len(files)
        ev.static = self._stage(ev, "static", lambda: self.static.analyze(repo))
        ev.imports = self._stage(ev, "import_check", lambda: self.runtime.import_check(repo))
        ev.tests = self._stage(ev, "tests", lambda: self.tests.run(
            repo, acceptance_tests=acceptance_tests, run_candidate_tests=run_candidate_tests))
        for spec in run_specs or []:
            result = self._stage(ev, f"run:{spec.name}", lambda s=spec: self.runtime.measure(
                repo, s, repeats=self.run_repeats))
            if result is not None:
                ev.runs.append(result)
        ev.seconds = time.monotonic() - started
        return ev

    @staticmethod
    def _attempted(runs: list) -> list:
        """Runs that really started. One that could not be launched at all is an environment problem, not a verdict."""
        return [r for r in runs if not (r.error and not r.runs)]

    def _relative_references(self, evaluations) -> dict:
        """Best completed wall time / peak memory per run spec (with a noise floor): the 1.0 mark for everyone."""
        best: dict = {}
        for ev in evaluations:
            for run in self._attempted(ev.runs):
                if not run.completed:
                    continue
                sec, mb = best.get(run.name, (None, None))
                if run.wall_seconds is not None and (sec is None or run.wall_seconds < sec):
                    sec = run.wall_seconds
                if run.peak_rss_mb is not None and (mb is None or run.peak_rss_mb < mb):
                    mb = run.peak_rss_mb
                best[run.name] = (sec, mb)
        best = {name: (None if sec is None else max(sec, MIN_RELATIVE_SECONDS),
                       None if mb is None else max(mb, MIN_RELATIVE_MB)) for name, (sec, mb) in best.items()}
        best[None] = (None, None)
        return best

    def _score(self, ev: Evaluation, refs: dict, require_tests: bool) -> None:
        if ev.empty:
            return
        try:
            static = ev.static.metrics if ev.static else {}
            runs_flag = ev.imports.runs if ev.imports else None

            # checks: acceptance tests + completion of the run specs (both written by the orchestrator, not the candidate)
            parts = []
            acceptance = ev.tests.acceptance if ev.tests else None
            if acceptance is not None and acceptance.pass_rate is not None:
                parts.append(acceptance.pass_rate)
            attempted = self._attempted(ev.runs)
            if attempted:
                parts.append(sum(1 for r in attempted if r.completed) / len(attempted))
            checks = sum(parts) / len(parts) if parts else None

            # candidate's own tests
            test_rate = ev.tests.candidate.pass_rate if ev.tests and ev.tests.candidate else None
            suite = ev.tests.candidate if ev.tests else None
            if test_rate is None and require_tests and suite and suite.ran and suite.no_tests:
                test_rate = 0.0
                ev.notes.append("candidate was required to write tests but wrote none: test_pass_rate = 0")

            # runtime / memory: mean over the run specs that produced a number
            runtime_vals, memory_vals = [], []
            for run in attempted:
                sec, mb = refs.get(run.name) or refs.get(None) or (None, None)
                m = run.metrics(reference_seconds=sec, reference_mb=mb)
                if m.get("runtime") is not None:
                    runtime_vals.append(m["runtime"])
                if m.get("memory") is not None:
                    memory_vals.append(m["memory"])

            metrics = {
                "correctness": score_correctness(compiles=static.get("compiles"), runs=runs_flag, checks=checks),
                "test_pass_rate": test_rate,
                "runtime": sum(runtime_vals) / len(runtime_vals) if runtime_vals else None,
                "memory": sum(memory_vals) / len(memory_vals) if memory_vals else None,
                "security": static.get("security"),
                "code_quality": static.get("code_quality"),
                "architecture": None,                      # no tool measures this; only a judge can
                "documentation": static.get("documentation"),
            }
            ev.score = compute_score(metrics, self.settings.weights_for(ev.task_type))
            if ev.static:
                skipped = ev.static.skipped_tools()
                if skipped:
                    ev.notes.append("static tools not available: " + ", ".join(skipped))
            if ev.stage_errors:
                ev.notes.append("stages that failed and were not measured: " + ", ".join(sorted(ev.stage_errors)))
        except Exception as exc:                           # scoring must never take the run down
            logger.error("Scoring %s failed: %s: %s", ev.label, type(exc).__name__, exc)
            ev.stage_errors["scoring"] = f"{type(exc).__name__}: {exc}"
            ev.score = failure_score(f"scoring failed: {exc}")


# --------------------------------------------------------------------------- #
# Self-test:  python -m evaluation.evaluator     (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from context.repository import Repository
    from core.logger import setup_logging
    from evaluation.scoring import rank_candidates as _rank

    setup_logging("ERROR")
    tmp = tempfile.TemporaryDirectory()
    settings = OrchestratorSettings()

    def make_repo(name, files):
        repo = Repository(Path(tmp.name) / name)
        if files:
            repo.write_files(files)
        return repo

    def make_evaluator(**kw):
        # tools forced "missing" so the result does not depend on what is installed on this machine
        return Evaluator(settings, static=StaticAnalyzer(settings, commands={}),
                         tests=TestRunner(settings, commands={"pytest": None}), **kw)

    CALC = ('"""Tiny calculator."""\n\n\ndef add(a, b):\n    """Add."""\n    return a + b\n\n\n'
            'def mul(a, b):\n    """Multiply."""\n    return a * b\n')
    GOOD_TESTS = ("import unittest\nfrom calc import add, mul\n\n\nclass T(unittest.TestCase):\n"
                  "    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n\n"
                  "    def test_mul(self):\n        self.assertEqual(mul(2, 3), 6)\n\n\n"
                  "if __name__ == '__main__':\n    unittest.main()\n")
    HALF_TESTS = GOOD_TESTS.replace("self.assertEqual(mul(2, 3), 6)", "self.assertEqual(mul(2, 3), 7)")
    README = "# Calc\n\n## Usage\n\n    python main.py\n"
    GOOD = {"calc.py": CALC, "test_calc.py": GOOD_TESTS, "README.md": README}
    ACCEPT = {"test_acceptance.py": ("import unittest\nfrom calc import add\n\n\nclass A(unittest.TestCase):\n"
                                     "    def test_a(self):\n        self.assertEqual(add(2, 2), 4)\n")}
    ev = make_evaluator()

    # 1. A healthy candidate: compiles, imports, tests pass, documented
    r1 = ev.evaluate(make_repo("good", GOOD), "backend_api", label="good", acceptance_tests=ACCEPT)
    m = r1.metrics
    assert m["correctness"] == 1.0 and m["test_pass_rate"] == 1.0, m
    assert r1.imports.runs is True and r1.tests.acceptance.pass_rate == 1.0
    assert m["documentation"] is not None and m["architecture"] is None and "architecture" in r1.score.skipped
    assert r1.final_score > 0.9 and not r1.stage_errors, r1.summary()
    print(f"PASS 1: healthy candidate -> {r1.summary()}")

    # 2. Syntax error: gated score, findings name the file and the import failure
    bad = make_repo("bad", {"calc.py": "def add(a, b)\n    return a + b\n", "test_calc.py": GOOD_TESTS})
    r2 = ev.evaluate(bad, "backend_api", label="bad")
    assert r2.metrics["correctness"] < 0.5 and r2.score.gate_factor < 1.0 and r2.final_score < r1.final_score
    text = r2.findings()
    assert "calc.py" in text and "do not compile" in text and "fail to import" in text, text
    print(f"PASS 2: broken candidate gated -> {r2.final_score:.3f} (x{r2.score.gate_factor:.2f}); findings name calc.py")

    # 3. Failing tests are reported by name and lower test_pass_rate
    r3 = ev.evaluate(make_repo("half", {"calc.py": CALC, "test_calc.py": HALF_TESTS, "README.md": README}),
                     "backend_api", label="half")
    assert r3.metrics["test_pass_rate"] == 0.5 and "test_mul" in r3.findings(), r3.findings()
    assert r3.final_score < r1.final_score
    print(f"PASS 3: failing test reported -> test_pass_rate {r3.metrics['test_pass_rate']}; findings mention test_mul")

    # 4. Empty / missing workspaces are failed candidates with a reason, not crashes
    r4 = ev.evaluate(make_repo("empty", None), "backend_api", label="empty")
    many_none = ev.evaluate_many({"ghost": None}, "backend_api")
    assert r4.empty and r4.final_score == 0.0 and "No files" in r4.findings()
    assert many_none["ghost"].final_score == 0.0 and many_none["ghost"].empty
    print("PASS 4: empty workspace and missing candidate -> score 0 with a reason")

    # 5. Acceptance tests drive `checks`: a candidate that fails them loses correctness
    wrong = make_repo("wrong", {"calc.py": CALC.replace("a + b", "a - b"), "test_calc.py": GOOD_TESTS.replace(
        "self.assertEqual(add(1, 2), 3)", "self.assertEqual(add(1, 2), -1)")})
    r5 = ev.evaluate(wrong, "backend_api", label="wrong", acceptance_tests=ACCEPT)
    assert r5.tests.candidate.pass_rate == 1.0 and r5.tests.acceptance.pass_rate == 0.0     # self-consistent but wrong
    assert r5.metrics["correctness"] < r1.metrics["correctness"] and "test_a" in r5.findings()
    print(f"PASS 5: self-consistent but wrong code caught by acceptance tests -> correctness {r5.metrics['correctness']:.2f}")

    # 6. Run specs and relative runtime/memory across candidates; a failing workload scores 0
    def prog(delay, out="42"):
        return {"main.py": f"import time\ntime.sleep({delay})\nprint('{out}')\n", "README.md": README}
    spec = RunSpec(command=["python", "main.py"], name="main", expected_stdout="42\n")
    runs = make_evaluator().evaluate_many(
        {"fast": make_repo("fast", prog(0.0)), "slow": make_repo("slow", prog(0.8)),
         "wrong": make_repo("wrongout", prog(0.0, "41"))}, "desktop_application", run_specs=[spec])
    assert runs["fast"].metrics["runtime"] == 1.0, runs["fast"].metrics
    assert 0.0 < runs["slow"].metrics["runtime"] < 0.5, runs["slow"].metrics          # 0.8s vs the 0.25s floor
    tiny = make_evaluator().evaluate_many({"a": make_repo("ta", prog(0.0)), "b": make_repo("tb", prog(0.05))},
                                          "desktop_application", run_specs=[spec])
    assert tiny["a"].metrics["runtime"] == tiny["b"].metrics["runtime"] == 1.0, "sub-floor differences are noise"
    assert runs["wrong"].metrics["runtime"] == 0.0 and "wrong output" in runs["wrong"].findings()
    assert [m for m, _ in Evaluator.rank(runs)][0] == "fast" and Evaluator.scores(runs)["fast"] > Evaluator.scores(runs)["wrong"]
    print(f"PASS 6: relative runtime fast={runs['fast'].metrics['runtime']:.2f} slow={runs['slow'].metrics['runtime']:.2f} "
          f"wrong-output=0.00; ranking {Evaluator.rank(runs)[0][0]} first")

    # 7. Absolute references; no references -> runtime stays unmeasured for a single evaluate()
    r7a = ev.evaluate(make_repo("fast2", prog(0.0)), "desktop_application", run_specs=[spec])
    r7b = ev.evaluate(make_repo("fast3", prog(0.0)), "desktop_application", run_specs=[spec], reference_seconds=30.0)
    assert r7a.metrics["runtime"] is None and "runtime" in r7a.score.skipped
    assert r7b.metrics["runtime"] == 1.0
    print("PASS 7: absolute reference scores runtime; without one it is reported as not measured")

    # 8. require_tests: no tests written -> 0 instead of "not measured"
    notests = {"calc.py": CALC, "README.md": README}
    a = make_evaluator().evaluate(make_repo("nt1", notests), "backend_api")
    b = make_evaluator(require_tests=True).evaluate(make_repo("nt2", notests), "backend_api")
    assert a.metrics["test_pass_rate"] is None and b.metrics["test_pass_rate"] == 0.0 and b.final_score < a.final_score
    assert any("wrote none" in n for n in b.notes)
    print("PASS 8: require_tests turns 'no tests' into 0.0 and says so")

    # 9. LLM judge blends only subjective metrics, on a copy, capped
    r9 = ev.apply_judge(r1, {"architecture": 0.2, "code_quality": 0.0, "correctness": 0.0, "security": 0.0})
    assert r9 is not r1 and r1.metrics["architecture"] is None                        # original untouched
    assert r9.metrics["correctness"] == r1.metrics["correctness"] and r9.metrics["security"] == r1.metrics["security"]
    assert r9.metrics["architecture"] == 0.2 and "architecture" not in r9.score.skipped
    assert ev.apply_judge(r1, {}) is r1 and ev.apply_judge(r4, {"architecture": 1.0}) is r4
    print(f"PASS 9: judge blended into a copy (architecture judge-only), objective metrics untouched -> {r9.final_score:.3f}")

    # 10. Failures inside a stage are contained and recorded
    class ExplodingStatic:
        def analyze(self, repo):
            raise RuntimeError("analyzer exploded")
    class ExplodingRuntime(RuntimeRunner):
        def import_check(self, repo, *a, **k):
            raise OSError("no fork")
    shaky = Evaluator(settings, static=ExplodingStatic(), tests=TestRunner(settings, commands={"pytest": None}),
                      runtime=ExplodingRuntime(settings))
    r10 = shaky.evaluate(make_repo("shaky", GOOD), "backend_api", label="shaky")
    assert set(r10.stage_errors) == {"static", "import_check"} and r10.metrics["test_pass_rate"] == 1.0
    assert r10.final_score > 0 and any("stages that failed" in n for n in r10.notes)
    print(f"PASS 10: stage failures contained -> {sorted(r10.stage_errors)} recorded, remaining stages still scored")

    # 11. Reports are JSON, saved atomically, and temp files are not left behind
    out = Path(tmp.name) / "reports" / "good.json"
    assert r1.save(out) and json.loads(out.read_text(encoding="utf-8"))["final_score"] == r1.final_score
    assert not [p for p in out.parent.iterdir() if p.suffix == ".tmp"]
    assert json.dumps(runs["fast"].to_dict(), default=str) and json.dumps(r4.to_dict())
    assert r1.save(Path(tmp.name) / "reports" / "good.json" / "nested" / "x.json") is False     # parent is a file: contained
    print("PASS 11: reports serialise to JSON, atomic save, unsavable path returns False instead of raising")

    # 12. Deterministic ranking across a mixed lite benchmark, usable by the scheduler/benchmark store
    mixed = ev.evaluate_many({"good": make_repo("g2", GOOD), "half": make_repo("h2", {
        "calc.py": CALC, "test_calc.py": HALF_TESTS, "README.md": README}), "broken": bad, "none": None}, "backend_api")
    order = [m for m, _ in Evaluator.rank(mixed)]
    assert order[0] == "good" and order[-1] in ("none", "broken") and set(Evaluator.scores(mixed)) == set(mixed)
    assert order == [m for m, _ in _rank({k: v.score for k, v in mixed.items()})]
    print(f"PASS 12: ranking {order}")

    tmp.cleanup()