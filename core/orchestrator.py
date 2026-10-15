"""
core/orchestrator.py

The end-to-end workflow: it owns the stage order and wires the scheduler, the agents, the evaluator and the
persistent state together. Nothing here talks to a provider; every model call goes through `registry.generate`.

    from core.orchestrator import Orchestrator, build_orchestrator

    orch = build_orchestrator(load_settings())                  # real wiring (main.py does this)
    result = orch.run("Build me a desktop app that monitors my downloads and categorizes them")
    result.status        # completed | needs_attention | paused | failed
    result.passed        # the review loop accepted the implementation
    result.workspace     # folder with the finished project (projects/<id>/implementation)
    orch.resume(result.project_id)       # continue a paused / failed / interrupted run, skipping finished stages

Stages (core.state_manager.STAGES; each is re-runnable, finished ones are skipped on resume):
  task_analysis      cheap model (or heuristics) -> task_spec.json
  workflow_decision  scheduler.plan(): direct single model, or orchestrated; reserves tokens for later stages
  lite_generation    lite benchmark spec (reduced task + acceptance tests)            [orchestrated, no history skip]
  benchmark          each candidate builds the lite task in its OWN workspace; objective evaluation; history recorded
  selection          scheduler.select_coder(): use / wait for a quota reset / alternative; picks an independent reviewer
  hld                architecture.md (fallback model on failure)
  coding             full implementation in batches; model failure -> next model, progress kept
  review             evaluate -> review -> revise loop (cap = max_review_iterations), rollback, escalation, optional judge

Behaviour worth knowing:
  * DIRECT mode (task too small to orchestrate): ONE model designs and codes, and the review loop uses objective
    evidence only (compile / import / tests / run results): no second model is spent.
  * Benchmark guard: model-written acceptance tests are never trusted blindly. If NO candidate passes any acceptance
    check, the tests are assumed wrong, the candidates are re-evaluated without them and the decision is logged.
    A candidate that failed only because the provider errored (quota, bad key) is excluded and NOT recorded, so an
    outage cannot ruin a model's history.
  * Quota: a "wait" decision sleeps when the reset is within `max_wait_seconds`; a longer wait (or no model at all)
    PAUSES the project (status paused) instead of burning retries; `resume()` continues later.
  * Review loop: every round is evaluated and recorded (state + reviews/round_NN.json). A revision that made things
    worse is rolled back (workspace snapshot). A blocking finding that survives two rounds is `stuck`: ONE escalation
    (revise the architecture and rebuild the affected files, or switch the coder). The program is only RUN by the
    evaluator when it is a plain terminating script; GUIs, servers, argparse/stdin programs are never run blindly.
  * Optional LLM judge at the end, only for orchestrated runs, only if an independent model can afford it.
  * Bookkeeping never crashes a run: logging, history recording and reports are wrapped.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agents.coder import CoderAgent, CoderError, LiteSpec, make_lite_spec
from agents.hld import HLDAgent, HLDError, HLDResult, parse_architecture
from agents.judge import JudgeAgent, JudgeError
from agents.reviewer import ReviewError, ReviewerAgent, ReviewResult, objective_findings
from config.settings import OrchestratorSettings, Settings
from context.indexer import RepoIndex
from context.repository import Repository
from core.benchmark_store import BenchmarkRecord
from core.logger import DecisionLogger, get_logger
from core.scheduler import MAX_WAIT_SECONDS, NoModelAvailable
from core.state_manager import (DONE_STATUSES, PENDING, PROJECT_PAUSED, ProjectState, StateManager)
from core.task_analyzer import TaskAnalysis, TaskAnalyzer
from evaluation.evaluator import Evaluation, Evaluator
from evaluation.runtime import RunSpec
from models import ModelError, ModelUnavailable
from models.base import FatalModelError, ModelRequest

logger = get_logger("core.orchestrator")

MAX_STAGE_ATTEMPTS = 6          # model switches / waits inside one stage before giving up
MAX_WAIT_ATTEMPTS = 3           # quota waits while planning or selecting the coder
MAX_CODING_ROUNDS = 2           # extra implement() calls for files that are still missing
MAX_ROLLBACKS = 2               # regressions tolerated before the review loop gives up
REGRESSION_SCORE_DROP = 0.05    # a score drop this large counts as a regression even if findings look equal
RUN_SPEC_TIMEOUT = 30.0
BLOCKING = ("blocker", "major")


# --------------------------------------------------------------------------- #
# Errors and results
# --------------------------------------------------------------------------- #

class OrchestratorError(Exception):
    """Base class for orchestration failures."""


class StageFailed(OrchestratorError):
    def __init__(self, stage: str, message: str, cause: Optional[BaseException] = None):
        super().__init__(f"{stage}: {message}")
        self.stage, self.cause = stage, cause


class RunPaused(OrchestratorError):
    """Not an error: nothing can run now, and waiting would take too long. The project resumes later."""

    def __init__(self, reason: str, wait_seconds: Optional[float] = None):
        super().__init__(reason)
        self.reason, self.wait_seconds = reason, wait_seconds


@dataclass
class RunResult:
    project_id: str
    status: str                                   # completed | needs_attention | paused | failed
    passed: bool = False
    message: str = ""
    mode: str = ""                                # direct | orchestrated
    workspace: str = ""
    coder_model: Optional[str] = None
    reviewer_model: Optional[str] = None
    hld_model: Optional[str] = None
    final_score: Optional[float] = None
    review_rounds: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    wait_seconds: Optional[float] = None
    warnings: list = field(default_factory=list)
    stages: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict:
        return dict(self.__dict__, ok=self.ok)

    def summary(self) -> str:
        lines = [f"Project {self.project_id}: {self.status.upper()}" + (f" - {self.message}" if self.message else "")]
        if self.mode:
            lines.append(f"  mode: {self.mode}; coder: {self.coder_model or '-'}; reviewer: {self.reviewer_model or '-'}; "
                         f"architect: {self.hld_model or '-'}")
        if self.final_score is not None:
            lines.append(f"  score {self.final_score:.3f} after {self.review_rounds} review round(s); "
                         f"{self.tokens} tokens, ${self.cost_usd:.4f}, {self.calls} model calls")
        if self.workspace:
            lines.append(f"  workspace: {self.workspace}")
        if self.status == "paused":
            lines.append("  resume with: --resume " + self.project_id)
        lines += [f"  warning: {w}" for w in self.warnings[:5]]
        return "\n".join(lines)


@dataclass
class _Run:
    """Everything one execution needs, rebuilt from disk on resume."""
    state: ProjectState
    force: Optional[bool] = None
    analysis: Optional[TaskAnalysis] = None
    plan: Any = None
    spec: Optional[LiteSpec] = None
    scores: dict = field(default_factory=dict)
    coder_model: Optional[str] = None
    reviewer_model: Optional[str] = None
    hld_model: Optional[str] = None
    design: Optional[HLDResult] = None
    repo: Optional[Repository] = None
    evaluation: Optional[Evaluation] = None
    review: Optional[ReviewResult] = None
    passed: bool = False
    warnings: list = field(default_factory=list)

    @property
    def orchestrated(self) -> bool:
        return bool(self.analysis and self.analysis.needs_orchestration)


# --------------------------------------------------------------------------- #
# Deciding what may be run
# --------------------------------------------------------------------------- #

_RISKY_WORDS = ("tkinter", "turtle", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "pygame", "kivy", "flask", "fastapi",
                "uvicorn", "django", "aiohttp", "tornado", "streamlit", "gradio", "serve_forever", "mainloop",
                "argparse", "click", "typer", "watchdog", "schedule", "curses", "readline", "getpass", "http.server",
                "socketserver", "asyncio.run")
_RISKY = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _RISKY_WORDS) + r")\b"
                    r"|\binput\s*\(|\bwhile\s+(?:True|1)\b|sys\.argv|\.run_forever\(")


def derive_run_specs(design: Any, analysis: Any, repo: Any, *, timeout: float = RUN_SPEC_TIMEOUT) -> list:
    """
    RunSpecs for the evaluator, or [] when the program should NOT be run automatically. A program is only run if it is
    a plain Python script that should terminate on its own: no UI, no server, no stdin, no required arguments,
    no endless loop in the entry file or anything it imports. When in doubt, do not run it.
    """
    try:
        if analysis is not None and getattr(analysis, "needs_ui", False):
            return []
        raw = (getattr(design, "run_command", "") or "").strip()
        entry = (getattr(design, "entry_point", "") or "").strip()
        command = shlex.split(raw) if raw else (["python", entry] if entry else [])
        if not command or command[0] not in ("python", "python3", "py") or "-m" in command[1:2]:
            return []
        command[0] = "python"
        script = next((c for c in command[1:] if c.endswith(".py")), "")
        if not script or not repo.exists(script):
            return []
        index = RepoIndex()
        index.update(repo)
        for path in [script] + index.dependencies(script, transitive=True):
            if _RISKY.search(repo.read_file(path, errors="replace")):
                return []
        return [RunSpec(command=command, name="main", timeout=timeout)]
    except Exception as exc:                              # not being able to decide means: do not run it
        logger.debug("derive_run_specs failed: %s", exc)
        return []


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_") or "model"


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #

class Orchestrator:
    def __init__(self, settings: Settings, registry: Any, resources: Any, store: Any, state_manager: StateManager,
                 scheduler: Any, decisions: Optional[DecisionLogger] = None, *,
                 evaluator: Optional[Evaluator] = None, hld: Optional[HLDAgent] = None,
                 reviewer: Optional[ReviewerAgent] = None, judge: Optional[JudgeAgent] = None,
                 coder_factory: Optional[Callable[[str], CoderAgent]] = None, wait_for_quota: bool = True,
                 max_wait_seconds: float = MAX_WAIT_SECONDS, use_judge: bool = True,
                 on_event: Optional[Callable[[str, str], None]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        """
        wait_for_quota:   sleep through quota resets shorter than max_wait_seconds (False = always pause instead)
        on_event:         on_event(project_id, message) for progress output (the CLI prints these)
        sleep:            injectable for tests
        evaluator/hld/reviewer/judge/coder_factory: injectable; defaults are built on registry.generate
        """
        self.settings = settings
        self.cfg: OrchestratorSettings = settings.orchestrator
        self.registry, self.resources, self.store = registry, resources, store
        self.state, self.scheduler = state_manager, scheduler
        self.decisions = decisions if decisions is not None else getattr(scheduler, "decisions", None)
        self.hld = hld or HLDAgent(self._generate)
        self.reviewer = reviewer or ReviewerAgent(self._generate)
        self.judge = judge or JudgeAgent(self._generate)
        self.evaluator_override = evaluator
        self._coder_factory = coder_factory
        self.wait_for_quota, self.max_wait_seconds, self.use_judge = wait_for_quota, max_wait_seconds, use_judge
        self.on_event, self._sleep = on_event, sleep
        self._dead: set = set()                 # models that failed with a non-retryable error in this run

    # ---- plumbing ------------------------------------------------------------------ #

    def _generate(self, model_id: str, request: ModelRequest):
        return self.registry.generate(model_id, request)

    def _coder(self, model_id: str) -> CoderAgent:
        if self._coder_factory:
            return self._coder_factory(model_id)
        try:
            limit = int(self.registry.config(model_id).max_output_tokens)
        except Exception:
            limit = 4096
        return CoderAgent(self._generate, max_output_tokens=limit)

    def _evaluator(self, require_tests: bool) -> Evaluator:
        return self.evaluator_override or Evaluator(self.cfg, require_tests=require_tests)

    def _emit(self, ctx: _Run, message: str) -> None:
        logger.info("[%s] %s", ctx.state.project_id, message)
        if self.on_event:
            try:
                self.on_event(ctx.state.project_id, message)
            except Exception:
                logger.debug("on_event callback failed", exc_info=True)

    def _decide(self, category: str, decision: str, reasoning: str = "", context: Optional[dict] = None) -> None:
        if self.decisions is not None:
            try:
                self.decisions.log(category, decision, reasoning, context or {})
            except Exception:                              # logging must never crash a run
                logger.debug("decision logging failed", exc_info=True)

    def _usage(self, ctx: _Run, stage: str, tokens: int = 0, cost: float = 0.0, calls: int = 0) -> None:
        usage = ctx.state.data.get("usage") or {"tokens": 0, "cost_usd": 0.0, "calls": 0, "by_stage": {}}
        usage["tokens"] += int(tokens or 0)
        usage["cost_usd"] = round(usage["cost_usd"] + float(cost or 0.0), 6)
        usage["calls"] += int(calls or 0)
        usage["by_stage"][stage] = usage["by_stage"].get(stage, 0) + int(tokens or 0)
        self.state.set_data(ctx.state, "usage", usage)

    def _persist_models(self, ctx: _Run) -> None:
        self.state.set_data(ctx.state, "models", {"coder": ctx.coder_model, "reviewer": ctx.reviewer_model,
                                                  "hld": ctx.hld_model})

    def _warn(self, ctx: _Run, message: str) -> None:
        if message not in ctx.warnings:
            ctx.warnings.append(message)
            self.state.set_data(ctx.state, "warnings", list(ctx.warnings))
        logger.warning("[%s] %s", ctx.state.project_id, message)

    def _impl_dir(self, state: ProjectState) -> Path:
        return self.state.project_dir(state.project_id) / "implementation"

    @contextmanager
    def _stage(self, ctx: _Run, name: str) -> Iterator[dict]:
        """Like StateManager.stage(), except that a pause leaves the stage pending instead of failed."""
        state = ctx.state
        self.state.start_stage(state, name)
        out: dict = {}
        try:
            yield out
        except RunPaused:
            rec = state.stages[name]
            rec.status, rec.error = PENDING, None
            self.state.set_status(state, PROJECT_PAUSED)
            raise
        except BaseException as exc:
            self.state.fail_stage(state, name, f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.state.complete_stage(state, name, out or None)

    # ---- waiting, pausing, switching models ------------------------------------------- #

    def _can_wait(self, seconds: Optional[float]) -> bool:
        return bool(self.wait_for_quota and seconds is not None and seconds <= self.max_wait_seconds)

    def _wait(self, ctx: _Run, seconds: float, reason: str) -> None:
        seconds = max(0.0, float(seconds)) + 1.0
        self._decide("quota_wait", f"waiting {seconds:.0f}s", reason, {"seconds": seconds})
        self._emit(ctx, f"waiting {seconds:.0f}s: {reason}")
        self._sleep(seconds)

    def _make_plan(self, ctx: _Run):
        for _ in range(MAX_WAIT_ATTEMPTS + 1):
            try:
                return self.scheduler.plan(ctx.analysis)
            except NoModelAvailable as exc:
                if not self._can_wait(exc.wait_seconds):
                    raise RunPaused(str(exc), exc.wait_seconds)
                self._wait(ctx, exc.wait_seconds, str(exc))
        raise RunPaused("no model became available after waiting")

    def _next_model(self, stage: str, ctx: _Run, failed: list, exc: BaseException) -> str:
        """The model that takes over `stage` after `failed` ones broke; may wait for a failed one to come back."""
        if isinstance(exc, FatalModelError) and failed:
            self._dead.add(failed[-1])          # bad key / unknown model: waiting will not fix it
        alt = self.scheduler.fallback_model(stage, ctx.analysis, failed, ctx.plan)
        if alt:
            self._emit(ctx, f"{stage}: {failed[-1]} failed ({type(exc).__name__}); {alt} takes over")
            return alt
        tokens = self.scheduler.stage_tokens(stage, ctx.analysis)
        waits = []
        for model in dict.fromkeys(failed):
            if model in self._dead:
                continue
            wait = self.scheduler.estimated_wait(model, tokens, ignore_reservation=stage)
            if wait:                                      # > 0: usable later; 0: usable now, so waiting is pointless
                waits.append((wait, model))
        if waits:
            wait, model = min(waits)
            if not self._can_wait(wait):
                raise RunPaused(f"{stage}: every model failed; {model} may be usable in ~{wait:.0f}s", wait)
            self._wait(ctx, wait, f"{stage}: every model failed; {model} should be back")
            failed[:] = [m for m in failed if m != model]
            return model
        raise StageFailed(stage, f"no model can take over after {', '.join(failed)} failed: {exc}", exc)

    def _call_with_fallback(self, stage: str, ctx: _Run, model: str, call: Callable[[str], Any],
                            catch: tuple = (ModelError, ModelUnavailable, HLDError)) -> tuple:
        failed: list = []
        for _ in range(MAX_STAGE_ATTEMPTS):
            try:
                return call(model), model
            except catch as exc:
                failed.append(model)
                self._decide("model_failure", f"{model} failed at {stage}", f"{type(exc).__name__}: {exc}",
                             {"stage": stage, "failed": list(failed)})
                model = self._next_model(stage, ctx, failed, exc)
        raise StageFailed(stage, "too many failed attempts")

    # ---- public API ------------------------------------------------------------------- #

    def run(self, requirement: str, *, project_id: Optional[str] = None,
            force_orchestration: Optional[bool] = None) -> RunResult:
        """Start a new project and run it to the end (or to a pause / failure). Never raises for model problems."""
        if not (requirement or "").strip():
            raise ValueError("the requirement is empty")
        state = self.state.create_project(requirement.strip(), project_id)
        ctx = _Run(state, force=force_orchestration, repo=Repository(self._impl_dir(state)))
        return self._execute(ctx)

    def resume(self, project_id: str) -> RunResult:
        """Continue a project: finished stages are skipped, interrupted ones re-run (agents resume their own work)."""
        state = self.state.load_project(project_id)
        ctx = _Run(state)
        self._restore(ctx)
        return self._execute(ctx)

    # ---- execution ---------------------------------------------------------------------- #

    def _execute(self, ctx: _Run) -> RunResult:
        state = ctx.state
        self._dead = set()
        if self.decisions is not None:
            self.decisions.project_id = state.project_id
        handlers = {"task_analysis": self._stage_analysis, "workflow_decision": self._stage_workflow,
                    "lite_generation": self._stage_lite, "benchmark": self._stage_benchmark,
                    "selection": self._stage_selection, "hld": self._stage_hld, "coding": self._stage_coding,
                    "review": self._stage_review}
        try:
            self.state.set_status(state, "running")
            for stage in list(state.stage_order):
                if state.stages[stage].status in DONE_STATUSES:
                    continue
                if stage not in ("task_analysis", "workflow_decision"):
                    self._ensure_plan(ctx)
                handlers[stage](ctx)
            return self._result(ctx, "completed" if ctx.passed else "needs_attention")
        except RunPaused as pause:
            self._emit(ctx, f"paused: {pause.reason}")
            self._decide("run_paused", "project paused", pause.reason, {"wait_seconds": pause.wait_seconds})
            return self._result(ctx, "paused", pause.reason, pause.wait_seconds)
        except StageFailed as exc:
            self._decide("run_failed", f"stage {exc.stage} failed", str(exc), {})
            return self._result(ctx, "failed", str(exc))
        except (CoderError, HLDError, ReviewError, JudgeError, ModelError, ModelUnavailable, OrchestratorError) as exc:
            self._decide("run_failed", "run failed", f"{type(exc).__name__}: {exc}", {})
            return self._result(ctx, "failed", f"{type(exc).__name__}: {exc}")
        finally:
            try:
                self.scheduler.release_all()
            except Exception:
                logger.debug("release_all failed", exc_info=True)

    def _result(self, ctx: _Run, status: str, message: str = "", wait: Optional[float] = None) -> RunResult:
        state = ctx.state
        usage = state.data.get("usage") or {}
        if not message:
            message = ("the review accepted the implementation" if status == "completed"
                       else "the review loop ended with blocking findings left (see reviews/ and final_report.md)")
        return RunResult(
            project_id=state.project_id, status=status, passed=ctx.passed and status == "completed", message=message,
            mode="orchestrated" if ctx.orchestrated else ("direct" if ctx.analysis else ""),
            workspace=str(self._impl_dir(state)), coder_model=ctx.coder_model, reviewer_model=ctx.reviewer_model,
            hld_model=ctx.hld_model, final_score=ctx.evaluation.final_score if ctx.evaluation else None,
            review_rounds=state.review_iteration, tokens=int(usage.get("tokens", 0)),
            cost_usd=float(usage.get("cost_usd", 0.0)), calls=int(usage.get("calls", 0)), wait_seconds=wait,
            warnings=list(ctx.warnings), stages={n: r.status for n, r in state.stages.items()})

    def _restore(self, ctx: _Run) -> None:
        """Rebuild the in-memory context of a project from its artifacts and saved data (resume)."""
        state, sm = ctx.state, self.state
        raw = sm.read_json_artifact(state, "task_spec.json")
        if raw:
            ctx.analysis = TaskAnalysis.from_dict(raw)
        lite = sm.read_json_artifact(state, "benchmark/lite_spec.json")
        if lite:
            ctx.spec = LiteSpec.from_dict(lite)
        results = sm.read_json_artifact(state, "benchmark/results.json")
        if results:
            ctx.scores = {m: float(s) for m, s in (results.get("scores") or {}).items()}
        models = state.data.get("models") or {}
        ctx.coder_model, ctx.reviewer_model, ctx.hld_model = models.get("coder"), models.get("reviewer"), models.get("hld")
        ctx.warnings = list(state.data.get("warnings") or [])
        text = sm.read_artifact(state, "architecture.md")
        if text:
            ctx.design = parse_architecture(text, ctx.hld_model or "")
        ctx.repo = Repository(self._impl_dir(state))

    def _ensure_plan(self, ctx: _Run) -> None:
        """After a resume there is no plan yet: make a fresh one (quotas changed) and keep reservations honest."""
        if ctx.plan is not None:
            return
        ctx.plan = self._make_plan(ctx)
        stages = ctx.state.stages
        for stage in ("hld", "coding", "review"):
            if stages[stage].status in DONE_STATUSES:
                self.scheduler.stage_done(stage)
        for stage, model in (("coding", ctx.coder_model), ("review", ctx.reviewer_model if ctx.orchestrated else None)):
            if model and stages[stage].status not in DONE_STATUSES:
                self.resources.release(stage)
                self.resources.reserve(model, self.scheduler.stage_tokens(stage, ctx.analysis), stage)

    # ---- stage 1: task analysis ---------------------------------------------------------- #

    def _stage_analysis(self, ctx: _Run) -> None:
        with self._stage(ctx, "task_analysis") as out:
            model = self.scheduler.pick_analyzer()
            generate = (lambda request, m=model: self.registry.generate(m, request)) if model else None
            n = min(self.cfg.max_benchmark_candidates, max(1, len(self.registry.ids())))
            analysis = TaskAnalyzer(self.cfg).analyze(ctx.state.requirement, generate=generate,
                                                      force_orchestration=ctx.force, n_candidates=n)
            ctx.analysis = analysis
            self.state.write_artifact(ctx.state, "task_spec.json", analysis.to_dict())
            self._usage(ctx, "task_analysis", analysis.analysis_tokens, 0.0, 1 if analysis.analysis_tokens else 0)
            out.update(task_type=analysis.task_type, complexity=analysis.complexity, orchestrate=analysis.needs_orchestration,
                       analyzer_model=model, source=analysis.source)
            self._decide("task_analysis", analysis.short(), analysis.orchestration_reason,
                         {"analyzer_model": model, "confidence": analysis.confidence})
            self._emit(ctx, f"analysis: {analysis.short()}")

    # ---- stage 2: workflow decision -------------------------------------------------------- #

    def _stage_workflow(self, ctx: _Run) -> None:
        with self._stage(ctx, "workflow_decision") as out:
            ctx.plan = self._make_plan(ctx)
            plan = ctx.plan
            out.update(mode="orchestrated" if plan.use_orchestration else "direct", skip_benchmark=plan.skip_benchmark,
                       benchmark_candidates=list(plan.benchmark_candidates),
                       assignments={s: a.model_id for s, a in plan.assignments.items()}, warnings=list(plan.warnings))
            for warning in plan.warnings:
                self._warn(ctx, warning)
            self._emit(ctx, f"workflow: {out['mode']}; benchmark candidates: {plan.benchmark_candidates or 'none'}")

    # ---- stages 3-4: lite generation and benchmark ------------------------------------------- #

    def _benchmark_skip_reason(self, ctx: _Run) -> str:
        if not ctx.plan.use_orchestration:
            return "direct workflow: no benchmark"
        if ctx.plan.skip_benchmark:
            return "benchmark skipped: " + (ctx.plan.reasons[-1] if ctx.plan.reasons else "history decides the coder")
        return ""

    def _stage_lite(self, ctx: _Run) -> None:
        reason = self._benchmark_skip_reason(ctx)
        if reason:
            self.state.skip_stage(ctx.state, "lite_generation", reason)
            return
        with self._stage(ctx, "lite_generation") as out:
            model = self.scheduler.pick_analyzer()
            spec = make_lite_spec(ctx.state.requirement, ctx.analysis, generate=self._generate if model else None,
                                  model_id=model, token_budget=self.cfg.lite_task_token_budget)
            ctx.spec = spec
            self.state.write_artifact(ctx.state, "benchmark/lite_spec.json", spec.to_dict())
            self._usage(ctx, "lite_generation", spec.tokens_used, 0.0, 1 if spec.tokens_used else 0)
            out.update(source=spec.source, acceptance_tests=sorted(spec.acceptance_tests), notes=spec.notes)
            self._emit(ctx, f"lite spec ready ({spec.source}, {len(spec.acceptance_tests)} acceptance test file(s))")

    def _run_candidate(self, ctx: _Run, model: str, tokens: int) -> tuple:
        """(workspace or None, meta or None). meta None = excluded from the benchmark (not evaluated, not recorded)."""
        state = ctx.state
        safe = _safe_name(model)
        repo_dir = self.state.project_dir(state.project_id) / "benchmark" / safe
        meta_name = f"benchmark/{safe}.meta.json"
        meta = self.state.read_json_artifact(state, meta_name)
        if meta and meta.get("model") == model and repo_dir.is_dir() and any(repo_dir.iterdir()):
            self._emit(ctx, f"benchmark: reusing the saved result of {model}")
            return Repository(repo_dir), meta
        if not self.resources.can_use(model, tokens):
            self._warn(ctx, f"{model} could not afford the benchmark any more and was left out")
            return None, None
        shutil.rmtree(repo_dir, ignore_errors=True)
        repo = Repository(repo_dir)
        result = self._coder(model).implement_lite(ctx.spec, repo, model)
        self._usage(ctx, "benchmark", result.total_tokens, result.cost_usd, result.calls)
        if result.error is not None and not result.written:
            self._warn(ctx, f"{model} failed during the benchmark ({type(result.error).__name__}); "
                            "it is excluded and not recorded against its history")
            return None, None
        meta = {"model": model, "tokens": result.total_tokens, "cost_usd": result.cost_usd,
                "latency": result.latency_seconds, "files": len(result.written),
                "error": repr(result.error) if result.error else "", "warnings": result.warnings[:5]}
        self.state.write_artifact(state, meta_name, meta)
        self._emit(ctx, f"benchmark: {model} wrote {len(result.written)} file(s), {result.total_tokens} tokens")
        return (repo if result.written else None), meta

    def _stage_benchmark(self, ctx: _Run) -> None:
        reason = self._benchmark_skip_reason(ctx)
        if reason:
            self.state.skip_stage(ctx.state, "benchmark", reason)
            return
        with self._stage(ctx, "benchmark") as out:
            analysis = ctx.analysis
            tokens = self.scheduler.stage_tokens("lite_benchmark", analysis)
            candidates: dict = {}
            metas: dict = {}
            for model in ctx.plan.benchmark_candidates:
                repo, meta = self._run_candidate(ctx, model, tokens)
                if meta is not None:
                    candidates[model], metas[model] = repo, meta
            if not candidates:
                self._warn(ctx, "no benchmark candidate produced a result: the coder is chosen from priors and history")
                ctx.scores = {}
                self.state.write_artifact(ctx.state, "benchmark/results.json", {"scores": {}, "models": {}})
                out.update(candidates=[], scores={})
                return

            acceptance = ctx.spec.acceptance_tests or None
            evaluator = self._evaluator(require_tests=True)
            evals = evaluator.evaluate_many(candidates, analysis.task_type, acceptance_tests=acceptance)
            discarded = False
            if acceptance:
                ran = [e.tests.acceptance.pass_rate for e in evals.values()
                       if not e.empty and e.tests and e.tests.acceptance and e.tests.acceptance.pass_rate is not None]
                if ran and all(rate == 0 for rate in ran):
                    discarded = True
                    self._decide("benchmark_guard", "acceptance tests discarded",
                                 "no candidate passed a single acceptance check: the model-written tests are "
                                 "probably wrong, so candidates are scored without them",
                                 {"candidates": sorted(candidates)})
                    self._warn(ctx, "all candidates failed every acceptance test: the tests were assumed wrong and discarded")
                    evals = evaluator.evaluate_many(candidates, analysis.task_type)
            ctx.scores = Evaluator.scores(evals)
            ranking = Evaluator.rank(evals)
            winner = ranking[0][0] if ranking and ranking[0][1] > 0 else None

            session = None
            try:
                records = [BenchmarkRecord(
                    model_id=m, task_type=analysis.task_type, final_score=e.final_score,
                    language=analysis.primary_language, complexity=analysis.complexity,
                    metrics={k: v for k, v in e.metrics.items() if v is not None},
                    tokens_used=int(metas[m].get("tokens", 0)), cost_usd=float(metas[m].get("cost_usd", 0.0)),
                    latency_seconds=float(metas[m].get("latency", 0.0)), success=(not e.empty and e.final_score > 0),
                    project_id=ctx.state.project_id, notes=e.summary()[:400]) for m, e in evals.items()]
                session = self.store.record_session(records, winner_model_id=winner, project_id=ctx.state.project_id)
            except Exception as exc:                          # history is bookkeeping
                self._warn(ctx, f"benchmark history could not be recorded: {type(exc).__name__}: {exc}")
            self.state.write_artifact(ctx.state, "benchmark/results.json", {
                "scores": ctx.scores, "session_id": session, "winner": winner, "acceptance_tests_discarded": discarded,
                "models": {m: {"score": e.final_score, "metrics": {k: v for k, v in e.metrics.items() if v is not None},
                               "summary": e.summary(), "tokens": metas[m].get("tokens", 0),
                               "cost_usd": metas[m].get("cost_usd", 0.0)} for m, e in evals.items()}})
            self.state.set_data(ctx.state, "benchmark_session", session)
            out.update(scores=ctx.scores, winner=winner, session=session, acceptance_discarded=discarded)
            self._decide("benchmark_result", f"benchmark ranking: {', '.join(f'{m}={s:.3f}' for m, s in ranking)}",
                         "objective evaluation of the lite projects", {"ranking": ranking, "discarded": discarded})
            self._emit(ctx, "benchmark ranking: " + ", ".join(f"{m}={s:.3f}" for m, s in ranking))

    # ---- stage 5: selection -------------------------------------------------------------------- #

    def _decide_coder(self, ctx: _Run):
        plan = ctx.plan
        for attempt in range(MAX_WAIT_ATTEMPTS + 1):
            if attempt == 0 and plan.skip_benchmark and plan.coder_decision is not None:
                decision = plan.coder_decision                  # decided at planning time (history / priors)
            else:
                decision = self.scheduler.select_coder(plan, ctx.analysis, ctx.scores)
            if decision.action == "use":
                return decision
            if decision.action == "fail":
                raise StageFailed("selection", decision.reason)
            if self._can_wait(decision.wait_seconds):
                self._wait(ctx, decision.wait_seconds, decision.reason)
                continue
            if decision.alternative_id:                         # too long to wait: take the best model usable now
                score = next((e["score"] for e in decision.ranking if e["model_id"] == decision.alternative_id), 0.0)
                plan.baseline_model = None
                alt = self.scheduler.select_coder(plan, ctx.analysis, {decision.alternative_id: score})
                if alt.action == "use":
                    return alt
            raise RunPaused(decision.reason, decision.wait_seconds)
        raise RunPaused("the preferred coding model did not become available after waiting")

    def _stage_selection(self, ctx: _Run) -> None:
        if not ctx.plan.use_orchestration:
            self.state.skip_stage(ctx.state, "selection", "direct workflow: the planned model codes")
            ctx.coder_model = ctx.plan.model_for("coding")
            ctx.reviewer_model, ctx.hld_model = None, ctx.coder_model
            self._persist_models(ctx)
            return
        with self._stage(ctx, "selection") as out:
            decision = self._decide_coder(ctx)
            ctx.coder_model, ctx.reviewer_model = decision.model_id, decision.reviewer_id
            ctx.hld_model = ctx.plan.model_for("hld")
            self._persist_models(ctx)
            session = ctx.state.data.get("benchmark_session")
            if session:
                try:
                    self.store.mark_winner(session, decision.model_id)
                except Exception:
                    logger.debug("mark_winner failed", exc_info=True)
            out.update(coder=decision.model_id, reviewer=decision.reviewer_id, reason=decision.reason)
            self._emit(ctx, f"coder: {decision.model_id}; reviewer: {decision.reviewer_id or 'none affordable'}")

    # ---- stage 6: architecture ------------------------------------------------------------------- #

    def _stage_hld(self, ctx: _Run) -> None:
        with self._stage(ctx, "hld") as out:
            model = ctx.hld_model or (ctx.coder_model if not ctx.orchestrated else None) \
                or self.scheduler.fallback_model("hld", ctx.analysis, [], ctx.plan)
            if not model:
                raise StageFailed("hld", "no model can afford the architecture stage")
            design, used = self._call_with_fallback(
                "hld", ctx, model, lambda m: self.hld.design(ctx.state.requirement, ctx.analysis, m))
            ctx.design, ctx.hld_model = design, used
            self._persist_models(ctx)
            self.state.write_artifact(ctx.state, "architecture.md", design.markdown)
            self._usage(ctx, "hld", design.total_tokens, design.cost_usd, design.calls)
            for warning in design.warnings[:3]:
                self._warn(ctx, f"architecture: {warning}")
            self.scheduler.stage_done("hld")
            out.update(model=used, files=len(design.files), entry_point=design.entry_point, tokens=design.total_tokens)
            self._emit(ctx, f"architecture by {used}: {len(design.files)} planned file(s)")

    # ---- stage 7: coding ---------------------------------------------------------------------------- #

    def _stage_coding(self, ctx: _Run) -> None:
        with self._stage(ctx, "coding") as out:
            repo = ctx.repo = Repository(self._impl_dir(ctx.state))
            model, failed, extra_rounds = ctx.coder_model, [], 0
            for _ in range(MAX_STAGE_ATTEMPTS):
                result = self._coder(model).implement(
                    ctx.design, ctx.state.requirement, ctx.analysis, repo, model, resume=True,
                    on_batch=lambda batch, res: self.state.set_data(
                        ctx.state, "coding_progress", {"files_written": len(res.written), "calls": res.calls}))
                self._usage(ctx, "coding", result.total_tokens, result.cost_usd, result.calls)
                for warning in result.warnings[:3]:
                    self._warn(ctx, f"coding: {warning}")
                if result.error is not None:
                    failed.append(model)
                    self._decide("model_failure", f"{model} failed while coding",
                                 f"{type(result.error).__name__}: {result.error}",
                                 {"written": sorted(result.written), "remaining": result.remaining()})
                    model = self._next_model("coding", ctx, failed, result.error)
                    ctx.coder_model = model
                    self._persist_models(ctx)
                    continue
                if result.complete:
                    break
                extra_rounds += 1
                if extra_rounds >= MAX_CODING_ROUNDS:
                    self._warn(ctx, "coding finished with files still missing or cut off: " + ", ".join(result.remaining()[:8]))
                    break
            else:
                raise StageFailed("coding", "too many failed attempts")
            self.state.checkpoint(ctx.state, "after_coding")
            self.scheduler.stage_done("coding")
            files = repo.list_files()
            out.update(model=ctx.coder_model, files=len(files), missing=result.remaining())
            self._emit(ctx, f"coding by {ctx.coder_model}: {len(files)} file(s) in the workspace")

    # ---- stage 8: review loop -------------------------------------------------------------------------- #

    def _objective_review(self, ctx: _Run, evaluation: Evaluation, previous: Optional[ReviewResult], iteration: int,
                          note: str = "") -> ReviewResult:
        """A review made of facts only (direct mode, or no reviewer affordable): no model call."""
        findings = sorted(objective_findings(evaluation), key=lambda f: (f.rank, f.title))
        before = {f.title: f for f in (previous.findings if previous else [])}
        for finding in findings:
            if finding.title in before:
                finding.rounds, finding.status = before[finding.title].rounds + 1, "persisting"
        passed = not any(f.severity in BLOCKING for f in findings)
        review = ReviewResult(passed=passed, findings=findings, summary="Objective checks only (no model review).",
                              verdict="pass" if passed else "fail", coder_model=ctx.coder_model or "", iteration=iteration)
        review.stuck = any(f.rounds >= 2 and f.severity in BLOCKING for f in findings)
        if note:
            review.warnings.append(note)
        return review

    def _model_review(self, ctx: _Run, evaluation: Evaluation, previous: Optional[ReviewResult], iteration: int,
                      diff_text: str, changed: list) -> ReviewResult:
        model = ctx.reviewer_model or self.scheduler.fallback_model("review", ctx.analysis, [], ctx.plan)
        if not model:
            self._warn(ctx, "no model can afford the review: using objective checks only")
            return self._objective_review(ctx, evaluation, previous, iteration, "no reviewer affordable")

        def call(m: str) -> ReviewResult:
            return self.reviewer.review(ctx.state.requirement, ctx.design, ctx.repo, m, coder_model=ctx.coder_model or "",
                                        analysis=ctx.analysis, evaluation=evaluation, previous=previous,
                                        iteration=iteration, diff_text=diff_text, changed_files=changed)
        try:
            review, used = self._call_with_fallback("review", ctx, model, call, catch=(ModelError, ModelUnavailable))
        except ReviewError as exc:
            raise StageFailed("review", str(exc), exc)
        if used != ctx.reviewer_model:
            ctx.reviewer_model = used
            self._persist_models(ctx)
        self._usage(ctx, "review", review.total_tokens, review.cost_usd, review.calls)
        return review

    @staticmethod
    def _regressed(review: ReviewResult, evaluation: Evaluation, previous: ReviewResult, previous_eval: Evaluation) -> str:
        """Why the latest revision made things worse ('' = it did not)."""
        if review.severity_score > previous.severity_score and evaluation.final_score <= previous_eval.final_score:
            return (f"severity {previous.severity_score:.1f} -> {review.severity_score:.1f} and score "
                    f"{previous_eval.final_score:.3f} -> {evaluation.final_score:.3f}")
        if evaluation.final_score < previous_eval.final_score - REGRESSION_SCORE_DROP and not review.improved_over(previous):
            return f"score dropped {previous_eval.final_score:.3f} -> {evaluation.final_score:.3f}"
        return ""

    def _revise(self, ctx: _Run, findings: str) -> Any:
        model, failed = ctx.coder_model, []
        for _ in range(MAX_STAGE_ATTEMPTS):
            try:
                result = self._coder(model).revise(ctx.repo, findings, model, requirement=ctx.state.requirement,
                                                   design=ctx.design, analysis=ctx.analysis)
            except CoderError as exc:
                raise StageFailed("review", f"revision impossible: {exc}", exc)
            self._usage(ctx, "revision", result.total_tokens, result.cost_usd, result.calls)
            if result.error is None:
                return result
            failed.append(model)
            self._decide("model_failure", f"{model} failed while revising", f"{type(result.error).__name__}: {result.error}", {})
            model = self._next_model("coding", ctx, failed, result.error)
            ctx.coder_model = model
            self._persist_models(ctx)
        raise StageFailed("review", "too many failed revision attempts")

    def _escalate(self, ctx: _Run, review: ReviewResult) -> str:
        """
        A blocking finding survived two rounds. Returns 'architecture' (design revised, affected files rebuilt),
        'coder' (coder switched; a normal revision follows) or '' (nothing else to try).
        """
        feedback = review.architecture_feedback()
        hld_model = ctx.hld_model or ctx.reviewer_model or ctx.coder_model
        if feedback and hld_model:
            try:
                revised = self.hld.revise(ctx.design, feedback, hld_model, requirement=ctx.state.requirement)
            except (HLDError, ModelError, ModelUnavailable) as exc:
                self._decide("escalation", "architecture revision failed", f"{type(exc).__name__}: {exc}", {})
                revised = None
            if revised is not None:
                self._usage(ctx, "escalation", revised.total_tokens, revised.cost_usd, revised.calls)
                planned = {f.path for f in revised.files}
                redo = sorted({f.file for f in review.blocking if f.file in planned})
                if not redo:
                    redo = [p for p in CoderAgent.files_mentioned(ctx.repo, review.to_revision_prompt()) if p in planned]
                redo = redo or ([revised.entry_point] if revised.entry_point in planned else [])
                ctx.design = revised
                self.state.write_artifact(ctx.state, "architecture.md", revised.markdown)
                self._decide("escalation", "architecture revised after a stuck review",
                             f"rebuilding {redo or 'nothing'} with {ctx.coder_model}", {"redo": redo})
                self._emit(ctx, f"stuck: architecture revised; rebuilding {len(redo)} file(s)")
                if redo:
                    result = self._coder(ctx.coder_model).implement(revised, ctx.state.requirement, ctx.analysis, ctx.repo,
                                                                    ctx.coder_model, resume=True, redo=redo)
                    self._usage(ctx, "escalation", result.total_tokens, result.cost_usd, result.calls)
                    if result.error is None:
                        return "architecture"
        alt = self.scheduler.fallback_model("coding", ctx.analysis, [ctx.coder_model], ctx.plan)
        if alt:
            self._decide("escalation", f"coder switched {ctx.coder_model} -> {alt}", "the same blocking finding survived "
                         "two revisions", {})
            self._emit(ctx, f"stuck: {alt} takes over as coder")
            ctx.coder_model = alt
            self._persist_models(ctx)
            return "coder"
        return ""

    def _review_loop(self, ctx: _Run) -> None:
        state, sm = ctx.state, self.state
        repo = ctx.repo = Repository(self._impl_dir(state))
        evaluator = self._evaluator(require_tests=False)
        run_specs = derive_run_specs(ctx.design, ctx.analysis, repo, timeout=min(RUN_SPEC_TIMEOUT, self.cfg.sandbox_timeout_seconds))
        if not run_specs:
            self._decide("run_policy", "program is not run automatically",
                         "not a plain terminating script (UI, server, stdin/argv, endless loop or unknown run command)", {})
        use_model_review = ctx.orchestrated
        previous = previous_eval = snapshot = None
        changed: list = []
        diff_text = note = ""
        escalated, rollbacks = False, 0

        last = sm.read_json_artifact(state, f"reviews/round_{state.review_iteration:02d}.json") if state.review_iteration else None
        if last and last.get("review"):
            previous = ReviewResult.from_dict(last["review"])

        for iteration in range(state.review_iteration + 1, self.cfg.max_review_iterations + 1):
            evaluation = evaluator.evaluate(repo, ctx.analysis.task_type, label=f"round-{iteration}", run_specs=run_specs)
            review = (self._model_review(ctx, evaluation, previous, iteration, diff_text, changed) if use_model_review
                      else self._objective_review(ctx, evaluation, previous, iteration))
            sm.record_review(state, review.passed, review.for_state(), ctx.reviewer_model, ctx.coder_model)
            sm.write_artifact(state, f"reviews/round_{iteration:02d}.json", {
                "review": review.to_dict(), "evaluation": evaluation.summary(), "score": evaluation.final_score})
            self._emit(ctx, f"round {iteration}: {review.summary_line()} | {evaluation.summary()}")

            if snapshot is not None and previous is not None and previous_eval is not None:
                why = self._regressed(review, evaluation, previous, previous_eval)
                if why:
                    rollbacks += 1
                    repo.restore(snapshot)
                    self._decide("rollback", "revision rolled back", why, {"round": iteration, "rollbacks": rollbacks})
                    self._emit(ctx, f"revision made things worse ({why}): rolled back")
                    review, evaluation = previous, previous_eval
                    note = ("\n\nNOTE: the previous attempt to fix this made the code worse and was rolled back. "
                            "Make a different, smaller and more careful change.")
                    snapshot, changed, diff_text = None, [], ""
                    if rollbacks >= MAX_ROLLBACKS:
                        ctx.review, ctx.evaluation = review, evaluation
                        self._warn(ctx, "revisions kept making things worse: stopped at the best version")
                        break
                else:
                    note = ""
            ctx.review, ctx.evaluation = review, evaluation
            if review.passed:
                ctx.passed = True
                break
            if iteration >= self.cfg.max_review_iterations:
                break

            snapshot = repo.snapshot("before-revision")
            action = ""
            if review.stuck and not escalated and ctx.orchestrated:
                escalated = True
                action = self._escalate(ctx, review)
            if action != "architecture":
                result = self._revise(ctx, review.to_revision_prompt() + note)
                changed = result.changed
                if not changed:
                    self._warn(ctx, "the coder changed nothing in response to the findings: stopped")
                    break
            else:
                changed = []
            try:
                diff_text = repo.diff(snapshot).unified
            except Exception:
                diff_text = ""
            previous, previous_eval = review, evaluation
            self.state.checkpoint(state, f"after_round_{iteration}")

    def _pick_judge(self, ctx: _Run) -> Optional[str]:
        tokens = self.judge.estimate_tokens()
        for exclude in ((ctx.coder_model, ctx.reviewer_model), (ctx.coder_model,)):
            pick = self.scheduler.pick_model("review", ctx.analysis, exclude=[m for m in exclude if m], tokens=tokens)
            if pick is not None:
                return pick.model_id
        return None

    def _final_judge(self, ctx: _Run) -> None:
        if not (self.use_judge and ctx.orchestrated) or ctx.evaluation is None or ctx.evaluation.empty:
            return
        model = self._pick_judge(ctx)
        if model is None:
            self._decide("judge", "judge skipped", "no independent model can afford a judging call", {})
            return
        try:
            result = self.judge.judge(ctx.state.requirement, ctx.design, ctx.repo, model, evaluation=ctx.evaluation,
                                      coder_model=ctx.coder_model or "", reviewer_model=ctx.reviewer_model or "")
        except (ModelError, ModelUnavailable, JudgeError) as exc:
            self._decide("judge", "judge failed", f"{type(exc).__name__}: {exc}; objective score kept", {"model": model})
            return
        self._usage(ctx, "judge", result.total_tokens, result.cost_usd, result.calls)
        if result.usable:
            before = ctx.evaluation.final_score
            ctx.evaluation = self._evaluator(False).apply_judge(ctx.evaluation, result.scores)
            self._decide("judge", f"{model} judged: {before:.3f} -> {ctx.evaluation.final_score:.3f}",
                         result.summary_line(), {"scores": result.scores})
        else:
            self._decide("judge", "judge gave no usable scores", result.summary_line(), {"model": model})

    def _stage_review(self, ctx: _Run) -> None:
        with self._stage(ctx, "review") as out:
            self._review_loop(ctx)
            self._final_judge(ctx)
            self.scheduler.stage_done("review")
            self._write_report(ctx)
            out.update(passed=ctx.passed, rounds=ctx.state.review_iteration,
                       final_score=ctx.evaluation.final_score if ctx.evaluation else None)
            self.state.checkpoint(ctx.state, "final")

    # ---- reporting ---------------------------------------------------------------------------------------- #

    def _write_report(self, ctx: _Run) -> None:
        """final_report.md + evaluation.json. Reporting problems never fail the run."""
        try:
            state, ev, review = ctx.state, ctx.evaluation, ctx.review
            usage = state.data.get("usage") or {}
            lines = [f"# Result: {state.project_id}", "", f"**Requirement:** {state.requirement}", "",
                     f"- Mode: {'orchestrated' if ctx.orchestrated else 'direct (single model)'}",
                     f"- Outcome: {'accepted by the review' if ctx.passed else 'NOT accepted: blocking findings remain'}",
                     f"- Coder: {ctx.coder_model}; reviewer: {ctx.reviewer_model or '-'}; architect: {ctx.hld_model or '-'}"]
            if ev is not None:
                lines.append(f"- Final score: {ev.final_score:.3f} ({ev.summary()})")
            lines += [f"- Usage: {usage.get('tokens', 0)} tokens, ${usage.get('cost_usd', 0.0):.4f}, "
                      f"{usage.get('calls', 0)} model calls", "", "## Review rounds", ""]
            for item in state.review_history:
                counts: dict = {}
                for f in item.get("findings", []):
                    counts[f["severity"]] = counts.get(f["severity"], 0) + 1
                lines.append(f"- Round {item['iteration']}: {'PASS' if item['passed'] else 'FAIL'} "
                             f"({', '.join(f'{n} {s}' for s, n in counts.items()) or 'no findings'})")
            if review is not None and not ctx.passed:
                lines += ["", "## Open findings", ""]
                lines += [f"- [{f.severity}] {f.location or 'general'}: {f.title}" for f in review.findings[:12]]
            if ctx.warnings:
                lines += ["", "## Warnings", ""] + [f"- {w}" for w in ctx.warnings]
            self.state.write_artifact(state, "final_report.md", "\n".join(lines) + "\n")
            if ev is not None:
                self.state.write_artifact(state, "evaluation.json", ev.to_dict())
        except Exception as exc:
            logger.warning("Could not write the final report: %s: %s", type(exc).__name__, exc)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

def build_orchestrator(settings: Optional[Settings] = None, *, projects_dir: Optional[Path] = None,
                       benchmark_db: Optional[Path] = None, resource_state: Optional[Path] = None,
                       decisions_log: Optional[Path] = None, **options: Any) -> Orchestrator:
    """Create every component from the settings and connect them (usage tracking included). main.py calls this."""
    from config.settings import load_settings
    from core.benchmark_store import BenchmarkStore
    from core.resource_manager import ResourceManager
    from core.scheduler import Scheduler
    from models import ModelRegistry

    settings = settings or load_settings()
    resources = ResourceManager(settings.enabled_models(), settings.orchestrator, state_path=resource_state)
    registry = ModelRegistry(settings, resources)
    store = BenchmarkStore(benchmark_db, settings=settings.orchestrator)
    decisions = DecisionLogger(log_path=decisions_log)
    scheduler = Scheduler(registry, resources, store, settings.orchestrator, decisions)
    return Orchestrator(settings, registry, resources, store, StateManager(projects_dir), scheduler, decisions, **options)


# --------------------------------------------------------------------------- #
# Self-test: real registry / resources / scheduler / store / state, scripted fake models
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace as NS

    from agents.hld import HEADINGS, REQUIRED
    from config.settings import ModelConfig
    from core.benchmark_store import BenchmarkStore
    from core.logger import setup_logging
    from core.resource_manager import ResourceManager
    from core.scheduler import Scheduler
    from models import ModelRegistry
    from models.base import FatalModelError, ModelAdapter, RawResult

    setup_logging("ERROR")
    tmp = tempfile.TemporaryDirectory()
    TMP = Path(tmp.name)
    counter = [0]

    GOOD = "import re\n\n\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
    BUGGY = 'def slugify(text):\n    return text.lower().replace(" ", "")\n'
    BROKEN = "def slugify(text)\n    return 1\n"
    ACCEPT = ("import unittest\nfrom textutil import slugify\n\n\nclass T(unittest.TestCase):\n"
              "    def test_basic(self):\n        self.assertEqual(slugify('Hello World'), 'hello-world')\n"
              "    def test_punct(self):\n        self.assertEqual(slugify('  A--B!! '), 'a-b')\n")
    PROJECT_TEST = ("import unittest\nfrom textutil import slugify\n\n\nclass T(unittest.TestCase):\n"
                    "    def test_basic(self):\n        self.assertEqual(slugify('Hello World'), 'hello-world')\n")
    PLAN = {"entry_point": "main.py", "run_command": "python main.py", "dependencies": [],
            "files": [{"path": "textutil.py", "purpose": "slugify", "depends_on": []},
                      {"path": "main.py", "purpose": "demo entry point", "depends_on": ["textutil.py"]},
                      {"path": "tests/test_textutil.py", "purpose": "tests", "depends_on": ["textutil.py"]},
                      {"path": "README.md", "purpose": "usage"}]}
    BODIES = {"overview": "A tiny text utility library.", "components": "textutil: slugify.",
              "file_layout": "```json\n" + json.dumps(PLAN) + "\n```",
              "interfaces": "```python\ndef slugify(text: str) -> str: ...\n```", "decisions": "Regex based.",
              "testing": "unittest.", "risks": "Unicode."}
    HLD_DOC = "# Textutil\n\n" + "\n\n".join(f"## {HEADINGS[k]}\n\n{BODIES[k]}" for k in REQUIRED)
    REQ = "Write a small python library textutil with a slugify function that turns a title into a url slug"

    class Brain:
        """Scripted behaviour shared by all fake models; records (model, purpose) of every call."""
        def __init__(self):
            self.lite = {"m_strong": BUGGY, "m_mid": GOOD, "m_weak": BROKEN}
            self.buggy_first = {"m_mid"}            # models whose FIRST full coding pass has the bug
            self.fatal: set = set()                 # {(model, purpose)} or {("*", purpose)}
            self.calls: list = []
            self.first_done: set = set()
            self.has_bug = False                    # does the workspace currently hold the buggy textutil.py?

        def count(self, purpose, model=None):
            return sum(1 for m, p in self.calls if p == purpose and (model is None or m == model))

        def reply(self, model, req) -> str:
            p = req.purpose
            if p == "task_analysis":
                return "not json"
            if p == "lite_benchmark":
                if "benchmarks" in (req.system or ""):
                    return json.dumps({"title": "slug", "task": "Write textutil.slugify(text) -> url slug. Stdlib only.",
                                       "interface": [{"path": "textutil.py", "signature": "def slugify(text: str) -> str",
                                                      "behaviour": "lowercase, hyphen separated"}],
                                       "acceptance_tests": {"test_acceptance.py": ACCEPT}})
                return f'<file path="textutil.py">\n{self.lite[model]}</file>'
            if p == "hld":
                return HLD_DOC
            if p == "coding":
                if "FINDINGS TO FIX" in req.prompt:
                    self.has_bug = False
                    return f'<file path="textutil.py">\n{GOOD}</file>'
                code = GOOD
                if model in self.buggy_first and model not in self.first_done:
                    code, _ = BUGGY, self.first_done.add(model)
                self.has_bug = code is BUGGY
                files = {"textutil.py": code, "main.py": "from textutil import slugify\n\nprint(slugify('Hello World'))\n",
                         "tests/test_textutil.py": PROJECT_TEST, "README.md": "# Textutil\n\nRun `python main.py`.\n"}
                block = req.prompt.split("WRITE THESE FILES NOW\n", 1)[-1].split("\n\n", 1)[0]
                paths = re.findall(r"^- ([\w./\-]+\.\w+):", block, re.M) or list(files)
                return "\n".join(f'<file path="{x}">\n{files.get(x, "VALUE = 1" + chr(10))}</file>' for x in paths)
            if p == "review":
                bad = self.has_bug
                find = [{"severity": "blocker", "category": "correctness", "file": "textutil.py", "line": 2,
                         "title": "slugify drops word separators", "detail": "returns helloworld",
                         "suggestion": "use hyphens"}] if bad else []
                return json.dumps({"verdict": "fail" if bad else "pass", "summary": "bug" if bad else "ok",
                                   "requirements": [{"requirement": "slugify", "status": "unmet" if bad else "met"}],
                                   "findings": find})
            if p == "judge":
                return json.dumps({"architecture": {"score": 7, "reason": "ok"}, "code_quality": {"score": 7, "reason": "ok"},
                                   "documentation": {"score": 6, "reason": "ok"}, "summary": "fine"})
            return "{}"

    class FakeAdapter(ModelAdapter):
        def __init__(self, config, settings, brain):
            super().__init__(config, settings)
            self.brain, self._sleep = brain, (lambda s: None)

        def _call(self, request):
            self.brain.calls.append((self.model_id, request.purpose))
            if (self.model_id, request.purpose) in self.brain.fatal or ("*", request.purpose) in self.brain.fatal:
                raise FatalModelError("scripted failure", self.model_id)
            return RawResult(text=self.brain.reply(self.model_id, request), input_tokens=900, output_tokens=500)

    class Clock:
        def __init__(self):
            self.t = 1_000_000.0

        def __call__(self):
            return self.t

    def make_env(**options):
        counter[0] += 1
        cfg = OrchestratorSettings()
        models = [ModelConfig(id=i, provider="fake", model_name=i, capabilities={"coding": c, "hld": c, "review": c},
                              cost_per_1k_input=0.001, cost_per_1k_output=0.002, token_budget_per_window=400_000)
                  for i, c in (("m_strong", 9.0), ("m_mid", 8.0), ("m_weak", 6.5))]
        clock = options.pop("clock", None) or Clock()
        rm = ResourceManager(models, cfg, persist=False, clock=clock)
        brain = Brain()
        registry = ModelRegistry(Settings(orchestrator=cfg, models=[]), rm)
        for m in models:
            registry.add_adapter(FakeAdapter(m, cfg, brain))
        store = BenchmarkStore(":memory:", settings=cfg, clock=clock)
        decisions = DecisionLogger(project_id="t", log_path=TMP / f"d{counter[0]}.jsonl")
        sm = StateManager(TMP / f"projects{counter[0]}")
        scheduler = Scheduler(registry, rm, store, cfg, decisions)
        events: list = []
        orch = Orchestrator(Settings(orchestrator=cfg, models=models), registry, rm, store, sm, scheduler, decisions,
                            on_event=lambda pid, msg: events.append(msg), **options)
        return NS(orch=orch, brain=brain, rm=rm, store=store, sm=sm, clock=clock, events=events, decisions=decisions,
                  models=[m.id for m in models])

    def project_file(env, res, name):
        return (Path(res.workspace) / name).read_text()

    # 1. Unit: which programs may be run automatically; regression detection
    def repo_of(files):
        counter[0] += 1
        repo = Repository(TMP / f"r{counter[0]}")
        repo.write_files(files)
        return repo
    D = NS(run_command="python main.py", entry_point="main.py")
    A = NS(needs_ui=False)
    assert len(derive_run_specs(D, A, repo_of({"main.py": "print(1)\n"}))) == 1
    assert derive_run_specs(D, NS(needs_ui=True), repo_of({"main.py": "print(1)\n"})) == []
    assert derive_run_specs(D, A, repo_of({"main.py": "import argparse\n"})) == []
    assert derive_run_specs(D, A, repo_of({"main.py": "x = input()\n"})) == []
    assert derive_run_specs(D, A, repo_of({"main.py": "import helper\n", "helper.py": "import tkinter\n"})) == []
    assert derive_run_specs(D, A, repo_of({"other.py": "print(1)\n"})) == []
    assert derive_run_specs(NS(run_command="npm start", entry_point=""), A, repo_of({"main.py": "x=1\n"})) == []
    assert derive_run_specs(NS(run_command="python -m app", entry_point=""), A, repo_of({"main.py": "x=1\n"})) == []
    prev_r, cur_r = NS(severity_score=3.0), NS(severity_score=9.0, improved_over=lambda p: False)
    prev_e, cur_e = NS(final_score=0.8), NS(final_score=0.6)
    assert Orchestrator._regressed(cur_r, cur_e, prev_r, prev_e)
    assert not Orchestrator._regressed(NS(severity_score=1.0, improved_over=lambda p: True), NS(final_score=0.9), prev_r, prev_e)
    assert Orchestrator._regressed(NS(severity_score=3.0, improved_over=lambda p: False), NS(final_score=0.7), prev_r, prev_e)
    print("PASS 1: run-policy and regression rules")

    # 2. Direct mode: one model, objective review only, no reviewer / judge / benchmark calls
    env = make_env()
    res = env.orch.run(REQ, force_orchestration=False)
    assert res.status == "completed" and res.passed and res.mode == "direct", res.summary()
    assert env.brain.count("review") == 0 and env.brain.count("judge") == 0 and env.brain.count("lite_benchmark") == 0
    assert res.reviewer_model is None and res.coder_model == res.hld_model
    assert res.stages["benchmark"] == "skipped" and res.stages["selection"] == "skipped" and res.stages["review"] == "completed"
    assert "def slugify" in project_file(env, res, "textutil.py") and (Path(res.workspace).parent / "final_report.md").exists()
    assert res.tokens > 0 and res.calls >= 2 and res.final_score is not None
    print(f"PASS 2: direct mode -> {res.coder_model} did everything, score {res.final_score:.2f}, {res.calls} calls")

    # 3. Orchestrated: benchmark ranks the candidates, best coder is picked, reviewer is independent,
    #    the buggy first pass is caught and fixed in a revision round, judge adjusts the score, history is recorded
    env = make_env()
    res = env.orch.run(REQ, force_orchestration=True)
    assert res.status == "completed" and res.passed and res.mode == "orchestrated", res.summary()
    results = env.sm.read_json_artifact(env.sm.load_project(res.project_id), "benchmark/results.json")
    scores = results["scores"]
    assert results["winner"] == "m_mid" and scores["m_mid"] > scores["m_strong"] > scores.get("m_weak", -1) - 1e-9 or \
        scores["m_mid"] == max(scores.values()), scores
    assert res.coder_model == "m_mid", (res.coder_model, scores)
    assert res.reviewer_model and res.reviewer_model != res.coder_model
    assert res.review_rounds == 2 and env.brain.count("review") == 2, (res.review_rounds, env.events[-8:], res.warnings)
    assert "re.sub" in project_file(env, res, "textutil.py")
    assert env.brain.count("judge") == 1 and env.brain.count("lite_benchmark", "m_weak") >= 1
    assert env.store.count() >= 2, env.store.count()
    state = env.sm.load_project(res.project_id)
    assert [h["passed"] for h in state.review_history] == [False, True]
    assert all(s == "completed" for s in res.stages.values()), res.stages
    assert env.rm.status("m_mid").reserved_tokens == 0                    # every reservation released
    print(f"PASS 3: orchestrated -> coder {res.coder_model}, reviewer {res.reviewer_model}, scores "
          f"{ {m: round(s, 2) for m, s in scores.items()} }, {res.review_rounds} review rounds, score {res.final_score:.2f}")

    # 4. A failing stage stops the run; resume skips finished stages and completes
    env = make_env()
    env.brain.fatal = {("*", "hld")}
    res = env.orch.run(REQ, force_orchestration=True)
    assert res.status == "failed" and "hld" in res.message, res.summary()
    assert res.stages["benchmark"] == "completed" and res.stages["hld"] == "failed"
    lite_calls = env.brain.count("lite_benchmark")
    env.brain.fatal = set()
    for m in env.models:                                                   # failed models cooled down: let time pass
        env.rm.set_cooldown(m, 0)
    res2 = env.orch.resume(res.project_id)
    assert res2.status == "completed" and res2.project_id == res.project_id, res2.summary()
    assert env.brain.count("lite_benchmark") == lite_calls, "benchmark was re-run on resume"
    assert res2.coder_model == res.coder_model or res.coder_model is None
    print(f"PASS 4: hld failure -> failed; resume completed without re-running the benchmark ({lite_calls} lite calls)")

    # 5. The chosen coder dies mid-coding: another model takes over, progress is kept
    env = make_env()
    env.brain.fatal = {("m_mid", "coding")}
    res = env.orch.run(REQ, force_orchestration=True)
    assert res.status == "completed" and res.coder_model != "m_mid", res.summary()
    assert any("failed while coding" in d for d in
               [json.loads(l)["decision"] for l in env.decisions.log_path.read_text().splitlines()])
    print(f"PASS 5: coder m_mid failed -> {res.coder_model} took over and finished")

    # 6. Nothing usable for a long time: the project pauses (no burnt retries); resume works once quota is back
    clock = Clock()
    env = make_env(clock=clock)
    for m in env.models:
        env.rm.set_cooldown(m, 5000)
    res = env.orch.run(REQ, force_orchestration=True)
    assert res.status == "paused" and env.brain.calls == [] and res.wait_seconds, res.summary()
    assert env.sm.load_project(res.project_id).status == PROJECT_PAUSED
    clock.t += 6000
    res2 = env.orch.resume(res.project_id)
    assert res2.status == "completed", res2.summary()
    print(f"PASS 6: all models cooling down -> paused (wait ~{res.wait_seconds:.0f}s); resumed and completed")

    # 7. A short quota wait is slept through, then the run continues
    clock = Clock()
    slept: list = []
    env = make_env(clock=clock, sleep=lambda s: (slept.append(s), setattr(clock, "t", clock.t + s))[0])
    for m in env.models:
        env.rm.set_cooldown(m, 30)
    res = env.orch.run(REQ, force_orchestration=True)
    assert res.status == "completed" and slept and 25 <= slept[0] <= 60, (res.summary(), slept)
    print(f"PASS 7: short cooldown -> waited {slept[0]:.0f}s then completed")

    print("\nAll orchestrator self-tests passed.")