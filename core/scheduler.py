"""
core/scheduler.py

Decides WHICH model does WHICH stage, and whether benchmarking is needed at all.

It combines: the task analysis (complexity, token plan), the model registry, the resource manager
(quota, health, reservations) and the benchmark store (history). It never calls a model itself.

Typical use (done by the orchestrator):

    scheduler = Scheduler(registry, resources, store, settings.orchestrator, decisions)

    analyzer_model = scheduler.pick_analyzer()               # cheapest adequate model for stage 1
    plan = scheduler.plan(analysis)                          # who does what, benchmark or not
    ...run the lite benchmark on plan.benchmark_candidates, score the results...
    decision = scheduler.select_coder(plan, analysis, {"gpt": 0.85, "gemini": 0.7})
    ...run each stage with plan.model_for("hld"), decision.model_id, plan.model_for("review")...
    scheduler.stage_done("hld")                              # frees that stage's reserved tokens
    backup = scheduler.fallback_model("coding", analysis, failed=["gpt"], plan=plan)

How a model is scored for a stage (all parts 0..1):
    quality      capability prior (config, 0-10 -> 0-1), blended with real benchmark history for coding
                 stages (history weight grows with the number of past runs, up to 70%)
    reliability  from the resource manager (success rate, with a prior for new models)
    cost         1 - (estimated stage cost / most expensive candidate's cost); free models score 1
    score = w_quality*quality + w_reliability*reliability + w_cost*cost
    where w_quality rises with task complexity (0.5 -> 0.9): hard tasks buy quality, easy tasks save money.

Token reservations: the plan reserves tokens for review/HLD/coding up front, so a cheaper earlier stage
(like the benchmark) can't use up the tokens a later, more valuable stage needs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

from config.settings import OrchestratorSettings
from core.benchmark_store import BenchmarkStore
from core.logger import DecisionLogger, get_logger
from core.resource_manager import ResourceManager
from core.task_analyzer import TaskAnalysis

if TYPE_CHECKING:                      # type hints only
    from models import ModelRegistry

logger = get_logger("core.scheduler")

# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #

TOKEN_ESTIMATE_HEADROOM = 1.2      # token estimates are rough: plan for 20% more than the estimate
MAX_WAIT_SECONDS = 600.0           # never wait longer than this for a preferred model
WAIT_SCORE_GAP = 0.08              # wait for the best coder only if the best AVAILABLE one is this much worse
TIE_MARGIN = 0.02                  # scores closer than this are a tie: prefer cheaper, then more reliable
ANALYSIS_TOKEN_ESTIMATE = 1500     # tokens for the task-analysis call (prompt + reply)
MAX_FALLBACKS = 2                  # backup models listed per assignment

RESERVABLE_STAGES = ("hld", "coding", "review")
_STAGE_CAPABILITY = {"lite_benchmark": "coding", "coding": "coding", "hld": "hld", "review": "review"}
_INPUT_FRACTION = {"lite_benchmark": 0.4, "coding": 0.4, "hld": 0.4, "review": 0.7}
_HISTORY_STAGES = ("lite_benchmark", "coding")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# Errors and result types
# --------------------------------------------------------------------------- #

class NoModelAvailable(Exception):
    """No model can run right now. `wait_seconds` is the soonest estimated time one might (None = unknown)."""

    def __init__(self, message: str, wait_seconds: Optional[float] = None):
        super().__init__(message)
        self.wait_seconds = wait_seconds


@dataclass
class ModelScore:
    model_id: str
    score: float
    quality: float
    reliability: float
    est_cost_usd: float
    eligible: bool
    reason: str                       # "ok" or why the model can't take the stage right now


@dataclass
class StageAssignment:
    stage: str
    model_id: Optional[str]           # None = nobody can take it right now
    estimated_tokens: int
    estimated_cost_usd: float
    reason: str
    fallbacks: list = field(default_factory=list)
    reserved: bool = False


@dataclass
class CoderDecision:
    action: str                       # "use" | "wait" | "fail"
    model_id: Optional[str]
    score: Optional[float]
    wait_seconds: Optional[float]     # for "wait": how long until model_id is usable
    alternative_id: Optional[str]     # for "wait": the best model usable right now (take it to skip the wait)
    reviewer_id: Optional[str]        # for "use": the reviewer chosen to complement the coder
    ranking: list                     # every candidate considered, best first (dicts)
    reason: str


@dataclass
class Plan:
    use_orchestration: bool
    skip_benchmark: bool = False
    benchmark_candidates: list = field(default_factory=list)
    baseline_model: Optional[str] = None       # proven historical leader that unproven models must beat
    baseline_score: Optional[float] = None
    assignments: dict = field(default_factory=dict)      # stage -> StageAssignment
    coder_decision: Optional[CoderDecision] = None       # set when the coder is chosen without a benchmark
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def model_for(self, stage: str) -> Optional[str]:
        assignment = self.assignments.get(stage)
        return assignment.model_id if assignment else None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #

class Scheduler:
    def __init__(
        self,
        registry: "ModelRegistry",
        resources: ResourceManager,
        store: BenchmarkStore,
        settings: OrchestratorSettings,
        decisions: Optional[DecisionLogger] = None,
    ):
        self.registry = registry
        self.resources = resources
        self.store = store
        self.settings = settings
        self.decisions = decisions

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log(self, category: str, decision: str, reasoning: str = "", context: Optional[dict] = None) -> None:
        if self.decisions:
            self.decisions.log(category, decision, reasoning, context)
        else:
            logger.info("[%s] %s | %s", category, decision, reasoning)

    # ------------------------------------------------------------------ #
    # Token estimates, eligibility, waiting
    # ------------------------------------------------------------------ #

    def stage_tokens(self, stage: str, analysis: TaskAnalysis) -> int:
        """Planned tokens for a stage (the analyzer's estimate plus headroom)."""
        plan = analysis.token_plan
        raw = {
            "lite_benchmark": plan["benchmark_per_candidate"],
            "hld": plan["hld"],
            "coding": plan["coding"],
            "review": plan["review"],
        }[stage]
        return int(raw * TOKEN_ESTIMATE_HEADROOM)

    def _eligibility(self, model_id: str, tokens: int, ignore_reservation: Optional[str] = None) -> tuple:
        status = self.resources.status(model_id)
        if not status.available:
            return False, status.reason
        if not self.resources.can_afford(model_id, tokens, ignore_reservation):
            usable = self.resources.usable_tokens(model_id, ignore_reservation)
            return False, f"needs ~{tokens} tokens but only {usable} are usable"
        return True, "ok"

    def estimated_wait(self, model_id: str, tokens: int, ignore_reservation: Optional[str] = None) -> Optional[float]:
        """
        Seconds until the model can probably take `tokens`: 0.0 = usable now, None = waiting won't help.
        Quota waits are estimates: they use the time until the oldest usage leaves the window.
        """
        status = self.resources.status(model_id)
        if status.reason == "disabled":
            return None

        waits = [0.0]
        if status.cooldown_remaining > 0:
            waits.append(status.cooldown_remaining)
        elif not status.available and status.reason.startswith("requests/min"):
            waits.append(60.0)

        if not self.resources.can_afford(model_id, tokens, ignore_reservation):
            budget = self.registry.config(model_id).token_budget_per_window
            ceiling = int(budget * (1 - self.settings.quota_safety_margin))
            if tokens > ceiling - self.resources.reserved_tokens(model_id, ignore_reservation):
                return None                       # would not fit even in an empty window
            reset = self.resources.seconds_until_reset(model_id)
            if reset is None:
                return None
            waits.append(reset)
        return max(waits)

    def _soonest_wait(self, tokens: int = 1) -> Optional[float]:
        waits = [w for w in (self.estimated_wait(m, tokens) for m in self.registry.ids()) if w is not None]
        return min(waits) if waits else None

    def _explain_unavailable(self) -> str:
        return "; ".join(f"{m}: {self.resources.status(m).reason}" for m in self.registry.ids())

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #

    def _history(self, analysis: TaskAnalysis) -> dict:
        try:
            performance = self.store.get_performance(
                analysis.task_type, analysis.primary_language, complexity=analysis.complexity)
            return {p.model_id: p for p in performance}
        except Exception:                          # bad history must never stop scheduling
            logger.exception("Could not read benchmark history; using capability priors only")
            return {}

    def _blend_history(self, prior: float, perf) -> float:
        if perf is None or perf.runs <= 0:
            return prior
        weight = 0.7 * min(1.0, perf.runs / max(1, self.settings.min_history_to_skip_benchmark))
        return (1 - weight) * prior + weight * perf.weighted_avg_score

    def rank_models(
        self,
        stage: str,
        analysis: TaskAnalysis,
        tokens: Optional[int] = None,
        exclude: Iterable = (),
        ignore_reservation: Optional[str] = None,
    ) -> list:
        """All models scored for a stage, eligible ones first, best first."""
        tokens = self.stage_tokens(stage, analysis) if tokens is None else tokens
        excluded = set(exclude)
        w_quality = _clamp(0.5 + 0.04 * analysis.complexity, 0.5, 0.9)
        w_reliability = 0.1
        w_cost = max(0.0, 1.0 - w_quality - w_reliability)

        history = self._history(analysis) if stage in _HISTORY_STAGES else {}
        capability = _STAGE_CAPABILITY[stage]
        input_tokens = int(tokens * _INPUT_FRACTION[stage])

        rows = []
        for model_id in self.registry.ids():
            prior = _clamp(self.registry.config(model_id).capabilities.get(capability, 5.0) / 10.0, 0.0, 1.0)
            quality = self._blend_history(prior, history.get(model_id))
            cost = self.resources.estimate_cost(model_id, input_tokens, tokens - input_tokens)
            eligible, reason = self._eligibility(model_id, tokens, ignore_reservation)
            rows.append((model_id, quality, self.resources.reliability(model_id), cost, eligible, reason))

        # Cost is normalised against ALL models (excluded ones too), so excluding the most expensive
        # model (e.g. after it failed) doesn't make the next-most-expensive one look cheap or costly.
        max_cost = max((r[3] for r in rows), default=0.0)
        scored = []
        for model_id, quality, reliability, cost, eligible, reason in rows:
            if model_id in excluded:
                continue
            cost_efficiency = 1.0 - cost / max_cost if max_cost > 0 else 1.0
            score = w_quality * quality + w_reliability * reliability + w_cost * cost_efficiency
            scored.append(ModelScore(model_id, round(score, 4), round(quality, 4), round(reliability, 4),
                                     round(cost, 6), eligible, reason))
        scored.sort(key=lambda s: (not s.eligible, -s.score, s.model_id))
        return scored

    def pick_model(self, stage, analysis, exclude: Iterable = (), tokens=None, ignore_reservation=None):
        """Best eligible ModelScore for a stage, or None."""
        for candidate in self.rank_models(stage, analysis, tokens, exclude, ignore_reservation):
            if candidate.eligible:
                return candidate
        return None

    def pick_analyzer(self, tokens: int = ANALYSIS_TOKEN_ESTIMATE) -> Optional[str]:
        """Cheapest usable model for the task-analysis call (free local models win); ties go to the stronger one."""
        options = []
        for model_id in self.registry.ids():
            if not self._eligibility(model_id, tokens)[0]:
                continue
            cost = self.resources.estimate_cost(model_id, int(tokens * 0.7), tokens - int(tokens * 0.7))
            strength = self.registry.config(model_id).capabilities.get("hld", 5.0)
            options.append((cost, -strength, model_id))
        return min(options)[2] if options else None

    # ------------------------------------------------------------------ #
    # Reservations
    # ------------------------------------------------------------------ #

    def release_all(self) -> None:
        for stage in RESERVABLE_STAGES:
            self.resources.release(stage)

    def stage_done(self, stage: str) -> None:
        """Call when a stage has finished (or been abandoned) to free its reserved tokens."""
        self.resources.release(stage)

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #

    def plan(self, analysis: TaskAnalysis) -> Plan:
        """Build the execution plan. Raises NoModelAvailable if nothing can run right now."""
        self.release_all()
        ids = self.registry.ids()
        if not ids:
            raise NoModelAvailable("No models are usable: enable a model and set its API key")
        usable = [m for m in ids if self.resources.status(m).available]
        if not usable:
            raise NoModelAvailable("No model is available right now: " + self._explain_unavailable(),
                                   self._soonest_wait())

        plan = Plan(use_orchestration=analysis.needs_orchestration)
        plan.reasons.append(analysis.orchestration_reason)
        if analysis.needs_orchestration:
            self._plan_orchestrated(plan, analysis, usable)
        else:
            self._plan_direct(plan, analysis)

        self._log("workflow_selection",
                  "orchestrated workflow" if plan.use_orchestration else "single model, direct",
                  analysis.orchestration_reason,
                  {"complexity": analysis.complexity, "task_type": analysis.task_type,
                   "assignments": {s: a.model_id for s, a in plan.assignments.items()},
                   "benchmark_candidates": plan.benchmark_candidates, "warnings": plan.warnings})
        return plan

    def _assign(self, stage, analysis, exclude=(), ignore_reservation=None, reserve=True) -> StageAssignment:
        tokens = self.stage_tokens(stage, analysis)
        ranked = self.rank_models(stage, analysis, tokens, exclude, ignore_reservation)
        eligible = [r for r in ranked if r.eligible]
        if not eligible:
            why = "; ".join(f"{r.model_id} ({r.reason})" for r in ranked) or "no models"
            return StageAssignment(stage, None, tokens, 0.0, f"no model can take this stage right now: {why}")
        best = eligible[0]
        reserved = bool(reserve and stage in RESERVABLE_STAGES
                        and self.resources.reserve(best.model_id, tokens, stage))
        return StageAssignment(
            stage=stage, model_id=best.model_id, estimated_tokens=tokens, estimated_cost_usd=best.est_cost_usd,
            reason=(f"best score {best.score:.2f} (quality {best.quality:.2f}, reliability "
                    f"{best.reliability:.2f}, est. cost ${best.est_cost_usd:.4f})"),
            fallbacks=[r.model_id for r in eligible[1:1 + MAX_FALLBACKS]], reserved=reserved)

    def _plan_direct(self, plan: Plan, analysis: TaskAnalysis) -> None:
        plan.skip_benchmark = True
        assignment = self._assign("coding", analysis)
        if assignment.model_id is None:
            raise NoModelAvailable(assignment.reason, self._soonest_wait(assignment.estimated_tokens))
        plan.assignments["coding"] = assignment
        self._log("model_selection", f"{assignment.model_id} for coding (direct)", assignment.reason,
                  {"estimated_tokens": assignment.estimated_tokens})

    def _plan_orchestrated(self, plan: Plan, analysis: TaskAnalysis, usable: list) -> None:
        # 1. Reserve the most valuable later stages first, so the benchmark can't spend their tokens.
        for stage in ("review", "hld"):
            assignment = self._assign(stage, analysis)
            plan.assignments[stage] = assignment
            if assignment.model_id is None:
                plan.warnings.append(f"{stage}: {assignment.reason}")
            else:
                self._log("model_selection", f"{assignment.model_id} for {stage}", assignment.reason,
                          {"estimated_tokens": assignment.estimated_tokens, "reserved": assignment.reserved})

        # 2. Does history already tell us who should code?
        rec = self.store.recommend(analysis.task_type, usable, language=analysis.primary_language,
                                   complexity=analysis.complexity)
        self._log("benchmark_decision", "skip benchmark" if rec.skip_benchmark else "benchmark needed",
                  rec.reason, {"best_model": rec.best_model, "baseline_score": rec.baseline_score,
                               "needs_benchmark": rec.needs_benchmark})

        if rec.skip_benchmark:
            plan.skip_benchmark = True
            pool = {p.model_id: (p.weighted_avg_score, "history") for p in rec.ranking if p.model_id in usable}
            plan.coder_decision = self._decide_coder(plan, analysis, pool)
            return

        # 3. Choose benchmark candidates among models that can afford the lite task
        tokens = self.stage_tokens("lite_benchmark", analysis)
        ranked = self.rank_models("lite_benchmark", analysis, tokens)
        eligible_ids = [r.model_id for r in ranked if r.eligible]
        candidates = [m for m in eligible_ids if m in rec.needs_benchmark][: self.settings.max_benchmark_candidates]
        for model_id in rec.needs_benchmark:
            if model_id not in eligible_ids:
                plan.warnings.append(f"{model_id} cannot afford the benchmark right now and will not compete")

        has_baseline = rec.best_model is not None and rec.baseline_score is not None and rec.best_model not in candidates
        if len(candidates) + (1 if has_baseline else 0) < 2:
            plan.skip_benchmark = True
            plan.warnings.append("not enough quota for a fair benchmark: choosing the coder from history "
                                 "and capability instead")
            plan.coder_decision = self._decide_coder(plan, analysis, {})
            return

        plan.benchmark_candidates = candidates
        if has_baseline:
            plan.baseline_model, plan.baseline_score = rec.best_model, rec.baseline_score
        self._log("model_selection", f"benchmark with {', '.join(candidates)}", rec.reason,
                  {"per_candidate_tokens": tokens, "baseline": plan.baseline_model})

    # ------------------------------------------------------------------ #
    # Choosing the coder
    # ------------------------------------------------------------------ #

    def select_coder(self, plan: Plan, analysis: TaskAnalysis, results: Mapping) -> CoderDecision:
        """
        Pick the coding model from benchmark scores (model_id -> 0..1). A proven historical leader
        (plan.baseline_model) competes with its historical score without being re-benchmarked.
        Also picks the reviewer, preferring a different model from the coder.
        """
        pool = {m: (float(score), "benchmark") for m, score in results.items() if m in self.registry}
        if plan.baseline_model and plan.baseline_model not in pool and plan.baseline_model in self.registry \
                and plan.baseline_score is not None:
            pool[plan.baseline_model] = (plan.baseline_score, "history")
        decision = self._decide_coder(plan, analysis, pool)
        plan.coder_decision = decision
        return decision

    def _decide_coder(self, plan: Plan, analysis: TaskAnalysis, pool: dict) -> CoderDecision:
        tokens = self.stage_tokens("coding", analysis)
        note = ""
        if not any(score > 0 for score, _ in pool.values()):
            ranked = self.rank_models("coding", analysis, tokens, ignore_reservation="coding")
            pool = {r.model_id: (r.score, "prior") for r in ranked}
            note = "no usable benchmark or history scores: ranked by capability, reliability and cost. "

        entries = []
        for model_id, (score, source) in pool.items():
            entries.append({
                "model_id": model_id, "score": round(score, 4), "source": source,
                "wait": self.estimated_wait(model_id, tokens, ignore_reservation="coding"),
                "cost": self.resources.estimate_cost(model_id, int(tokens * 0.4), tokens - int(tokens * 0.4)),
                "reliability": self.resources.reliability(model_id),
            })
        if not entries:
            return self._finish_decision(CoderDecision("fail", None, None, None, None, None, [],
                                                       note + "no models are configured"), plan, analysis)

        top = max(e["score"] for e in entries)
        contenders = sorted((e for e in entries if top - e["score"] < TIE_MARGIN),
                            key=lambda e: (e["cost"], -e["reliability"], e["model_id"]))
        rest = sorted((e for e in entries if top - e["score"] >= TIE_MARGIN),
                      key=lambda e: (-e["score"], e["model_id"]))
        ranking = contenders + rest
        best = ranking[0]
        feasible = [e for e in ranking if e["wait"] == 0.0]

        if feasible:
            chosen = feasible[0]
            can_wait = best["wait"] is not None and 0.0 < best["wait"] <= MAX_WAIT_SECONDS
            if chosen is not best and can_wait and best["score"] - chosen["score"] > WAIT_SCORE_GAP:
                decision = CoderDecision(
                    "wait", best["model_id"], best["score"], best["wait"], chosen["model_id"], None, ranking,
                    note + f"{best['model_id']} scores {best['score']:.2f} vs {chosen['score']:.2f} for "
                           f"{chosen['model_id']} but is unavailable for ~{best['wait']:.0f}s: waiting is worth it "
                           f"(or take {chosen['model_id']} now)")
            else:
                why = "highest score" if chosen is best else (
                    f"best score among models usable now ({best['model_id']} is unavailable "
                    f"{'for ~%.0fs' % best['wait'] if best['wait'] is not None else 'for the foreseeable future'}"
                    f"{', too long to wait' if best['wait'] else ''})")
                decision = CoderDecision("use", chosen["model_id"], chosen["score"], None, None, None, ranking,
                                         note + f"{chosen['model_id']} chosen ({chosen['source']} score "
                                                f"{chosen['score']:.2f}): {why}")
        else:
            waitable = [e for e in ranking if e["wait"] is not None]
            if waitable:
                short = [e for e in waitable if e["wait"] <= MAX_WAIT_SECONDS]
                target = short[0] if short else min(waitable, key=lambda e: e["wait"])
                decision = CoderDecision("wait", target["model_id"], target["score"], target["wait"], None, None,
                                         ranking, note + f"no model is usable right now; {target['model_id']} "
                                                         f"should be in ~{target['wait']:.0f}s")
            else:
                decision = CoderDecision("fail", None, None, None, None, None, ranking,
                                         note + "no model can afford the coding stage, even after waiting")
        return self._finish_decision(decision, plan, analysis)

    def _finish_decision(self, decision: CoderDecision, plan: Plan, analysis: TaskAnalysis) -> CoderDecision:
        if decision.action == "use":
            tokens = self.stage_tokens("coding", analysis)
            self.resources.release("coding")
            reserved = self.resources.reserve(decision.model_id, tokens, "coding")
            entry = next(e for e in decision.ranking if e["model_id"] == decision.model_id)
            plan.assignments["coding"] = StageAssignment(
                "coding", decision.model_id, tokens, round(entry["cost"], 6), decision.reason,
                [e["model_id"] for e in decision.ranking if e["model_id"] != decision.model_id
                 and e["wait"] == 0.0][:MAX_FALLBACKS], reserved)
            decision.reviewer_id = self.choose_reviewer(plan, analysis, decision.model_id)
        self._log("coder_selection", f"{decision.action}: {decision.model_id}", decision.reason,
                  {"ranking": decision.ranking, "wait_seconds": decision.wait_seconds,
                   "alternative": decision.alternative_id, "reviewer": decision.reviewer_id})
        return decision

    def choose_reviewer(self, plan: Plan, analysis: TaskAnalysis, coder_id: str) -> Optional[str]:
        """(Re)assign the review stage, preferring a different model from the coder for an independent view."""
        tokens = self.stage_tokens("review", analysis)
        ranked = [r for r in self.rank_models("review", analysis, tokens, ignore_reservation="review") if r.eligible]
        different = [r for r in ranked if r.model_id != coder_id]
        pick = (different or ranked or [None])[0]

        self.resources.release("review")
        if pick is None:
            plan.assignments["review"] = StageAssignment("review", None, tokens, 0.0,
                                                         "no model can afford the review right now")
            plan.warnings.append("review: no model can afford it right now")
            return None
        reserved = self.resources.reserve(pick.model_id, tokens, "review")
        plan.assignments["review"] = StageAssignment(
            "review", pick.model_id, tokens, pick.est_cost_usd,
            ("independent of the coder; " if pick.model_id != coder_id else "same model as the coder (no alternative); ")
            + f"best score {pick.score:.2f}",
            [r.model_id for r in ranked if r.model_id != pick.model_id][:MAX_FALLBACKS], reserved)
        return pick.model_id

    # ------------------------------------------------------------------ #
    # Failures
    # ------------------------------------------------------------------ #

    def fallback_model(self, stage: str, analysis: TaskAnalysis, failed: Iterable,
                       plan: Optional[Plan] = None) -> Optional[str]:
        """Next-best model for a stage after `failed` ones broke. Moves the stage's reservation along."""
        tokens = self.stage_tokens(stage, analysis)
        ignore = stage if stage in RESERVABLE_STAGES else None
        pick = self.pick_model(stage, analysis, exclude=failed, tokens=tokens, ignore_reservation=ignore)
        failed = list(failed)
        if pick is None:
            self._log("fallback", f"no fallback for {stage}", f"failed: {failed}; nobody else can take it")
            return None
        reserved = False
        if ignore:
            self.resources.release(stage)
            reserved = self.resources.reserve(pick.model_id, tokens, stage)
            if plan is not None:
                plan.assignments[stage] = StageAssignment(
                    stage, pick.model_id, tokens, pick.est_cost_usd,
                    f"fallback after {', '.join(failed)} failed; score {pick.score:.2f}", [], reserved)
        self._log("fallback", f"{pick.model_id} takes over {stage}", f"failed: {failed}",
                  {"stage": stage, "score": pick.score})
        return pick.model_id


# --------------------------------------------------------------------------- #
# Self-test (no network, no API keys): python -m core.scheduler
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from config.settings import ModelConfig
    from core.benchmark_store import BenchmarkRecord
    from core.logger import setup_logging
    from core.task_analyzer import TaskAnalyzer

    setup_logging("ERROR")

    class FakeClock:
        def __init__(self, t: float = 1_000_000.0):
            self.t = t

        def __call__(self) -> float:
            return self.t

    class _Registry:
        """Minimal stand-in for ModelRegistry: the scheduler only needs ids(), config() and `in`."""

        def __init__(self, models):
            self._m = {m.id: m for m in models}

        def ids(self):
            return list(self._m)

        def config(self, model_id):
            return self._m[model_id]

        def __contains__(self, model_id):
            return model_id in self._m

    def make_models():
        return [
            ModelConfig(id="claude", provider="anthropic", model_name="c",
                        capabilities={"coding": 9.0, "hld": 8.5, "review": 9.0},
                        cost_per_1k_input=0.003, cost_per_1k_output=0.015, token_budget_per_window=50_000),
            ModelConfig(id="gpt", provider="openai", model_name="g",
                        capabilities={"coding": 8.0, "hld": 8.5, "review": 8.0},
                        cost_per_1k_input=0.0025, cost_per_1k_output=0.01, token_budget_per_window=100_000),
            ModelConfig(id="gemini", provider="gemini", model_name="m",
                        capabilities={"coding": 7.5, "hld": 7.5, "review": 7.5},
                        cost_per_1k_input=0.00125, cost_per_1k_output=0.01, token_budget_per_window=200_000),
            ModelConfig(id="local", provider="local", model_name="q",
                        capabilities={"coding": 6.0, "hld": 5.0, "review": 5.0}),
        ]

    settings = OrchestratorSettings(min_history_to_skip_benchmark=5, history_confidence_margin=0.05)
    analyzer = TaskAnalyzer(settings)
    COMPLEX = analyzer.analyze("Build me a desktop app that monitors my downloads and automatically categorizes them.")
    SIMPLE = analyzer.analyze("Write a simple python script that renames all files in a folder to lowercase.")
    HEAVY = analyzer.analyze("Build a scalable REST API with JWT authentication, PostgreSQL, Redis caching, a "
                             "background job queue, role-based access control and Docker deployment")
    assert COMPLEX.needs_orchestration and not SIMPLE.needs_orchestration

    tmp = tempfile.TemporaryDirectory()

    def make_env(model_ids=None):
        clock = FakeClock()
        models = [m for m in make_models() if model_ids is None or m.id in model_ids]
        rm = ResourceManager(models, settings, persist=False, clock=clock)
        store = BenchmarkStore(":memory:", settings=settings, clock=clock)
        log_path = Path(tmp.name) / f"decisions_{id(clock)}.jsonl"
        decisions = DecisionLogger(project_id="test", log_path=log_path)
        return Scheduler(_Registry(models), rm, store, settings, decisions), rm, store, decisions

    def add_history(store, scores, runs=6, analysis=COMPLEX):
        for model_id, score in scores.items():
            for _ in range(runs):
                store.record_result(BenchmarkRecord(
                    model_id=model_id, task_type=analysis.task_type, final_score=score,
                    language="python", complexity=analysis.complexity))

    # 1. Simple task: direct path, the free local model wins on cost; tokens are reserved
    s, rm, store, dec = make_env()
    plan = s.plan(SIMPLE)
    assert not plan.use_orchestration and plan.skip_benchmark and plan.model_for("coding") == "local"
    assert "hld" not in plan.assignments and rm.status("local").reserved_tokens > 0
    print("PASS 1: simple task -> direct, coder =", plan.model_for("coding"))

    # 2. Hard task: quality beats cost
    assert s.pick_model("coding", HEAVY).model_id == "claude"
    print("PASS 2: heavy task buys quality ->", s.pick_model("coding", HEAVY).model_id)

    # 3. Orchestrated, no history: benchmark with 3 candidates; hld/review reserved; coder undecided
    s, rm, store, dec = make_env()
    plan = s.plan(COMPLEX)
    assert plan.use_orchestration and not plan.skip_benchmark and len(plan.benchmark_candidates) == 3
    assert plan.baseline_model is None and "coding" not in plan.assignments
    hld_model, review_model = plan.model_for("hld"), plan.model_for("review")
    assert hld_model and review_model and plan.assignments["hld"].reserved and plan.assignments["review"].reserved
    assert rm.status(review_model).reserved_tokens >= plan.assignments["review"].estimated_tokens
    print("PASS 3: benchmark with", plan.benchmark_candidates, "| hld:", hld_model, "| review:", review_model)

    # 4. Clear history leader: benchmark skipped, coder chosen, reviewer is a different model
    s, rm, store, dec = make_env()
    add_history(store, {"claude": 0.9, "gpt": 0.7, "gemini": 0.6, "local": 0.4})
    plan = s.plan(COMPLEX)
    d = plan.coder_decision
    assert plan.skip_benchmark and d.action == "use" and d.model_id == "claude"
    assert plan.model_for("coding") == "claude" and plan.assignments["coding"].reserved
    assert d.reviewer_id and d.reviewer_id != "claude" and plan.model_for("review") == d.reviewer_id
    print("PASS 4: history skip -> coder", d.model_id, "| reviewer", d.reviewer_id)

    # 5. Quota pressure: claude nearly out, resets soon -> wait for it (alternative offered)
    s, rm, store, dec = make_env()
    add_history(store, {"claude": 0.9, "gpt": 0.7, "gemini": 0.6, "local": 0.4})
    rm.set_remaining("claude", 14_000, resets_in_seconds=300)
    plan = s.plan(COMPLEX)
    d = plan.coder_decision
    assert d.action == "wait" and d.model_id == "claude" and abs(d.wait_seconds - 300) < 1e-6
    assert d.alternative_id == "gpt" and "coding" not in plan.assignments
    print("PASS 5: claude resets in 300s -> wait, alternative =", d.alternative_id)

    # 6. Same, but the reset is far away -> don't wait, use the best available model
    s, rm, store, dec = make_env()
    add_history(store, {"claude": 0.9, "gpt": 0.7, "gemini": 0.6, "local": 0.4})
    rm.set_remaining("claude", 14_000, resets_in_seconds=1860)
    plan = s.plan(COMPLEX)
    d = plan.coder_decision
    assert d.action == "use" and d.model_id == "gpt" and "too long" in d.reason
    print("PASS 6: claude resets in 1860s -> use", d.model_id)

    # 7. History leader + unproven models: only the unproven benchmark; baseline competes without re-running
    s, rm, store, dec = make_env()
    add_history(store, {"claude": 0.9})
    plan = s.plan(COMPLEX)
    assert not plan.skip_benchmark and plan.baseline_model == "claude" and plan.baseline_score == 0.9
    assert "claude" not in plan.benchmark_candidates and len(plan.benchmark_candidates) == 3
    d = s.select_coder(plan, COMPLEX, {"gpt": 0.8, "gemini": 0.7, "local": 0.5})
    assert d.action == "use" and d.model_id == "claude" and d.ranking[0]["source"] == "history"
    d = s.select_coder(plan, COMPLEX, {"gpt": 0.95, "gemini": 0.7})
    assert d.model_id == "gpt" and plan.model_for("coding") == "gpt"
    print("PASS 7: baseline", plan.baseline_model, "must be beaten; gpt 0.95 > 0.9 wins")

    # 8. select_coder: clear winner, tie goes to the cheaper model, reviewer differs
    s, rm, store, dec = make_env()
    plan = s.plan(COMPLEX)
    d = s.select_coder(plan, COMPLEX, {"claude": 0.82, "gpt": 0.85, "gemini": 0.70})
    assert d.model_id == "gpt" and d.reviewer_id != "gpt"
    d = s.select_coder(plan, COMPLEX, {"claude": 0.85, "gpt": 0.84, "gemini": 0.5})
    assert d.model_id == "gpt", "within the tie margin the cheaper model should win"
    assert plan.model_for("coding") == "gpt" and plan.model_for("review") != "gpt"
    reserved_total = sum(rm.status(m).reserved_tokens for m in ("claude", "gpt", "gemini", "local"))
    assert reserved_total == sum(a.estimated_tokens for a in plan.assignments.values() if a.reserved)
    print("PASS 8: select_coder (winner, tie -> cheaper, independent reviewer, no leaked reservations)")

    # 9. All candidates failed (score 0): fall back to capability ranking
    d = s.select_coder(plan, COMPLEX, {"gpt": 0.0, "gemini": 0.0})
    assert d.action == "use" and "no usable" in d.reason
    print("PASS 9: all-zero benchmark -> prior-based choice:", d.model_id)

    # 10. Too little quota for a fair benchmark: skip it with a warning
    s, rm, store, dec = make_env(["claude", "gpt"])
    rm.set_remaining("gpt", 2_000)
    plan = s.plan(COMPLEX)
    assert plan.skip_benchmark and plan.coder_decision.model_id == "claude"
    assert any("benchmark" in w for w in plan.warnings)
    print("PASS 10: quota too low for a fair benchmark ->", plan.warnings[-1])

    # 11. Fallback after a failure moves the reservation
    s, rm, store, dec = make_env()
    plan = s.plan(COMPLEX)
    s.select_coder(plan, COMPLEX, {"claude": 0.9, "gpt": 0.7})
    before = rm.status("claude").reserved_tokens
    new = s.fallback_model("coding", COMPLEX, failed=["claude"], plan=plan)
    assert new and new != "claude" and plan.model_for("coding") == new
    assert rm.status("claude").reserved_tokens < before or before == 0
    assert rm.status(new).reserved_tokens >= plan.assignments["coding"].estimated_tokens
    assert s.fallback_model("lite_benchmark", COMPLEX, failed=["claude", "gpt", "gemini", "local"]) is None
    print("PASS 11: fallback ->", new)

    # 12. stage_done frees reserved tokens
    s, rm, store, dec = make_env()
    plan = s.plan(COMPLEX)
    hld_model = plan.model_for("hld")
    before = rm.status(hld_model).reserved_tokens
    s.stage_done("hld")
    assert rm.status(hld_model).reserved_tokens == before - plan.assignments["hld"].estimated_tokens
    print("PASS 12: stage_done releases the reservation")

    # 13. Analyzer pick is the cheap one; nothing available raises with a wait estimate
    s, rm, store, dec = make_env()
    assert s.pick_analyzer() == "local"
    for m in ("claude", "gpt", "gemini", "local"):
        rm.set_cooldown(m, 120)
    try:
        s.plan(COMPLEX)
        raise SystemExit("FAIL 13")
    except NoModelAvailable as e:
        assert abs(e.wait_seconds - 120) < 1e-6 and "cooling down" in str(e)
    print("PASS 13: pick_analyzer = local; all models cooling down -> NoModelAvailable (wait 120s)")

    # 14. Decisions are logged and the plan serialises
    s, rm, store, dec = make_env()
    add_history(store, {"claude": 0.9, "gpt": 0.7, "gemini": 0.6, "local": 0.4})
    plan = s.plan(COMPLEX)
    categories = {r["category"] for r in dec.read_all()}
    assert {"workflow_selection", "benchmark_decision", "model_selection", "coder_selection"} <= categories, categories
    import json
    json.dumps(plan.to_dict(), default=str)
    print("PASS 14: decisions logged:", sorted(categories))
    tmp.cleanup()