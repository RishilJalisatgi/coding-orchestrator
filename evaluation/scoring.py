"""
evaluation/scoring.py

Turns measurements into the 0..1 metrics the orchestrator ranks models by, and combines them into a
final score. Pure functions, no I/O, no third-party packages.

    from evaluation.scoring import (compute_score, score_tests, score_security, score_lint,
                                    blend_judge, rank_candidates)

    metrics = {
        "test_pass_rate": score_tests(passed=18, failed=2, errors=0),     # 0.9
        "correctness":    score_correctness(compiles=1.0, runs=True, checks=0.9),
        "security":       score_security(high=0, medium=1, low=3),
        "code_quality":   score_lint(issue_count=12, lines=400),
        "runtime":        None,        # not measured (tool missing / not applicable)
    }
    result = compute_score(metrics, settings.orchestrator.weights_for("backend_api"))
    result.final_score     # 0..1, computed over the metrics that WERE measured
    result.skipped         # weighted metrics that were not measured (say so, don't pretend)
    result.coverage        # share of the total weight that was actually measured

Rules that keep the numbers honest:
  * A metric of None means "not measured". It is dropped and the remaining weights are renormalised;
    it is never silently counted as 0 or 1. `coverage` / `reliable` tell the caller how much to trust it.
  * Correctness gate: code that does not work cannot win on style. Below GATE_THRESHOLD correctness,
    the final score is scaled down proportionally.
  * The LLM judge supplements, never replaces: it may only influence the subjective metrics
    (code_quality, architecture, documentation) and is capped at `judge_share` of each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional

METRIC_KEYS = ("correctness", "test_pass_rate", "runtime", "memory", "security",
               "code_quality", "architecture", "documentation")
SUBJECTIVE_KEYS = ("code_quality", "architecture", "documentation")     # the judge may influence these only

GATE_THRESHOLD = 0.5          # correctness below this scales the whole score down
MIN_COVERAGE = 0.5            # below this share of measured weight, the score is flagged unreliable
DEFAULT_JUDGE_SHARE = 0.25


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class ScoreResult:
    final_score: float
    metrics: dict = field(default_factory=dict)          # metric -> 0..1 or None (as supplied, cleaned)
    weights: dict = field(default_factory=dict)          # effective weights over MEASURED metrics (sum = 1)
    contributions: dict = field(default_factory=dict)    # metric -> weight * value
    skipped: list = field(default_factory=list)          # weighted metrics that were not measured
    coverage: float = 0.0                                # measured share of the original weight (0..1)
    gate_factor: float = 1.0                             # < 1 when the correctness gate reduced the score
    ungated_score: float = 0.0
    judge_applied: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.coverage >= MIN_COVERAGE

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["reliable"] = self.reliable
        return data

    @classmethod
    def from_dict(cls, raw: dict) -> "ScoreResult":
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})

    def explain(self) -> str:
        """One line for the decision log."""
        parts = [f"{k}={self.metrics[k]:.2f}" for k in METRIC_KEYS if self.metrics.get(k) is not None]
        text = f"score {self.final_score:.3f} ({', '.join(parts) or 'nothing measured'})"
        if self.skipped:
            text += f"; not measured: {', '.join(self.skipped)}"
        if self.gate_factor < 1.0:
            text += f"; correctness gate x{self.gate_factor:.2f}"
        if not self.reliable:
            text += f"; LOW COVERAGE {self.coverage:.0%}"
        return text


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def clamp01(value) -> Optional[float]:
    """Clamp to 0..1. None, NaN, inf and non-numbers become None (not measured)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return max(0.0, min(1.0, number))


def _mean(values) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _weighted_mean(pairs) -> Optional[float]:
    """pairs: [(value or None, weight)]. Missing values are dropped and weights renormalised."""
    present = [(v, w) for v, w in pairs if v is not None and w > 0]
    total = sum(w for _, w in present)
    return sum(v * w for v, w in present) / total if total else None


# --------------------------------------------------------------------------- #
# Raw measurements -> metrics (all return 0..1, or None when nothing was measured)
# --------------------------------------------------------------------------- #

def score_tests(passed: int = 0, failed: int = 0, errors: int = 0, skipped: int = 0) -> Optional[float]:
    """test_pass_rate: passed / (passed + failed + errors). Skipped tests are ignored. No tests -> None."""
    total = max(0, passed) + max(0, failed) + max(0, errors)
    if total == 0:
        return None
    return clamp01(max(0, passed) / total)


def score_correctness(compiles: Optional[float] = None, runs: Optional[bool] = None,
                      checks: Optional[float] = None) -> Optional[float]:
    """
    correctness from independent signals (any may be missing):
      compiles  share of source files that parse/compile (0..1)
      runs      the program/module imports or starts without raising (True/False)
      checks    share of acceptance checks/tests written by the ORCHESTRATOR (not the candidate) that pass
    """
    return _weighted_mean([
        (clamp01(compiles), 0.2),
        (None if runs is None else (1.0 if runs else 0.0), 0.3),
        (clamp01(checks), 0.5),
    ])


def score_lint(issue_count: int, lines: int) -> Optional[float]:
    """Lint/type issues per 100 lines -> 0..1 (0 issues = 1.0, 2/100 lines = 0.5, 10/100 = 0.17)."""
    if lines is None or lines <= 0 or issue_count is None:
        return None
    density = max(0, issue_count) / lines * 100.0
    return clamp01(1.0 / (1.0 + density / 2.0))


def score_complexity(avg_cyclomatic: Optional[float]) -> Optional[float]:
    """Average cyclomatic complexity: <= 5 is 1.0, >= 20 is 0.0, linear between."""
    if avg_cyclomatic is None:
        return None
    return clamp01(1.0 - (avg_cyclomatic - 5.0) / 15.0)


def score_quality(lint: Optional[float] = None, types: Optional[float] = None,
                  complexity: Optional[float] = None) -> Optional[float]:
    """code_quality = weighted mean of whichever of lint / type-check / complexity scores exist."""
    return _weighted_mean([(clamp01(lint), 0.5), (clamp01(types), 0.3), (clamp01(complexity), 0.2)])


def score_security(high: int = 0, medium: int = 0, low: int = 0, scanned: bool = True) -> Optional[float]:
    """1 - penalty: a high-severity finding costs 0.4, medium 0.15, low 0.03. `scanned=False` -> None."""
    if not scanned:
        return None
    penalty = 0.4 * max(0, high) + 0.15 * max(0, medium) + 0.03 * max(0, low)
    return clamp01(1.0 - penalty)


def score_runtime(seconds: Optional[float], reference_seconds: Optional[float],
                  completed: bool = True) -> Optional[float]:
    """1.0 at or under the reference time, 1/ratio beyond it. A run that timed out/crashed scores 0."""
    if not completed:
        return 0.0
    if seconds is None or reference_seconds is None or reference_seconds <= 0:
        return None
    ratio = max(seconds, 0.0) / reference_seconds
    return clamp01(1.0 if ratio <= 1.0 else 1.0 / ratio)


def score_memory(peak_mb: Optional[float], reference_mb: Optional[float], completed: bool = True) -> Optional[float]:
    """Same shape as runtime: at or under the reference is 1.0, 1/ratio beyond."""
    return score_runtime(peak_mb, reference_mb, completed)


def score_documentation(docstring_ratio: Optional[float] = None, has_readme: Optional[bool] = None,
                        has_usage: Optional[bool] = None) -> Optional[float]:
    """Share of public functions/classes with docstrings (60%), a README (25%), usage instructions (15%)."""
    return _weighted_mean([
        (clamp01(docstring_ratio), 0.60),
        (None if has_readme is None else float(bool(has_readme)), 0.25),
        (None if has_usage is None else float(bool(has_usage)), 0.15),
    ])


def relative_to_best(values: Mapping, lower_is_better: bool = True) -> dict:
    """
    Score candidates against each other when there is no absolute reference (e.g. runtime of three
    candidates running the same workload): the best gets 1.0, others best/value (or value/best).
    Entries that are None stay None. A zero best is handled (everyone equal -> 1.0).
    """
    present = {k: float(v) for k, v in values.items() if v is not None and not math.isnan(float(v))}
    out: dict = {k: None for k in values}
    if not present:
        return out
    best = min(present.values()) if lower_is_better else max(present.values())
    for key, value in present.items():
        if lower_is_better:
            out[key] = 1.0 if value <= best else clamp01(best / value)
        else:
            out[key] = 1.0 if best <= 0 else clamp01(value / best)
    return out


# --------------------------------------------------------------------------- #
# Combining
# --------------------------------------------------------------------------- #

def compute_score(metrics: Mapping, weights: Mapping, gate: bool = True) -> ScoreResult:
    """
    Weighted mean over the metrics that were measured; unmeasured ones are listed in `skipped`.
    Weights for unknown metric names are ignored; metrics with no weight do not count.
    """
    clean = {k: clamp01(metrics.get(k)) for k in METRIC_KEYS if k in metrics}
    weight = {k: float(w) for k, w in weights.items() if k in METRIC_KEYS and w and w > 0}
    total_weight = sum(weight.values())
    notes: list = []
    ignored = sorted(k for k in weights if k not in METRIC_KEYS)
    if ignored:
        notes.append(f"ignored unknown weight keys: {', '.join(ignored)}")

    measured = {k: w for k, w in weight.items() if clean.get(k) is not None}
    skipped = sorted(k for k in weight if k not in measured)
    measured_weight = sum(measured.values())

    if not measured_weight:
        notes.append("no weighted metric was measured")
        return ScoreResult(0.0, clean, {}, {}, skipped, 0.0, 1.0, 0.0, [], notes)

    effective = {k: w / measured_weight for k, w in measured.items()}
    exact = {k: effective[k] * clean[k] for k in effective}
    ungated = sum(exact.values())                      # sum the unrounded values: rounding first loses 1e-6
    contributions = {k: round(v, 6) for k, v in exact.items()}

    factor = 1.0
    correctness = clean.get("correctness")
    if gate and correctness is not None and correctness < GATE_THRESHOLD:
        factor = correctness / GATE_THRESHOLD
        notes.append(f"correctness {correctness:.2f} < {GATE_THRESHOLD}: score scaled by {factor:.2f}")

    return ScoreResult(
        final_score=round(clamp01(ungated * factor) or 0.0, 6),
        metrics=clean, weights={k: round(v, 6) for k, v in effective.items()}, contributions=contributions,
        skipped=skipped, coverage=round(measured_weight / total_weight, 6) if total_weight else 0.0,
        gate_factor=round(factor, 6), ungated_score=round(ungated, 6), notes=notes)


def failure_score(reason: str = "candidate produced nothing usable") -> ScoreResult:
    """A candidate that failed to produce code (model error, nothing parsed) scores 0 and says why."""
    return ScoreResult(0.0, {}, {}, {}, [], 1.0, 1.0, 0.0, [], [reason])


def blend_judge(result: ScoreResult, judge: Mapping, weights: Mapping,
                judge_share: float = DEFAULT_JUDGE_SHARE) -> ScoreResult:
    """
    Add an LLM judge's opinion WITHOUT letting it override objective evidence.
      * only code_quality / architecture / documentation can change; other judge keys are ignored
      * when the objective value exists: blended = (1 - share) * objective + share * judge
      * when it does not (e.g. ruff missing): the judge value is used, and a note says so
    Returns a new ScoreResult recomputed with the same weights.
    """
    share = max(0.0, min(0.5, judge_share))          # the judge can never be the majority voice
    merged = dict(result.metrics)
    applied, notes = [], list(result.notes)
    for key in SUBJECTIVE_KEYS:
        opinion = clamp01(judge.get(key)) if key in judge else None
        if opinion is None:
            continue
        objective = clamp01(merged.get(key))
        if objective is None:
            merged[key] = opinion
            notes.append(f"{key}: judge only (no objective measurement)")
        else:
            merged[key] = (1.0 - share) * objective + share * opinion
        applied.append(key)
    dropped = sorted(k for k in judge if k in METRIC_KEYS and k not in SUBJECTIVE_KEYS)
    if dropped:
        notes.append(f"judge opinion ignored for objective metrics: {', '.join(dropped)}")
    out = compute_score(merged, weights)
    out.judge_applied = applied
    out.notes = [n for n in out.notes if n not in notes] + notes
    return out


def rank_candidates(results: Mapping) -> list:
    """
    [(model_id, final_score)] best first. Ties are broken by higher coverage (better evidence), then
    higher correctness, then model id, so the order is deterministic.
    """
    def key(item):
        model_id, res = item
        return (-res.final_score, -res.coverage, -(res.metrics.get("correctness") or 0.0), model_id)
    return [(m, r.final_score) for m, r in sorted(results.items(), key=key)]


def margin(ranked: list) -> float:
    """Lead of the first over the second (0 when there is only one candidate)."""
    return round(ranked[0][1] - ranked[1][1], 6) if len(ranked) >= 2 else 0.0


# --------------------------------------------------------------------------- #
# Self-test:  python -m evaluation.scoring     (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from config.settings import OrchestratorSettings

    def close(a, b, tol=1e-6):
        return abs(a - b) <= tol

    orch = OrchestratorSettings()
    W = orch.weights_for("backend_api")

    # 1. Raw measurements -> metrics
    assert score_tests(18, 2, 0) == 0.9 and score_tests(0, 0, 0) is None and score_tests(3, 0, 1, skipped=9) == 0.75
    assert score_lint(0, 100) == 1.0 and close(score_lint(2, 100), 0.5) and score_lint(5, 0) is None
    assert close(score_lint(10, 100), 1 / 6)
    assert score_security(0, 0, 0) == 1.0 and close(score_security(1, 1, 0), 0.45) and score_security(3) == 0.0
    assert score_security(scanned=False) is None
    assert score_runtime(1.0, 2.0) == 1.0 and close(score_runtime(4.0, 2.0), 0.5)
    assert score_runtime(None, 2.0) is None and score_runtime(None, None, completed=False) == 0.0
    assert score_complexity(4) == 1.0 and score_complexity(20) == 0.0 and close(score_complexity(12.5), 0.5)
    assert close(score_documentation(0.5, True, False), (0.5 * 0.6 + 0.25 + 0) / 1.0)
    assert score_documentation() is None
    print("PASS 1: raw measurements -> 0..1 metrics (tests, lint, security, runtime, complexity, docs)")

    # 2. Correctness combines whatever signals exist
    assert close(score_correctness(1.0, True, 0.9), 0.2 + 0.3 + 0.45)
    assert close(score_correctness(runs=False, checks=1.0), (0 * 0.3 + 1.0 * 0.5) / 0.8)
    assert score_correctness() is None
    print("PASS 2: correctness from compile / run / acceptance-check signals")

    # 3. Weighted score; unmeasured metrics are dropped and weights renormalised (not counted as 0)
    full = {"correctness": 1.0, "test_pass_rate": 1.0, "security": 1.0, "architecture": 1.0,
            "code_quality": 1.0, "documentation": 1.0}
    r = compute_score(full, W)
    assert r.final_score == 1.0 and r.skipped == [] and r.coverage == 1.0 and r.reliable
    part = dict(full, security=None, documentation=None)
    r = compute_score(part, W)
    assert r.final_score == 1.0 and r.skipped == ["documentation", "security"] and close(r.coverage, 0.75)
    assert close(sum(r.weights.values()), 1.0)
    print("PASS 3: unmeasured metrics skipped + renormalised ->", r.skipped, f"coverage {r.coverage:.0%}")

    # 4. Different task types weigh differently
    m = {"correctness": 0.9, "test_pass_rate": 0.9, "runtime": 0.2, "security": 1.0,
         "architecture": 1.0, "code_quality": 1.0, "documentation": 1.0, "memory": 1.0}
    comp = compute_score(m, orch.weights_for("competitive_programming")).final_score
    api = compute_score(m, orch.weights_for("backend_api")).final_score
    assert comp < api, (comp, api)             # slow code hurts competitive programming far more
    print(f"PASS 4: task-type weights matter -> competitive {comp:.3f} < backend {api:.3f}")

    # 5. Correctness gate: broken code cannot win on style
    pretty = {"correctness": 0.1, "test_pass_rate": 0.2, "code_quality": 1.0, "architecture": 1.0,
              "documentation": 1.0, "security": 1.0}
    r = compute_score(pretty, W)
    assert r.gate_factor == 0.2 and close(r.final_score, 0.2 * r.ungated_score)
    assert compute_score(pretty, W, gate=False).final_score == r.ungated_score
    print(f"PASS 5: correctness gate -> ungated {r.ungated_score:.3f} -> {r.final_score:.3f}")

    # 6. Nothing measured / unknown keys / junk values
    r = compute_score({}, W)
    assert r.final_score == 0.0 and r.coverage == 0.0 and not r.reliable and r.notes
    r = compute_score({"correctness": float("nan"), "test_pass_rate": "x", "security": 1.7}, {"security": 1, "bogus": 5})
    assert r.metrics["security"] == 1.0 and r.metrics["correctness"] is None and r.final_score == 1.0
    assert any("bogus" in n for n in r.notes)
    assert failure_score("model crashed").final_score == 0.0
    print("PASS 6: empty input, NaN/junk values and unknown weight keys handled")

    # 7. Low coverage is flagged as unreliable, not hidden
    r = compute_score({"correctness": 1.0}, W)
    assert r.final_score == 1.0 and not r.reliable and "LOW COVERAGE" in r.explain()
    print("PASS 7: low coverage flagged ->", r.explain())

    # 8. Judge supplements objective data, capped, subjective keys only
    base = compute_score({"correctness": 1.0, "test_pass_rate": 1.0, "code_quality": 0.4, "architecture": None}, W)
    j = blend_judge(base, {"code_quality": 1.0, "architecture": 0.8, "correctness": 0.0, "security": 0.0}, W,
                    judge_share=0.9)                                   # share is capped at 0.5
    assert close(j.metrics["code_quality"], 0.5 * 0.4 + 0.5 * 1.0)
    assert j.metrics["architecture"] == 0.8 and j.metrics["correctness"] == 1.0
    assert j.judge_applied == ["code_quality", "architecture"] and any("ignored" in n for n in j.notes)
    assert any("judge only" in n for n in j.notes) and j.final_score > base.final_score
    assert base.metrics["code_quality"] == 0.4                          # original result untouched
    print("PASS 8: judge blended (capped 50%), objective metrics untouched, judge-only noted")

    # 9. Ranking: deterministic tie-breaks
    a = compute_score({"correctness": 0.8, "test_pass_rate": 0.8}, W)
    b = compute_score({"correctness": 0.8, "test_pass_rate": 0.8, "security": 0.8}, W)   # same score, more evidence
    c = compute_score({"correctness": 1.0, "test_pass_rate": 1.0}, W)
    ranked = rank_candidates({"zeta": a, "alpha": b, "mid": c})
    assert [m for m, _ in ranked] == ["mid", "alpha", "zeta"], ranked
    assert close(margin(ranked), 0.2) and margin(ranked[:1]) == 0.0
    print("PASS 9: ranking ->", ranked)

    # 10. Relative scoring between candidates
    rel = relative_to_best({"a": 2.0, "b": 4.0, "c": None, "d": 8.0})
    assert rel == {"a": 1.0, "b": 0.5, "c": None, "d": 0.25}
    assert relative_to_best({"a": 0.0, "b": 0.0}) == {"a": 1.0, "b": 1.0}
    assert relative_to_best({"a": 10, "b": 5}, lower_is_better=False) == {"a": 1.0, "b": 0.5}
    assert relative_to_best({}) == {}
    print("PASS 10: relative_to_best ->", rel)

    # 11. Serialisation round trip
    r = compute_score(m, W)
    again = ScoreResult.from_dict(r.to_dict())
    assert again.final_score == r.final_score and again.weights == r.weights and again.skipped == r.skipped
    import json
    json.dumps(r.to_dict())
    print("PASS 11: ScoreResult survives to_dict/from_dict and is JSON-serialisable")