"""
core/benchmark_store.py

Persistent memory of past lite-benchmark results (SQLite).

Written to by the evaluation pipeline after each benchmark:
    store.record_session(results, winner_model_id="gpt")

Read by the scheduler before deciding whether to benchmark at all:
    rec = store.recommend("backend_api", ["claude-sonnet", "gpt", "gemini"], language="python")
    if rec.skip_benchmark:
        use rec.best_model directly
    else:
        benchmark only rec.needs_benchmark (compare against rec.baseline_score)

Scores are expected in the range 0..1 (scoring.py produces them).
Task type and language are stored lower-case, so lookups are case-insensitive.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from config.settings import BENCHMARK_DB_PATH, OrchestratorSettings, ensure_dirs
from core.logger import get_logger

logger = get_logger("core.benchmark_store")

HALF_LIFE_DAYS = 30.0                 # a result this old counts half as much as a fresh one
DEFAULT_COMPLEXITY_TOLERANCE = 2.0    # "similar complexity" = within +/- 2 points (0-10 scale)
SECONDS_PER_DAY = 86400.0
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    session_id TEXT,
    project_id TEXT,
    task_type TEXT NOT NULL,
    language TEXT,
    complexity REAL,
    model_id TEXT NOT NULL,
    final_score REAL NOT NULL,
    metrics TEXT NOT NULL DEFAULT '{}',
    tokens_used INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_seconds REAL NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1,
    is_winner INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_bench_task ON benchmark_results(task_type, language);
CREATE INDEX IF NOT EXISTS idx_bench_model ON benchmark_results(model_id);
CREATE INDEX IF NOT EXISTS idx_bench_session ON benchmark_results(session_id);
"""


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkRecord:
    """One model's result on one lite benchmark."""
    model_id: str
    task_type: str
    final_score: float                      # 0..1
    language: Optional[str] = None
    complexity: Optional[float] = None      # 0..10, from the task analyzer
    metrics: dict = field(default_factory=dict)   # e.g. {"correctness": 0.9, "security": 0.7}
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    success: bool = True                    # False = produced nothing usable (score should be 0)
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    is_winner: bool = False
    notes: str = ""
    created_at: Optional[float] = None      # filled in by the store
    id: Optional[int] = None                # filled in by the store


@dataclass
class ModelPerformance:
    """Aggregated history of one model for a given kind of task."""
    model_id: str
    runs: int
    avg_score: float
    weighted_avg_score: float               # recency-weighted: this is what rankings use
    std_dev: float
    wins: int
    win_rate: float
    failure_rate: float
    avg_tokens: float
    avg_cost_usd: float
    avg_latency_seconds: float
    score_per_1k_tokens: Optional[float]    # efficiency: quality per token spent
    avg_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Recommendation:
    skip_benchmark: bool
    best_model: Optional[str]               # best *proven* candidate (None if no history)
    baseline_score: Optional[float]         # that model's historical weighted score
    lead_margin: Optional[float]            # lead over the runner-up (None if no runner-up)
    needs_benchmark: list                   # models that still have to be benchmarked
    ranking: list                           # ModelPerformance for every candidate with history
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ranking"] = [p.to_dict() if isinstance(p, ModelPerformance) else p for p in self.ranking]
        return d


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #

class BenchmarkStore:
    def __init__(
        self,
        db_path: Union[str, Path, None] = None,
        settings: Optional[OrchestratorSettings] = None,
        clock: Callable[[], float] = time.time,
        half_life_days: float = HALF_LIFE_DAYS,
    ):
        self._settings = settings or OrchestratorSettings()
        self._clock = clock
        self._half_life = half_life_days
        self._lock = threading.RLock()

        path = BENCHMARK_DB_PATH if db_path is None else db_path
        in_memory = str(path) == ":memory:"
        if not in_memory:
            ensure_dirs()
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if not in_memory:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    # ---- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "BenchmarkStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(rec: BenchmarkRecord) -> None:
        if not rec.model_id or not rec.task_type:
            raise ValueError("model_id and task_type are required")
        if not 0.0 <= rec.final_score <= 1.0:
            raise ValueError(f"final_score must be between 0 and 1, got {rec.final_score}")
        if rec.tokens_used < 0:
            raise ValueError("tokens_used cannot be negative")

    def _insert(self, rec: BenchmarkRecord, now: float) -> int:
        cur = self._conn.execute(
            """INSERT INTO benchmark_results
               (created_at, session_id, project_id, task_type, language, complexity, model_id,
                final_score, metrics, tokens_used, cost_usd, latency_seconds, success, is_winner, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                now, rec.session_id, rec.project_id, rec.task_type.lower(),
                rec.language.lower() if rec.language else None, rec.complexity, rec.model_id,
                rec.final_score, json.dumps(rec.metrics, default=str), int(rec.tokens_used),
                float(rec.cost_usd), float(rec.latency_seconds), int(rec.success),
                int(rec.is_winner), rec.notes,
            ),
        )
        return cur.lastrowid

    def record_result(self, rec: BenchmarkRecord) -> int:
        """Store one result. Returns its row id."""
        self._validate(rec)
        with self._lock, self._conn:
            row_id = self._insert(rec, self._clock())
        logger.debug("Recorded benchmark: %s on %s = %.3f", rec.model_id, rec.task_type, rec.final_score)
        return row_id

    def record_session(
        self,
        results: list,
        winner_model_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """
        Store all candidates of one benchmark run atomically, flag the winner and
        return the session id.
        """
        if not results:
            raise ValueError("record_session needs at least one result")
        for r in results:
            self._validate(r)
        if winner_model_id and winner_model_id not in {r.model_id for r in results}:
            raise ValueError(f"Winner {winner_model_id!r} is not among the results")

        session_id = uuid.uuid4().hex[:12]
        now = self._clock()
        with self._lock, self._conn:        # one transaction: all rows or none
            for r in results:
                r.session_id = session_id
                if project_id:
                    r.project_id = project_id
                r.is_winner = winner_model_id is not None and r.model_id == winner_model_id
                self._insert(r, now)
        logger.info("Recorded benchmark session %s (%d candidates, winner=%s)",
                    session_id, len(results), winner_model_id)
        return session_id

    def mark_winner(self, session_id: str, model_id: str) -> bool:
        """Set (or change) a session's winner. Returns False if the session/model doesn't exist."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE benchmark_results SET is_winner = CASE WHEN model_id = ? THEN 1 ELSE 0 END "
                "WHERE session_id = ?", (model_id, session_id))
            found = self._conn.execute(
                "SELECT COUNT(*) FROM benchmark_results WHERE session_id = ? AND is_winner = 1",
                (session_id,)).fetchone()[0]
        return found > 0

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    @staticmethod
    def _where(task_type, language, model_id, complexity, tolerance) -> tuple:
        clauses, params = [], []
        if task_type:
            clauses.append("task_type = ?")
            params.append(task_type.lower())
        if language:
            clauses.append("language = ?")
            params.append(language.lower())
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)
        if complexity is not None:
            tol = DEFAULT_COMPLEXITY_TOLERANCE if tolerance is None else tolerance
            clauses.append("complexity BETWEEN ? AND ?")
            params += [complexity - tol, complexity + tol]
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _to_record(row: sqlite3.Row) -> BenchmarkRecord:
        try:
            metrics = json.loads(row["metrics"] or "{}")
        except ValueError:
            metrics = {}
        return BenchmarkRecord(
            id=row["id"], created_at=row["created_at"], session_id=row["session_id"],
            project_id=row["project_id"], task_type=row["task_type"], language=row["language"],
            complexity=row["complexity"], model_id=row["model_id"], final_score=row["final_score"],
            metrics=metrics, tokens_used=row["tokens_used"], cost_usd=row["cost_usd"],
            latency_seconds=row["latency_seconds"], success=bool(row["success"]),
            is_winner=bool(row["is_winner"]), notes=row["notes"],
        )

    def get_results(
        self,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        model_id: Optional[str] = None,
        complexity: Optional[float] = None,
        complexity_tolerance: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list:
        """Raw results, newest first."""
        where, params = self._where(task_type, language, model_id, complexity, complexity_tolerance)
        sql = f"SELECT * FROM benchmark_results{where} ORDER BY created_at DESC, id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._to_record(r) for r in rows]

    def count(self, task_type: Optional[str] = None, language: Optional[str] = None,
              model_id: Optional[str] = None) -> int:
        where, params = self._where(task_type, language, model_id, None, None)
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM benchmark_results{where}", params).fetchone()[0]

    def get_performance(
        self,
        task_type: Optional[str] = None,
        language: Optional[str] = None,
        model_id: Optional[str] = None,
        complexity: Optional[float] = None,
        complexity_tolerance: Optional[float] = None,
    ) -> list:
        """Per-model aggregates for matching tasks, best (recency-weighted) first."""
        records = self.get_results(task_type, language, model_id, complexity, complexity_tolerance)
        groups: dict = {}
        for r in records:
            groups.setdefault(r.model_id, []).append(r)

        now = self._clock()
        out = []
        for mid, recs in groups.items():
            scores = [r.final_score for r in recs]
            weights = [0.5 ** (max(0.0, now - r.created_at) / SECONDS_PER_DAY / self._half_life) for r in recs]
            total_w = sum(weights)
            avg = sum(scores) / len(scores)
            weighted = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else avg

            wins = sum(1 for r in recs if r.is_winner)
            failures = sum(1 for r in recs if not r.success)
            avg_tokens = sum(r.tokens_used for r in recs) / len(recs)

            metric_values: dict = {}
            for r in recs:
                for key, value in r.metrics.items():
                    if isinstance(value, (int, float)):
                        metric_values.setdefault(key, []).append(float(value))

            out.append(ModelPerformance(
                model_id=mid,
                runs=len(recs),
                avg_score=round(avg, 4),
                weighted_avg_score=round(weighted, 4),
                std_dev=round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
                wins=wins,
                win_rate=round(wins / len(recs), 4),
                failure_rate=round(failures / len(recs), 4),
                avg_tokens=round(avg_tokens, 1),
                avg_cost_usd=round(sum(r.cost_usd for r in recs) / len(recs), 6),
                avg_latency_seconds=round(sum(r.latency_seconds for r in recs) / len(recs), 3),
                score_per_1k_tokens=round(avg / (avg_tokens / 1000), 4) if avg_tokens > 0 else None,
                avg_metrics={k: round(sum(v) / len(v), 4) for k, v in metric_values.items()},
            ))
        return sorted(out, key=lambda p: (p.weighted_avg_score, p.runs), reverse=True)

    # ------------------------------------------------------------------ #
    # The decision: benchmark or not?
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        task_type: str,
        candidates: list,
        language: Optional[str] = None,
        complexity: Optional[float] = None,
    ) -> Recommendation:
        """
        Decide which candidates (if any) still need benchmarking for this kind of task.

        A candidate is 'proven' once it has at least min_history_to_skip_benchmark similar
        past runs. Skipping requires every candidate to be proven AND the leader to be ahead
        of the runner-up by history_confidence_margin.
        """
        if not candidates:
            raise ValueError("recommend() needs at least one candidate")
        candidates = list(dict.fromkeys(candidates))        # de-duplicate, keep order

        min_runs = self._settings.min_history_to_skip_benchmark
        margin = self._settings.history_confidence_margin

        perf = {p.model_id: p for p in self.get_performance(task_type, language, complexity=complexity)}
        with_history = sorted((perf[c] for c in candidates if c in perf),
                              key=lambda p: p.weighted_avg_score, reverse=True)
        proven = [p for p in with_history if p.runs >= min_runs]
        proven_ids = {p.model_id for p in proven}
        unproven = [c for c in candidates if c not in proven_ids]

        # Case 1: no usable history at all
        if not proven:
            return Recommendation(
                skip_benchmark=False, best_model=None, baseline_score=None, lead_margin=None,
                needs_benchmark=unproven, ranking=with_history,
                reason=f"No candidate has {min_runs}+ similar past runs; benchmark all {len(unproven)}.",
            )

        best = proven[0]
        lead = round(best.weighted_avg_score - proven[1].weighted_avg_score, 4) if len(proven) > 1 else None
        contenders = [p for p in proven if best.weighted_avg_score - p.weighted_avg_score < margin]

        # Case 2: the top proven models are too close to call
        if len(contenders) > 1:
            needs = [p.model_id for p in contenders] + unproven
            return Recommendation(
                skip_benchmark=False, best_model=best.model_id, baseline_score=best.weighted_avg_score,
                lead_margin=lead, needs_benchmark=needs, ranking=with_history,
                reason=(f"Top models are within {margin:.2f} of each other "
                        f"({', '.join(p.model_id for p in contenders)}); benchmark the close contenders"
                        + (f" and the unproven: {', '.join(unproven)}." if unproven else ".")),
            )

        # Case 3: clear leader, but some candidates have too little history
        if unproven:
            return Recommendation(
                skip_benchmark=False, best_model=best.model_id, baseline_score=best.weighted_avg_score,
                lead_margin=lead, needs_benchmark=unproven, ranking=with_history,
                reason=(f"{best.model_id} leads on history ({best.weighted_avg_score:.2f}); benchmark only "
                        f"the unproven ({', '.join(unproven)}) and compare against that baseline."),
            )

        # Case 4: clear leader, everyone proven -> skip
        lead_text = f"leads by {lead:.2f}" if lead is not None else "is the only candidate"
        return Recommendation(
            skip_benchmark=True, best_model=best.model_id, baseline_score=best.weighted_avg_score,
            lead_margin=lead, needs_benchmark=[], ranking=with_history,
            reason=(f"{best.model_id} {lead_text} over {best.runs} similar runs "
                    f"(score {best.weighted_avg_score:.2f}); benchmark skipped."),
        )


# --------------------------------------------------------------------------- #
# Self-test: python -m core.benchmark_store
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile

    from core.logger import setup_logging

    setup_logging("WARNING")

    class FakeClock:
        def __init__(self, t: float = 1_000_000.0):
            self.t = t

        def __call__(self) -> float:
            return self.t

    TASK = "backend_api"

    def rec(model, score, **kw):
        defaults = dict(task_type=TASK, language="python", complexity=6.0, tokens_used=1000,
                        cost_usd=0.01, latency_seconds=2.0)
        defaults.update(kw)
        return BenchmarkRecord(model_id=model, final_score=score, **defaults)

    settings = OrchestratorSettings(min_history_to_skip_benchmark=5, history_confidence_margin=0.05)

    def new_store():
        clock = FakeClock()
        return BenchmarkStore(":memory:", settings=settings, clock=clock), clock

    def fill(store, model, score, n, **kw):
        for _ in range(n):
            store.record_result(rec(model, score, **kw))

    # 1. Basic record / read / validation
    store, clock = new_store()
    rid = store.record_result(rec("gpt", 0.8, metrics={"correctness": 0.9, "security": 0.7}))
    got = store.get_results(model_id="gpt")[0]
    assert rid == got.id and got.metrics["security"] == 0.7 and got.language == "python"
    for bad in (rec("gpt", 1.5), rec("", 0.5)):
        try:
            store.record_result(bad)
            raise SystemExit("FAIL 1: bad record accepted")
        except ValueError:
            pass
    print("PASS 1: record, read, validation")

    # 2. Sessions and winners
    store, clock = new_store()
    sid = store.record_session([rec("a", 0.9), rec("b", 0.7)], winner_model_id="a", project_id="proj1")
    rows = {r.model_id: r for r in store.get_results()}
    assert rows["a"].is_winner and not rows["b"].is_winner and rows["a"].project_id == "proj1"
    assert store.mark_winner(sid, "b")
    rows = {r.model_id: r for r in store.get_results()}
    assert rows["b"].is_winner and not rows["a"].is_winner
    assert not store.mark_winner("nope", "a")
    try:
        store.record_session([rec("a", 0.9)], winner_model_id="zzz")
        raise SystemExit("FAIL 2")
    except ValueError:
        pass
    print("PASS 2: sessions, winner flag, mark_winner")

    # 3. Aggregation, failures, efficiency
    store, clock = new_store()
    for s in (0.8, 0.9, 1.0):
        store.record_result(rec("a", s, metrics={"correctness": s}))
    store.record_result(rec("b", 0.5))
    store.record_result(rec("b", 0.0, success=False))
    perf = {p.model_id: p for p in store.get_performance(TASK)}
    assert perf["a"].runs == 3 and abs(perf["a"].avg_score - 0.9) < 1e-9
    assert abs(perf["a"].avg_metrics["correctness"] - 0.9) < 1e-9
    assert perf["b"].failure_rate == 0.5 and perf["a"].score_per_1k_tokens == 0.9
    assert store.get_performance(TASK)[0].model_id == "a"
    print("PASS 3: aggregation (avg, failure rate, efficiency, ranking)")

    # 4. Recency weighting
    store, clock = new_store()
    fill(store, "a", 0.9, 5)
    clock.t += 90 * SECONDS_PER_DAY
    fill(store, "a", 0.5, 5)
    p = store.get_performance(TASK)[0]
    assert abs(p.avg_score - 0.7) < 1e-9 and p.weighted_avg_score < p.avg_score - 0.1
    print(f"PASS 4: recency weighting (plain {p.avg_score}, weighted {p.weighted_avg_score})")

    # 5. Recommend: clear leader, everyone proven -> skip
    store, clock = new_store()
    fill(store, "a", 0.9, 6)
    fill(store, "b", 0.7, 6)
    r = store.recommend(TASK, ["a", "b"], language="python")
    assert r.skip_benchmark and r.best_model == "a" and r.needs_benchmark == []
    print("PASS 5: skip benchmark ->", r.reason)

    # 6. Recommend: clear leader, one unproven candidate -> benchmark only that one
    fill(store, "c", 0.95, 2)
    r = store.recommend(TASK, ["a", "b", "c"], language="python")
    assert not r.skip_benchmark and r.needs_benchmark == ["c"] and r.baseline_score is not None
    print("PASS 6: only the unproven model needs benchmarking ->", r.needs_benchmark)

    # 7. Recommend: top models too close -> benchmark the contenders
    store, clock = new_store()
    fill(store, "a", 0.90, 6)
    fill(store, "b", 0.88, 6)
    fill(store, "c", 0.50, 6)
    r = store.recommend(TASK, ["a", "b", "c"], language="python")
    assert not r.skip_benchmark and set(r.needs_benchmark) == {"a", "b"}
    print("PASS 7: close contenders ->", r.needs_benchmark)

    # 8. Recommend: no history
    store, clock = new_store()
    r = store.recommend(TASK, ["a", "b", "a"])
    assert not r.skip_benchmark and r.best_model is None and r.needs_benchmark == ["a", "b"]
    print("PASS 8: no history -> benchmark everyone")

    # 9. Filters: language (case-insensitive) and complexity band
    store, clock = new_store()
    store.record_result(rec("a", 0.9))
    store.record_result(rec("a", 0.9, language="javascript"))
    store.record_result(rec("a", 0.9, complexity=9.0))
    assert len(store.get_results(task_type=TASK, language="Python")) == 2
    assert len(store.get_results(task_type=TASK, language="python", complexity=6.0)) == 1
    assert store.count(task_type=TASK) == 3
    print("PASS 9: language and complexity filters")

    # 10. Persistence across reopen
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.db"
        s1 = BenchmarkStore(path, settings=settings)
        s1.record_result(rec("a", 0.9))
        s1.close()
        s2 = BenchmarkStore(path, settings=settings)
        assert s2.count() == 1
        s2.close()
    print("PASS 10: data survives a restart")