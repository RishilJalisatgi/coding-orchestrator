"""
core/resource_manager.py

Tracks quota, cost, health and performance for every model.

Wiring (done by the orchestrator):
    rm = ResourceManager(settings.models, settings.orchestrator)
    adapter.add_listener(rm.record)          # every model call reports its usage here

The scheduler then asks questions like:
    rm.can_use("claude-sonnet", 6000)        # budget + health check
    rm.reserve("claude-sonnet", 8000, "final_review")   # earmark tokens for later
    rm.status("gpt")                         # full snapshot for decision logging

Quota model: ROLLING window. Each call's tokens count against the model until
`window_seconds` have passed since that call.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config.settings import STATE_DIR, ModelConfig, OrchestratorSettings, ensure_dirs
from core.logger import get_logger
from models.base import UsageEvent

logger = get_logger("core.resource_manager")

# Health policy
RATE_LIMIT_COOLDOWN_FAILED = 60.0      # seconds, when a call ended in failure after rate limiting
RATE_LIMIT_COOLDOWN_RECOVERED = 10.0   # seconds, when the call succeeded after rate limiting
CIRCUIT_BREAKER_FAILURES = 3           # consecutive failed calls before the model is paused
CIRCUIT_BREAKER_COOLDOWN = 120.0       # seconds the model is paused

# Reliability = (successes + PRIOR_SUCCESSES) / (calls + PRIOR_CALLS)
# A new model starts at 0.9 and moves toward its real success rate as data accumulates.
PRIOR_SUCCESSES = 1.8
PRIOR_CALLS = 2.0


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass
class ModelStats:
    """Mutable per-model state. Plain types only, so it serialises to JSON directly."""
    calls: int = 0                       # generate() calls
    successes: int = 0
    failures: int = 0
    attempts: int = 0                    # raw API attempts (includes retries)
    rate_limit_hits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency: float = 0.0           # sum over successful calls
    consecutive_failures: int = 0
    cooldown_until: float = 0.0          # wall-clock timestamp
    window_events: list = field(default_factory=list)    # [[timestamp, tokens], ...]
    recent_requests: list = field(default_factory=list)  # [timestamp, ...] for the RPM check


@dataclass
class ModelStatus:
    """Read-only snapshot, convenient for the scheduler and for decision logs."""
    model_id: str
    available: bool
    reason: str
    remaining_tokens: Optional[int]        # None = unlimited
    usable_tokens: Optional[int]           # after safety margin and reservations
    reserved_tokens: int
    seconds_until_reset: Optional[float]   # until the oldest usage expires (None = nothing in window)
    cooldown_remaining: float
    requests_last_minute: int
    reliability: float
    avg_latency_seconds: Optional[float]
    total_cost_usd: float
    calls: int

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Resource manager
# --------------------------------------------------------------------------- #

class ResourceManager:
    def __init__(
        self,
        models: list[ModelConfig],
        settings: OrchestratorSettings,
        state_path: Optional[Path] = None,
        persist: bool = True,
        clock: Callable[[], float] = time.time,
    ):
        self._models: dict[str, ModelConfig] = {m.id: m for m in models}
        self._settings = settings
        self._clock = clock
        self._persist = persist
        self._state_path = Path(state_path) if state_path else STATE_DIR / "resource_state.json"
        self._lock = threading.RLock()

        self._stats: dict[str, ModelStats] = {mid: ModelStats() for mid in self._models}
        self._reservations: dict[str, dict[str, int]] = {mid: {} for mid in self._models}

        if self._persist:
            self._load()

    # ------------------------------------------------------------------ #
    # Recording usage
    # ------------------------------------------------------------------ #

    def record(self, event: UsageEvent) -> None:
        """Usage listener: pass this to adapter.add_listener()."""
        with self._lock:
            st = self._stats.get(event.model_id)
            if st is None:
                logger.warning("Usage event for unknown model %r ignored", event.model_id)
                return

            now = self._clock()
            st.calls += 1
            st.attempts += event.attempts
            st.recent_requests.extend([now] * max(1, min(event.attempts, 50)))

            if event.success:
                st.successes += 1
                st.consecutive_failures = 0
                st.total_latency += event.latency_seconds
            else:
                st.failures += 1
                st.consecutive_failures += 1

            st.total_input_tokens += event.input_tokens
            st.total_output_tokens += event.output_tokens
            st.total_cost_usd += event.cost_usd
            if event.total_tokens > 0:
                st.window_events.append([now, event.total_tokens])

            if event.rate_limited:
                st.rate_limit_hits += 1
                cooldown = RATE_LIMIT_COOLDOWN_RECOVERED if event.success else RATE_LIMIT_COOLDOWN_FAILED
                self._extend_cooldown(event.model_id, st, now, cooldown, "rate limited")

            if st.consecutive_failures >= CIRCUIT_BREAKER_FAILURES:
                self._extend_cooldown(event.model_id, st, now, CIRCUIT_BREAKER_COOLDOWN,
                                      f"{st.consecutive_failures} consecutive failures")

            self._prune(event.model_id, now)
            self._save()

    def _extend_cooldown(self, model_id: str, st: ModelStats, now: float, seconds: float, why: str) -> None:
        until = now + seconds
        if until > st.cooldown_until:
            st.cooldown_until = until
            logger.warning("[%s] paused for %.0fs (%s)", model_id, seconds, why)

    # ------------------------------------------------------------------ #
    # Quota queries
    # ------------------------------------------------------------------ #

    def used_tokens(self, model_id: str) -> int:
        with self._lock:
            self._prune(model_id, self._clock())
            return sum(int(e[1]) for e in self._stats[model_id].window_events)

    def remaining_tokens(self, model_id: str) -> Optional[int]:
        """Tokens left in the current window. None means unlimited."""
        budget = self._models[model_id].token_budget_per_window
        if budget is None:
            return None
        return max(0, budget - self.used_tokens(model_id))

    def reserved_tokens(self, model_id: str, ignore_reservation: Optional[str] = None) -> int:
        with self._lock:
            return sum(t for label, t in self._reservations[model_id].items()
                       if label != ignore_reservation)

    def usable_tokens(self, model_id: str, ignore_reservation: Optional[str] = None) -> Optional[int]:
        """Tokens the scheduler may plan to spend: budget minus safety margin, usage and reservations."""
        budget = self._models[model_id].token_budget_per_window
        if budget is None:
            return None
        ceiling = int(budget * (1 - self._settings.quota_safety_margin))
        return max(0, ceiling - self.used_tokens(model_id)
                   - self.reserved_tokens(model_id, ignore_reservation))

    def can_afford(self, model_id: str, tokens: int, ignore_reservation: Optional[str] = None) -> bool:
        """Budget check only (ignores health). Pass ignore_reservation to spend your own reserved tokens."""
        usable = self.usable_tokens(model_id, ignore_reservation)
        return True if usable is None else tokens <= usable

    def can_use(self, model_id: str, tokens: int, ignore_reservation: Optional[str] = None) -> bool:
        """Budget AND health check: the one the scheduler should normally call."""
        return self.status(model_id).available and self.can_afford(model_id, tokens, ignore_reservation)

    def seconds_until_reset(self, model_id: str) -> Optional[float]:
        """Seconds until the oldest in-window usage expires and capacity starts to free up."""
        with self._lock:
            now = self._clock()
            self._prune(model_id, now)
            events = self._stats[model_id].window_events
            if not events:
                return None
            oldest = min(e[0] for e in events)
            return max(0.0, oldest + self._models[model_id].window_seconds - now)

    # ------------------------------------------------------------------ #
    # Reservations
    # ------------------------------------------------------------------ #

    def reserve(self, model_id: str, tokens: int, label: str) -> bool:
        """
        Earmark tokens for a future stage (e.g. 'final_review'). Returns False if they
        can't be afforded. Re-reserving the same label replaces the old amount.
        """
        with self._lock:
            if not self.can_afford(model_id, tokens, ignore_reservation=label):
                logger.info("[%s] cannot reserve %d tokens for %r", model_id, tokens, label)
                return False
            self._reservations[model_id][label] = tokens
            logger.info("[%s] reserved %d tokens for %r", model_id, tokens, label)
            return True

    def release(self, label: str, model_id: Optional[str] = None) -> None:
        """Release a reservation (call it when the stage has run, or the plan changed)."""
        with self._lock:
            targets = [model_id] if model_id else list(self._reservations)
            for mid in targets:
                self._reservations[mid].pop(label, None)

    # ------------------------------------------------------------------ #
    # Manual overrides
    # ------------------------------------------------------------------ #

    def set_remaining(self, model_id: str, remaining: int, resets_in_seconds: Optional[float] = None) -> None:
        """
        Override quota with real-world knowledge, e.g. 'Claude has 14k tokens left, resets in 31 min'.
        Replaces tracked usage with one synthetic entry that expires at the given time.
        """
        cfg = self._models[model_id]
        if cfg.token_budget_per_window is None:
            logger.info("[%s] is unlimited; set_remaining ignored", model_id)
            return
        with self._lock:
            now = self._clock()
            used = max(0, cfg.token_budget_per_window - max(0, remaining))
            reset_in = cfg.window_seconds if resets_in_seconds is None else min(resets_in_seconds, cfg.window_seconds)
            st = self._stats[model_id]
            st.window_events = [[now + reset_in - cfg.window_seconds, used]] if used > 0 else []
            logger.info("[%s] quota manually set: %d remaining, resets in %.0fs", model_id, remaining, reset_in)
            self._save()

    def set_cooldown(self, model_id: str, seconds: float) -> None:
        """Manually pause a model (e.g. you know it is down)."""
        with self._lock:
            self._stats[model_id].cooldown_until = self._clock() + seconds
            self._save()

    # ------------------------------------------------------------------ #
    # Health and performance
    # ------------------------------------------------------------------ #

    def reliability(self, model_id: str) -> float:
        st = self._stats[model_id]
        return (st.successes + PRIOR_SUCCESSES) / (st.calls + PRIOR_CALLS)

    def avg_latency(self, model_id: str) -> Optional[float]:
        st = self._stats[model_id]
        return st.total_latency / st.successes if st.successes else None

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        cfg = self._models[model_id]
        return (input_tokens / 1000 * cfg.cost_per_1k_input
                + output_tokens / 1000 * cfg.cost_per_1k_output)

    def status(self, model_id: str) -> ModelStatus:
        with self._lock:
            now = self._clock()
            self._prune(model_id, now)
            cfg, st = self._models[model_id], self._stats[model_id]

            cooldown_remaining = max(0.0, st.cooldown_until - now)
            rpm_used = len(st.recent_requests)

            available, reason = True, "ok"
            if not cfg.enabled:
                available, reason = False, "disabled"
            elif cooldown_remaining > 0:
                available, reason = False, f"cooling down for {cooldown_remaining:.0f}s"
            elif cfg.requests_per_minute is not None and rpm_used >= cfg.requests_per_minute:
                available, reason = False, f"requests/min limit reached ({rpm_used}/{cfg.requests_per_minute})"
            elif self.remaining_tokens(model_id) == 0:
                available, reason = False, "token quota exhausted"

            return ModelStatus(
                model_id=model_id,
                available=available,
                reason=reason,
                remaining_tokens=self.remaining_tokens(model_id),
                usable_tokens=self.usable_tokens(model_id),
                reserved_tokens=self.reserved_tokens(model_id),
                seconds_until_reset=self.seconds_until_reset(model_id),
                cooldown_remaining=cooldown_remaining,
                requests_last_minute=rpm_used,
                reliability=round(self.reliability(model_id), 4),
                avg_latency_seconds=self.avg_latency(model_id),
                total_cost_usd=round(st.total_cost_usd, 6),
                calls=st.calls,
            )

    def snapshot(self) -> dict[str, dict]:
        """All model statuses as plain dicts, ready for decision logging."""
        return {mid: self.status(mid).to_dict() for mid in self._models}

    # ------------------------------------------------------------------ #
    # Internals: pruning and persistence
    # ------------------------------------------------------------------ #

    def _prune(self, model_id: str, now: float) -> None:
        st, cfg = self._stats[model_id], self._models[model_id]
        window_cutoff = now - cfg.window_seconds
        st.window_events = [e for e in st.window_events if e[0] > window_cutoff]
        rpm_cutoff = now - 60
        st.recent_requests = [t for t in st.recent_requests if t > rpm_cutoff]

    def _save(self) -> None:
        if not self._persist:
            return
        try:
            ensure_dirs()
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"saved_at": self._clock(),
                       "models": {mid: asdict(st) for mid, st in self._stats.items()}}
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self._state_path)     # atomic: never leaves a half-written file
        except OSError:
            logger.exception("Could not save resource state")   # never crash a run over bookkeeping

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            now = self._clock()
            for mid, raw in data.get("models", {}).items():
                if mid in self._stats:
                    self._stats[mid] = ModelStats(**raw)
                    self._prune(mid, now)
            logger.info("Resource state restored from %s", self._state_path)
        except (OSError, ValueError, TypeError):
            logger.exception("Resource state file unreadable; starting fresh")


# --------------------------------------------------------------------------- #
# Self-test: python -m core.resource_manager
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

    def ev(model_id, tokens=0, success=True, rate_limited=False, attempts=1, latency=1.0):
        return UsageEvent(model_id=model_id, purpose="test", success=success,
                          input_tokens=tokens * 3 // 4, output_tokens=tokens - tokens * 3 // 4,
                          cost_usd=0.0, latency_seconds=latency, attempts=attempts,
                          rate_limited=rate_limited)

    models = [
        ModelConfig(id="a", provider="x", model_name="a", token_budget_per_window=1000, window_seconds=100),
        ModelConfig(id="b", provider="x", model_name="b", token_budget_per_window=None, requests_per_minute=3),
        ModelConfig(id="c", provider="x", model_name="c", token_budget_per_window=None),
    ]
    cfg = OrchestratorSettings(quota_safety_margin=0.1)

    def fresh(persist=False, path=None):
        clock = FakeClock()
        return ResourceManager(models, cfg, state_path=path, persist=persist, clock=clock), clock

    # 1. Window accounting and safety margin
    rm, clock = fresh()
    rm.record(ev("a", 400))
    assert rm.remaining_tokens("a") == 600
    assert rm.usable_tokens("a") == 500            # 900 ceiling - 400 used
    assert rm.can_afford("a", 500) and not rm.can_afford("a", 501)
    print("PASS 1: window accounting + safety margin")

    # 2. Reservations
    assert rm.reserve("a", 300, "review")
    assert rm.usable_tokens("a") == 200
    assert not rm.can_afford("a", 300)
    assert rm.can_afford("a", 300, ignore_reservation="review")
    assert not rm.reserve("a", 300, "other")       # only 200 left
    rm.release("review")
    assert rm.usable_tokens("a") == 500
    print("PASS 2: reservations")

    # 3. Rolling window expiry
    clock.t += 101
    assert rm.remaining_tokens("a") == 1000 and rm.seconds_until_reset("a") is None
    print("PASS 3: usage expires after the window")

    # 4. Manual override
    rm.set_remaining("a", 200, resets_in_seconds=50)
    assert rm.remaining_tokens("a") == 200
    assert abs(rm.seconds_until_reset("a") - 50) < 1e-6
    clock.t += 51
    assert rm.remaining_tokens("a") == 1000
    print("PASS 4: manual set_remaining with reset time")

    # 5. Requests-per-minute limit
    rm, clock = fresh()
    for _ in range(3):
        rm.record(ev("b", 10))
    s = rm.status("b")
    assert not s.available and "requests/min" in s.reason
    clock.t += 61
    assert rm.status("b").available
    print("PASS 5: RPM limit blocks, then clears")

    # 6. Rate-limit cooldown
    rm, clock = fresh()
    rm.record(ev("c", 0, success=False, rate_limited=True))
    assert not rm.status("c").available
    clock.t += 61
    assert rm.status("c").available
    print("PASS 6: rate-limit cooldown")

    # 7. Circuit breaker + reliability
    rm, clock = fresh()
    for _ in range(3):
        rm.record(ev("c", 0, success=False))
    assert not rm.status("c").available and rm.reliability("c") < 0.5
    clock.t += 121
    assert rm.status("c").available
    print("PASS 7: circuit breaker, reliability =", round(rm.reliability("c"), 2))

    # 8. Unlimited model
    assert rm.remaining_tokens("c") is None and rm.can_afford("c", 10**9)
    print("PASS 8: unlimited model")

    # 9. Persistence across restarts
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        rm1, clock1 = fresh(persist=True, path=path)
        rm1.record(ev("a", 400))
        rm2 = ResourceManager(models, cfg, state_path=path, persist=True, clock=clock1)
        assert rm2.remaining_tokens("a") == 600
    print("PASS 9: state survives a restart")

    print("\nSample snapshot for 'a':", rm.status("a").to_dict())