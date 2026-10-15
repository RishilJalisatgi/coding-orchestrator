"""
models/base.py

Abstract model adapter + shared types.

Concrete adapters (claude.py, openai.py, gemini.py, local.py) subclass ModelAdapter
and implement ONE method: _call(request) -> RawResult.

Everything else (retries, backoff, timeouts, cost calculation, usage reporting,
error classification) lives here, so it is written and tested once.

Error contract for adapters (raise these from _call):
    RateLimitError        -> provider said "slow down" (retried, honours retry_after)
    TransientModelError   -> timeouts, 5xx, connection resets (retried)
    FatalModelError       -> bad key, bad request, model not found (NOT retried)
Any other exception is treated as transient and retried.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from config.settings import ModelConfig, OrchestratorSettings
from core.logger import get_logger

logger = get_logger("models.base")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class ModelError(Exception):
    """Base class for all model failures. `attempts` is filled in by the adapter."""

    def __init__(self, message: str, model_id: Optional[str] = None, attempts: int = 0):
        super().__init__(message)
        self.model_id = model_id
        self.attempts = attempts


class TransientModelError(ModelError):
    """Temporary failure: worth retrying."""


class RateLimitError(TransientModelError):
    """Provider rate limit hit. Optionally carries a retry-after hint (seconds)."""

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class FatalModelError(ModelError):
    """Permanent failure: retrying will not help."""


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass
class ModelRequest:
    prompt: str
    system: Optional[str] = None
    max_tokens: Optional[int] = None      # capped to the model's max_output_tokens
    temperature: float = 0.2
    purpose: str = "general"              # e.g. "lite_benchmark", "hld", "coding", "review"
    metadata: dict = field(default_factory=dict)


@dataclass
class RawResult:
    """What a concrete adapter's _call() returns."""
    text: str
    input_tokens: Optional[int] = None    # None -> estimated by the base class
    output_tokens: Optional[int] = None
    raw: Any = None                       # original provider response, for debugging


@dataclass
class ModelResponse:
    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float                # total, including retries/backoff
    attempts: int
    purpose: str = "general"
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageEvent:
    """Emitted after every generate() call (success or failure). Consumed by the resource manager."""
    model_id: str
    purpose: str
    success: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    attempts: int
    error: Optional[str] = None
    rate_limited: bool = False            # True if any attempt hit a rate limit

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


UsageListener = Callable[[UsageEvent], None]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 chars/token). Used when a provider doesn't report usage."""
    return max(1, len(text) // 4) if text else 0


# --------------------------------------------------------------------------- #
# Base adapter
# --------------------------------------------------------------------------- #

class ModelAdapter(ABC):
    def __init__(
        self,
        config: ModelConfig,
        settings: OrchestratorSettings,
        listeners: Optional[list[UsageListener]] = None,
    ):
        self.config = config
        self.settings = settings
        self.listeners: list[UsageListener] = list(listeners or [])
        self._sleep: Callable[[float], None] = time.sleep   # injectable for tests

    # ---- public properties ------------------------------------------------- #

    @property
    def model_id(self) -> str:
        return self.config.id

    @property
    def timeout(self) -> int:
        """Per-request timeout in seconds. Concrete adapters should pass this to their SDK/client."""
        return self.settings.request_timeout_seconds

    # ---- to implement in subclasses ---------------------------------------- #

    @abstractmethod
    def _call(self, request: ModelRequest) -> RawResult:
        """Make ONE raw API call. Raise RateLimitError / TransientModelError / FatalModelError."""

    # ---- listeners --------------------------------------------------------- #

    def add_listener(self, listener: UsageListener) -> None:
        self.listeners.append(listener)

    def _emit(self, event: UsageEvent) -> None:
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:   # a broken listener must never break a model call
                logger.exception("Usage listener failed for %s", self.model_id)

    # ---- cost -------------------------------------------------------------- #

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000 * self.config.cost_per_1k_input
            + output_tokens / 1000 * self.config.cost_per_1k_output
        )

    # ---- backoff ----------------------------------------------------------- #

    def _backoff_delay(self, attempt: int, error: ModelError) -> float:
        """Delay before retry number `attempt` (1-based). Exponential + jitter, capped."""
        s = self.settings
        delay = min(s.backoff_base_seconds * (2 ** (attempt - 1)), s.backoff_max_seconds)
        delay *= random.uniform(0.5, 1.0)   # jitter avoids synchronized retries
        if isinstance(error, RateLimitError) and error.retry_after:
            delay = max(delay, min(error.retry_after, s.backoff_max_seconds))
        return delay

    # ---- main entry point -------------------------------------------------- #

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Call the model with retries. Returns a ModelResponse or raises ModelError
        (after retries are exhausted, or immediately for fatal errors).
        """
        # Cap max_tokens to what the model allows
        limit = self.config.max_output_tokens
        request.max_tokens = min(request.max_tokens or limit, limit)

        max_attempts = self.settings.max_retries + 1
        started = time.monotonic()
        last_error: Optional[ModelError] = None
        rate_limited = False
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            try:
                raw = self._call(request)
                return self._build_success(request, raw, attempt, started, rate_limited)

            except FatalModelError as e:
                last_error = e
                logger.error("[%s] fatal error (no retry): %s", self.model_id, e)
                break

            except RateLimitError as e:
                rate_limited = True
                last_error = e
                logger.warning("[%s] rate limited (attempt %d/%d): %s",
                               self.model_id, attempt, max_attempts, e)

            except TransientModelError as e:
                last_error = e
                logger.warning("[%s] transient error (attempt %d/%d): %s",
                               self.model_id, attempt, max_attempts, e)

            except ModelError as e:   # unclassified ModelError subclass: treat as transient
                last_error = e
                logger.warning("[%s] model error (attempt %d/%d): %s",
                               self.model_id, attempt, max_attempts, e)

            except Exception as e:    # unmapped SDK exception: treat as transient
                last_error = TransientModelError(f"{type(e).__name__}: {e}", model_id=self.model_id)
                logger.warning("[%s] unexpected error (attempt %d/%d): %s: %s",
                               self.model_id, attempt, max_attempts, type(e).__name__, e)

            if attempt < max_attempts:
                delay = self._backoff_delay(attempt, last_error)
                logger.info("[%s] retrying in %.1fs", self.model_id, delay)
                self._sleep(delay)

        return self._raise_failure(request, last_error, attempt, started, rate_limited)

    def try_generate(self, request: ModelRequest) -> Optional[ModelResponse]:
        """Like generate(), but returns None on failure instead of raising."""
        try:
            return self.generate(request)
        except ModelError:
            return None

    # ---- internals --------------------------------------------------------- #

    def _build_success(
        self, request: ModelRequest, raw: RawResult, attempts: int, started: float, rate_limited: bool
    ) -> ModelResponse:
        input_tokens = raw.input_tokens
        if input_tokens is None:
            input_tokens = estimate_tokens((request.system or "") + request.prompt)
        output_tokens = raw.output_tokens
        if output_tokens is None:
            output_tokens = estimate_tokens(raw.text)

        cost = self.calculate_cost(input_tokens, output_tokens)
        latency = time.monotonic() - started

        response = ModelResponse(
            text=raw.text,
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_seconds=latency,
            attempts=attempts,
            purpose=request.purpose,
            raw=raw.raw,
        )
        self._emit(UsageEvent(
            model_id=self.model_id, purpose=request.purpose, success=True,
            input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost,
            latency_seconds=latency, attempts=attempts, rate_limited=rate_limited,
        ))
        return response

    def _raise_failure(
        self, request: ModelRequest, error: Optional[ModelError], attempts: int,
        started: float, rate_limited: bool,
    ) -> ModelResponse:
        error = error or ModelError("Unknown failure", model_id=self.model_id)
        error.model_id = self.model_id
        error.attempts = attempts

        # Failed calls may still have consumed quota; we can't know, so record zero tokens
        # but keep the failure for reliability stats.
        self._emit(UsageEvent(
            model_id=self.model_id, purpose=request.purpose, success=False,
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            latency_seconds=time.monotonic() - started, attempts=attempts,
            error=str(error), rate_limited=rate_limited,
        ))
        raise error


# --------------------------------------------------------------------------- #
# Self-test: python -m models.base
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from core.logger import setup_logging

    setup_logging("INFO")

    class FakeAdapter(ModelAdapter):
        """Plays back a scripted list of outcomes: an Exception to raise, or a str to return."""

        def __init__(self, *args, script, **kwargs):
            super().__init__(*args, **kwargs)
            self.script = list(script)

        def _call(self, request):
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return RawResult(text=outcome, input_tokens=100, output_tokens=50)

    cfg = ModelConfig(id="fake", provider="local", model_name="fake",
                      cost_per_1k_input=0.01, cost_per_1k_output=0.02)
    settings = OrchestratorSettings(max_retries=3)
    events: list[UsageEvent] = []

    def make(script):
        a = FakeAdapter(cfg, settings, script=script, listeners=[events.append])
        a._sleep = lambda s: None   # skip real waiting in the test
        return a

    # 1. Succeeds after two transient failures
    r = make([TransientModelError("timeout"), RateLimitError("429", retry_after=1), "hello"]).generate(
        ModelRequest(prompt="hi"))
    assert r.text == "hello" and r.attempts == 3 and abs(r.cost_usd - 0.002) < 1e-9
    print("PASS 1: retries then succeeds  ->", r.attempts, "attempts, cost", r.cost_usd)

    # 2. Fatal error: no retries
    try:
        make([FatalModelError("bad key"), "never reached"]).generate(ModelRequest(prompt="hi"))
        raise SystemExit("FAIL 2: expected an error")
    except FatalModelError as e:
        assert e.attempts == 1
        print("PASS 2: fatal error stops immediately")

    # 3. Exhausted retries
    r = make([TransientModelError("x")] * 4).try_generate(ModelRequest(prompt="hi"))
    assert r is None
    print("PASS 3: exhausted retries -> try_generate returned None")

    # 4. Events were emitted for every call
    assert [e.success for e in events] == [True, False, False]
    assert events[0].rate_limited is True
    print("PASS 4: usage events emitted:", len(events))