"""
models/claude.py

Adapter for Anthropic's Claude models (Messages API).

Requires:  pip install anthropic
API key:   read from the env var named in the model's config (ANTHROPIC_API_KEY by default),
           or passed explicitly via api_key=.

Retries/backoff/usage reporting come from ModelAdapter; this class only performs ONE call
(_call) and maps Anthropic exceptions onto the base error types.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from config.settings import ModelConfig, OrchestratorSettings
from core.logger import get_logger
from models.base import (
    FatalModelError,
    ModelAdapter,
    ModelRequest,
    RateLimitError,
    RawResult,
    TransientModelError,
    UsageListener,
)

logger = get_logger("models.claude")

# HTTP statuses that will never succeed on retry (bad key, bad request, unknown model, too large...)
FATAL_STATUSES = {400, 401, 402, 403, 404, 413, 422}
# Statuses worth retrying (timeout, conflict, server errors, and Anthropic's 529 "overloaded")
TRANSIENT_STATUSES = {408, 409, 429}


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Read the retry-after header from an API error, if present."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_exception(exc: Exception, model_id: str) -> Exception:
    """
    Map an Anthropic SDK exception to RateLimitError / TransientModelError / FatalModelError.
    Uses class names and status codes (duck typing), so it needs no SDK import.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    msg = f"{name}: {exc}"

    if status == 429 or name == "RateLimitError":
        return RateLimitError(msg, retry_after=_retry_after_seconds(exc), model_id=model_id)

    # Timeouts and connection problems carry no status code
    if name in ("APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectionError"):
        return TransientModelError(msg, model_id=model_id)

    if isinstance(status, int):
        if status in FATAL_STATUSES or name in ("AuthenticationError", "PermissionDeniedError",
                                                "NotFoundError", "BadRequestError"):
            return FatalModelError(msg, model_id=model_id)
        if status >= 500 or status in TRANSIENT_STATUSES:
            return TransientModelError(msg, model_id=model_id)

    # Unknown error type: retrying is the safer default
    return TransientModelError(msg, model_id=model_id)


class ClaudeAdapter(ModelAdapter):
    def __init__(
        self,
        config: ModelConfig,
        settings: OrchestratorSettings,
        listeners: Optional[list[UsageListener]] = None,
        api_key: Optional[str] = None,
        client: Any = None,             # inject a ready-made/fake client (used by tests)
    ):
        super().__init__(config, settings, listeners)
        self._api_key = api_key
        self._client = client

    # ---- client ------------------------------------------------------------ #

    def _get_client(self) -> Any:
        """Create the SDK client on first use, so importing this module never needs the SDK."""
        if self._client is not None:
            return self._client

        key = self._api_key or (os.environ.get(self.config.api_key_env) if self.config.api_key_env else None)
        if not key:
            raise FatalModelError(
                f"No API key for {self.model_id}: set the {self.config.api_key_env} environment variable",
                model_id=self.model_id,
            )
        try:
            import anthropic
        except ImportError as e:
            raise FatalModelError(
                "The 'anthropic' package is not installed: run `pip install anthropic`",
                model_id=self.model_id,
            ) from e

        # max_retries=0: ModelAdapter.generate() owns retries, so each attempt is counted exactly once.
        self._client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=self.timeout)
        return self._client

    # ---- the one API call -------------------------------------------------- #

    def _call(self, request: ModelRequest) -> RawResult:
        client = self._get_client()

        params: dict = {
            "model": self.config.model_name,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "timeout": self.timeout,
        }
        if request.system:
            params["system"] = request.system
        if request.temperature is not None:
            params["temperature"] = request.temperature

        try:
            message = client.messages.create(**params)
        except Exception as e:
            raise classify_exception(e, self.model_id) from e

        text = "".join(
            block.text for block in (getattr(message, "content", None) or [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        )
        if not text.strip():
            stop = getattr(message, "stop_reason", None)
            raise TransientModelError(f"Empty response from model (stop_reason={stop})", model_id=self.model_id)

        if getattr(message, "stop_reason", None) == "max_tokens":
            logger.warning("[%s] response was cut off at max_tokens=%s (purpose=%s)",
                           self.model_id, request.max_tokens, request.purpose)

        usage = getattr(message, "usage", None)
        return RawResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            raw=message,
        )


# --------------------------------------------------------------------------- #
# Self-test (no network, no API key): python -m models.claude
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from types import SimpleNamespace

    from core.logger import setup_logging

    setup_logging("ERROR")

    # Fake exceptions named like the SDK's, so classification is tested without the SDK.
    class RateLimitError_(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "7"})
    RateLimitError_.__name__ = "RateLimitError"

    class AuthenticationError(Exception):
        status_code = 401

    class InternalServerError(Exception):
        status_code = 500

    class OverloadedError(Exception):
        status_code = 529

    class APITimeoutError(Exception):
        pass

    class BadRequestError(Exception):
        status_code = 400

    def message(text="hello", stop="end_turn", inp=120, out=40):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=inp, output_tokens=out), stop_reason=stop)

    class FakeClient:
        def __init__(self, script):
            self.script, self.calls = list(script), []
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **params):
            self.calls.append(params)
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    cfg = ModelConfig(id="claude-test", provider="anthropic", model_name="claude-sonnet-5-5",
                      api_key_env="ANTHROPIC_API_KEY", cost_per_1k_input=0.003, cost_per_1k_output=0.015,
                      max_output_tokens=2000)
    settings = OrchestratorSettings(max_retries=4)   # 4 transient errors + 1 success = 5 attempts

    def make(script):
        client = FakeClient(script)
        adapter = ClaudeAdapter(cfg, settings, client=client)
        adapter._sleep = lambda s: None
        return adapter, client

    # 1. Happy path: text, real token counts, cost, request parameters
    a, c = make([message("def f(): pass")])
    r = a.generate(ModelRequest(prompt="write f", system="be brief", max_tokens=5000, temperature=0.1))
    p = c.calls[0]
    assert r.text == "def f(): pass" and (r.input_tokens, r.output_tokens) == (120, 40)
    assert abs(r.cost_usd - (0.12 * 0.003 + 0.04 * 0.015)) < 1e-9
    assert p["model"] == "claude-sonnet-5-5" and p["max_tokens"] == 2000     # capped to model limit
    assert p["system"] == "be brief" and p["temperature"] == 0.1 and p["timeout"] == 120
    print("PASS 1: request/response mapping, token counts, cost")

    # 2. No system prompt -> 'system' not sent; multiple text blocks are joined
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="A"),
                                   SimpleNamespace(type="tool_use"),
                                   SimpleNamespace(type="text", text="B")],
                          usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="end_turn")
    a, c = make([msg])
    assert a.generate(ModelRequest(prompt="x")).text == "AB" and "system" not in c.calls[0]
    print("PASS 2: optional system prompt, multi-block text")

    # 3. Rate limit + 500 + overloaded + timeout are retried, then succeed
    a, c = make([RateLimitError_("slow down"), InternalServerError("oops"),
                 OverloadedError("busy"), APITimeoutError("timed out"), message()])
    r = a.generate(ModelRequest(prompt="x"))
    assert r.attempts == 5 and len(c.calls) == 5
    print("PASS 3: transient errors retried ->", r.attempts, "attempts")

    # 4. Fatal errors are not retried
    for exc in (AuthenticationError("bad key"), BadRequestError("credit balance too low")):
        a, c = make([exc, message()])
        try:
            a.generate(ModelRequest(prompt="x"))
            raise SystemExit("FAIL 4")
        except FatalModelError as e:
            assert e.attempts == 1 and len(c.calls) == 1
    print("PASS 4: auth / bad-request errors fail immediately")

    # 5. retry-after header is honoured
    e = classify_exception(RateLimitError_("x"), "m")
    assert isinstance(e, RateLimitError) and e.retry_after == 7.0
    print("PASS 5: retry-after header parsed ->", e.retry_after)

    # 6. Empty response is treated as transient and retried
    a, c = make([message(text="  "), message("ok")])
    assert a.generate(ModelRequest(prompt="x")).text == "ok"
    print("PASS 6: empty response retried")

    # 7. Missing API key / usage events reach listeners
    os_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        ClaudeAdapter(cfg, settings)._call(ModelRequest(prompt="x", max_tokens=10))
        raise SystemExit("FAIL 7")
    except FatalModelError as e:
        assert "ANTHROPIC_API_KEY" in str(e)
    finally:
        if os_key:
            os.environ["ANTHROPIC_API_KEY"] = os_key
    events = []
    a, c = make([message()])
    a.add_listener(events.append)
    a.generate(ModelRequest(prompt="x", purpose="lite_benchmark"))
    assert events[0].success and events[0].purpose == "lite_benchmark" and events[0].total_tokens == 160
    print("PASS 7: missing key is fatal, usage event emitted")