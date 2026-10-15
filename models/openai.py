"""
models/openai.py

Adapter for OpenAI models (Chat Completions API). Also works with any
OpenAI-compatible server if `base_url` is set on the model config.

Requires:  pip install openai
API key:   read from the env var named in the model's config (OPENAI_API_KEY by default),
           or passed explicitly via api_key=.

NOTE: always run from the project root (python -m ...). Running this file directly from
inside models/ would let it shadow the real `openai` package.

Retries/backoff/usage reporting come from ModelAdapter; this class only performs ONE call
(_call) and maps OpenAI exceptions onto the base error types.
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

logger = get_logger("models.openai")

FATAL_STATUSES = {400, 401, 402, 403, 404, 413, 422}
TRANSIENT_STATUSES = {408, 409, 429}

# Error codes that arrive as retryable-looking statuses but can never succeed on retry
FATAL_ERROR_CODES = {"insufficient_quota", "billing_hard_limit_reached", "model_not_found",
                     "invalid_api_key", "context_length_exceeded"}

# Reasoning models reject a custom temperature
REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def is_reasoning_model(model_name: str) -> bool:
    name = model_name.lower().split("/")[-1]       # tolerate "openai/o3-mini"-style names
    return name.startswith(REASONING_PREFIXES)


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_exception(exc: Exception, model_id: str) -> Exception:
    """Map an OpenAI SDK exception to RateLimitError / TransientModelError / FatalModelError."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    msg = f"{name}: {exc}"

    # A quota/billing problem is reported as 429 but retrying cannot fix it
    if isinstance(code, str) and code in FATAL_ERROR_CODES:
        return FatalModelError(msg, model_id=model_id)

    if status == 429 or name == "RateLimitError":
        return RateLimitError(msg, retry_after=_retry_after_seconds(exc), model_id=model_id)

    if name in ("APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectionError"):
        return TransientModelError(msg, model_id=model_id)

    if isinstance(status, int):
        if status in FATAL_STATUSES or name in ("AuthenticationError", "PermissionDeniedError",
                                                "NotFoundError", "BadRequestError",
                                                "UnprocessableEntityError"):
            return FatalModelError(msg, model_id=model_id)
        if status >= 500 or status in TRANSIENT_STATUSES:
            return TransientModelError(msg, model_id=model_id)

    return TransientModelError(msg, model_id=model_id)


class OpenAIAdapter(ModelAdapter):
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
        if self._client is not None:
            return self._client

        key = self._api_key or (os.environ.get(self.config.api_key_env) if self.config.api_key_env else None)
        if not key:
            raise FatalModelError(
                f"No API key for {self.model_id}: set the {self.config.api_key_env} environment variable",
                model_id=self.model_id,
            )
        try:
            import openai
        except ImportError as e:
            raise FatalModelError(
                "The 'openai' package is not installed: run `pip install openai`",
                model_id=self.model_id,
            ) from e

        kwargs: dict = {"api_key": key, "max_retries": 0, "timeout": self.timeout}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        # max_retries=0: ModelAdapter.generate() owns retries, so each attempt is counted once.
        self._client = openai.OpenAI(**kwargs)
        return self._client

    # ---- the one API call -------------------------------------------------- #

    def _call(self, request: ModelRequest) -> RawResult:
        client = self._get_client()

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        params: dict = {
            "model": self.config.model_name,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None and not is_reasoning_model(self.config.model_name):
            params["temperature"] = request.temperature

        try:
            response = client.chat.completions.create(**params)
        except Exception as e:
            raise classify_exception(e, self.model_id) from e

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise TransientModelError("Response contained no choices", model_id=self.model_id)

        choice = choices[0]
        message = getattr(choice, "message", None)
        finish = getattr(choice, "finish_reason", None)
        text = (getattr(message, "content", None) or "")

        refusal = getattr(message, "refusal", None)
        if refusal and not text.strip():
            raise FatalModelError(f"Model refused the request: {refusal}", model_id=self.model_id)

        if not text.strip():
            if finish == "length":
                # Typically a reasoning model that spent the whole budget thinking
                raise FatalModelError(
                    "No visible output: the token budget was used up before the answer "
                    "(raise max_output_tokens for this model)", model_id=self.model_id)
            raise TransientModelError(f"Empty response from model (finish_reason={finish})",
                                      model_id=self.model_id)

        if finish == "length":
            logger.warning("[%s] response was cut off at max_completion_tokens=%s (purpose=%s)",
                           self.model_id, request.max_tokens, request.purpose)

        usage = getattr(response, "usage", None)
        return RawResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),   # includes reasoning tokens
            raw=response,
        )


# --------------------------------------------------------------------------- #
# Self-test (no network, no API key): python -m models.openai
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from types import SimpleNamespace

    from core.logger import setup_logging

    setup_logging("ERROR")

    # Fake exceptions named like the SDK's, so classification is tested without the SDK.
    class RateLimitError_(Exception):
        status_code = 429
        code = "rate_limit_exceeded"
        response = SimpleNamespace(headers={"retry-after": "7"})
    RateLimitError_.__name__ = "RateLimitError"

    class QuotaError(Exception):
        status_code = 429
        code = "insufficient_quota"
    QuotaError.__name__ = "RateLimitError"

    class AuthenticationError(Exception):
        status_code = 401
        code = "invalid_api_key"

    class InternalServerError(Exception):
        status_code = 500

    class APITimeoutError(Exception):
        pass

    class BadRequestError(Exception):
        status_code = 400
        code = None

    def reply(text="hello", finish="stop", inp=120, out=40, refusal=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text, refusal=refusal),
                                     finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=inp, completion_tokens=out))

    class FakeClient:
        def __init__(self, script):
            self.script, self.calls = list(script), []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **params):
            self.calls.append(params)
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def make(script, model_name="gpt-4o"):
        cfg = ModelConfig(id="gpt-test", provider="openai", model_name=model_name,
                          api_key_env="OPENAI_API_KEY", cost_per_1k_input=0.0025,
                          cost_per_1k_output=0.01, max_output_tokens=2000)
        client = FakeClient(script)
        adapter = OpenAIAdapter(cfg, OrchestratorSettings(max_retries=3), client=client)
        adapter._sleep = lambda s: None
        return adapter, client

    # 1. Happy path
    a, c = make([reply("print(1)")])
    r = a.generate(ModelRequest(prompt="write it", system="be brief", max_tokens=5000, temperature=0.1))
    p = c.calls[0]
    assert r.text == "print(1)" and (r.input_tokens, r.output_tokens) == (120, 40)
    assert abs(r.cost_usd - (0.12 * 0.0025 + 0.04 * 0.01)) < 1e-9
    assert p["model"] == "gpt-4o" and p["max_completion_tokens"] == 2000 and p["temperature"] == 0.1
    assert p["messages"] == [{"role": "system", "content": "be brief"},
                             {"role": "user", "content": "write it"}]
    print("PASS 1: request/response mapping, token counts, cost")

    # 2. Reasoning models get no temperature; no system message when none given
    for name in ("o3-mini", "gpt-5", "openai/o4-mini"):
        a, c = make([reply()], model_name=name)
        a.generate(ModelRequest(prompt="x", temperature=0.3))
        assert "temperature" not in c.calls[0] and len(c.calls[0]["messages"]) == 1
    assert is_reasoning_model("gpt-4o") is False
    print("PASS 2: temperature omitted for reasoning models")

    # 3. Transient errors are retried
    a, c = make([RateLimitError_("slow down"), InternalServerError("oops"),
                 APITimeoutError("timed out"), reply()])
    r = a.generate(ModelRequest(prompt="x"))
    assert r.attempts == 4 and len(c.calls) == 4
    print("PASS 3: transient errors retried ->", r.attempts, "attempts")

    # 4. Fatal errors, including quota exhaustion disguised as a 429, are not retried
    for exc in (AuthenticationError("bad key"), BadRequestError("bad params"), QuotaError("no credit")):
        a, c = make([exc, reply()])
        try:
            a.generate(ModelRequest(prompt="x"))
            raise SystemExit("FAIL 4")
        except FatalModelError as e:
            assert e.attempts == 1 and len(c.calls) == 1
    print("PASS 4: auth / bad-request / insufficient_quota fail immediately")

    # 5. retry-after header
    e = classify_exception(RateLimitError_("x"), "m")
    assert isinstance(e, RateLimitError) and e.retry_after == 7.0
    print("PASS 5: retry-after header parsed ->", e.retry_after)

    # 6. Empty / budget exhausted / refusal
    a, c = make([reply(text=""), reply("ok")])
    assert a.generate(ModelRequest(prompt="x")).text == "ok"
    a, c = make([reply(text="", finish="length"), reply("never")], model_name="o3-mini")
    try:
        a.generate(ModelRequest(prompt="x"))
        raise SystemExit("FAIL 6a")
    except FatalModelError as e:
        assert "max_output_tokens" in str(e) and len(c.calls) == 1
    a, c = make([reply(text=None, refusal="I can't help with that"), reply("never")])
    try:
        a.generate(ModelRequest(prompt="x"))
        raise SystemExit("FAIL 6b")
    except FatalModelError as e:
        assert "refused" in str(e) and len(c.calls) == 1
    print("PASS 6: empty retried; exhausted budget and refusals are fatal")

    # 7. Missing API key, and usage events reach listeners
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        cfg = ModelConfig(id="g", provider="openai", model_name="gpt-4o", api_key_env="OPENAI_API_KEY")
        OpenAIAdapter(cfg, OrchestratorSettings())._call(ModelRequest(prompt="x", max_tokens=10))
        raise SystemExit("FAIL 7")
    except FatalModelError as e:
        assert "OPENAI_API_KEY" in str(e)
    finally:
        if saved:
            os.environ["OPENAI_API_KEY"] = saved
    events = []
    a, c = make([reply()])
    a.add_listener(events.append)
    a.generate(ModelRequest(prompt="x", purpose="lite_benchmark"))
    assert events[0].success and events[0].purpose == "lite_benchmark" and events[0].total_tokens == 160
    print("PASS 7: missing key is fatal, usage event emitted")