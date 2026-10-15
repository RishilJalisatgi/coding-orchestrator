"""
models/gemini.py

Adapter for Google Gemini models via the `google-genai` SDK.

Requires:  pip install google-genai      (NOT the deprecated 'google-generativeai')
API key:   read from the env var named in the model's config (GEMINI_API_KEY by default),
           or passed explicitly via api_key=.

Retries/backoff/usage reporting come from ModelAdapter; this class only performs ONE call
(_call) and maps Gemini errors onto the base error types.

Gemini specifics handled here:
  * thinking tokens are billed as output but reported separately -> added to output_tokens
  * max_output_tokens also covers thinking, so a budget used up by thinking is a clear error
  * blocked prompts / safety-blocked responses are fatal (retrying the same prompt won't help)
  * 429 errors may carry a retryDelay hint, which is passed to the backoff logic
"""

from __future__ import annotations

import os
import re
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

logger = get_logger("models.gemini")

FATAL_STATUSES = {400, 401, 403, 404, 413, 422}
TRANSIENT_STATUSES = {408, 409, 429, 499}

TRANSIENT_EXCEPTION_NAMES = {
    "TimeoutError", "ConnectionError", "TimeoutException", "ReadTimeout", "ConnectTimeout",
    "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError", "RemoteProtocolError",
}

# finish reasons meaning the response was withheld (retrying the same prompt gives the same result)
BLOCKED_FINISH_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}


def _retry_delay_seconds(exc: Exception) -> Optional[float]:
    """Find a RetryInfo.retryDelay (e.g. '34s' or '34.5s') in a Gemini error's details."""
    details = getattr(exc, "details", None)
    stack = [details]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            delay = item.get("retryDelay")
            if isinstance(delay, str):
                match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", delay)
                if match:
                    return float(match.group(1))
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return None


def classify_exception(exc: Exception, model_id: str) -> Exception:
    """Map a google-genai (or httpx) exception to RateLimitError / TransientModelError / FatalModelError."""
    name = type(exc).__name__
    status = getattr(exc, "code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status_code", None)
    msg = f"{name}: {exc}"

    if status == 429:
        return RateLimitError(msg, retry_after=_retry_delay_seconds(exc), model_id=model_id)

    if name in TRANSIENT_EXCEPTION_NAMES:
        return TransientModelError(msg, model_id=model_id)

    if isinstance(status, int):
        if status in FATAL_STATUSES:
            return FatalModelError(msg, model_id=model_id)
        if status >= 500 or status in TRANSIENT_STATUSES:
            return TransientModelError(msg, model_id=model_id)

    return TransientModelError(msg, model_id=model_id)     # unknown: retrying is the safer default


def _enum_name(value: Any) -> Optional[str]:
    """finish_reason may be an enum or a string, depending on the SDK version."""
    if value is None:
        return None
    return getattr(value, "name", None) or str(value).split(".")[-1]


class GeminiAdapter(ModelAdapter):
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
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise FatalModelError(
                "The 'google-genai' package is not installed: run `pip install google-genai`",
                model_id=self.model_id,
            ) from e

        # The SDK takes the timeout in milliseconds. Its own retries are not enabled,
        # because ModelAdapter.generate() owns retries and counts every attempt.
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
        )
        return self._client

    # ---- the one API call -------------------------------------------------- #

    def _call(self, request: ModelRequest) -> RawResult:
        client = self._get_client()

        gen_config: dict = {"max_output_tokens": request.max_tokens}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.system:
            gen_config["system_instruction"] = request.system

        try:
            response = client.models.generate_content(
                model=self.config.model_name,
                contents=request.prompt,
                config=gen_config,
            )
        except Exception as e:
            raise classify_exception(e, self.model_id) from e

        # Prompt rejected outright
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = _enum_name(getattr(feedback, "block_reason", None))
        if block_reason:
            raise FatalModelError(f"Prompt blocked by Gemini (block_reason={block_reason})",
                                  model_id=self.model_id)

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise TransientModelError("Response contained no candidates", model_id=self.model_id)

        candidate = candidates[0]
        finish = _enum_name(getattr(candidate, "finish_reason", None))
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        text = "".join(
            p.text for p in parts
            if getattr(p, "text", None) and not getattr(p, "thought", False)
        )

        if not text.strip():
            if finish in BLOCKED_FINISH_REASONS:
                raise FatalModelError(f"Response blocked by Gemini (finish_reason={finish})",
                                      model_id=self.model_id)
            if finish == "MAX_TOKENS":
                raise FatalModelError(
                    "No visible output: the token budget was used up (likely by thinking) before "
                    "the answer (raise max_output_tokens for this model)", model_id=self.model_id)
            raise TransientModelError(f"Empty response from model (finish_reason={finish})",
                                      model_id=self.model_id)

        if finish == "MAX_TOKENS":
            logger.warning("[%s] response was cut off at max_output_tokens=%s (purpose=%s)",
                           self.model_id, request.max_tokens, request.purpose)

        usage = getattr(response, "usage_metadata", None)
        candidates_tokens = getattr(usage, "candidates_token_count", None)
        thoughts_tokens = getattr(usage, "thoughts_token_count", None)
        output_tokens = None
        if candidates_tokens is not None or thoughts_tokens is not None:
            output_tokens = (candidates_tokens or 0) + (thoughts_tokens or 0)   # thinking is billed as output

        return RawResult(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=output_tokens,
            raw=response,
        )


# --------------------------------------------------------------------------- #
# Self-test (no network, no API key): python -m models.gemini
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from types import SimpleNamespace

    from core.logger import setup_logging

    setup_logging("ERROR")

    # Fake exceptions shaped like google.genai.errors (they expose .code, .status, .details)
    class ClientError(Exception):
        def __init__(self, msg, code, status="", details=None):
            super().__init__(msg)
            self.code, self.status, self.details = code, status, details

    class ServerError(ClientError):
        pass

    class ReadTimeout(Exception):
        pass

    def response(text="hello", finish="STOP", prompt=120, cand=40, thoughts=None, block=None, parts=None):
        if parts is None:
            parts = [SimpleNamespace(text=text, thought=False)] if text is not None else []
        return SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts),
                                        finish_reason=SimpleNamespace(name=finish))],
            usage_metadata=SimpleNamespace(prompt_token_count=prompt, candidates_token_count=cand,
                                           thoughts_token_count=thoughts))

    class FakeClient:
        def __init__(self, script):
            self.script, self.calls = list(script), []
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, **params):
            self.calls.append(params)
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    cfg = ModelConfig(id="gemini-test", provider="gemini", model_name="gemini-2.5-pro",
                      api_key_env="GEMINI_API_KEY", cost_per_1k_input=0.00125,
                      cost_per_1k_output=0.01, max_output_tokens=2000)

    def make(script):
        client = FakeClient(script)
        adapter = GeminiAdapter(cfg, OrchestratorSettings(max_retries=3), client=client)
        adapter._sleep = lambda s: None
        return adapter, client

    # 1. Happy path
    a, c = make([response("print(1)")])
    r = a.generate(ModelRequest(prompt="write it", system="be brief", max_tokens=5000, temperature=0.1))
    p = c.calls[0]
    assert r.text == "print(1)" and (r.input_tokens, r.output_tokens) == (120, 40)
    assert p["model"] == "gemini-2.5-pro" and p["contents"] == "write it"
    assert p["config"] == {"max_output_tokens": 2000, "temperature": 0.1, "system_instruction": "be brief"}
    print("PASS 1: request/response mapping, token counts")

    # 2. Thinking tokens billed as output; thought parts excluded from the answer
    parts = [SimpleNamespace(text="(internal reasoning)", thought=True),
             SimpleNamespace(text="final answer", thought=False)]
    a, c = make([response(parts=parts, cand=40, thoughts=500)])
    r = a.generate(ModelRequest(prompt="x"))
    assert r.text == "final answer" and r.output_tokens == 540
    assert abs(r.cost_usd - (0.12 * 0.00125 + 0.54 * 0.01)) < 1e-9
    print("PASS 2: thinking tokens counted as output, thought text excluded")

    # 3. Transient errors are retried (429, 503, timeout)
    a, c = make([ClientError("quota", 429, "RESOURCE_EXHAUSTED"), ServerError("unavailable", 503, "UNAVAILABLE"),
                 ReadTimeout("slow"), response()])
    r = a.generate(ModelRequest(prompt="x"))
    assert r.attempts == 4 and len(c.calls) == 4
    print("PASS 3: transient errors retried ->", r.attempts, "attempts")

    # 4. Fatal errors are not retried
    for exc in (ClientError("bad key", 401, "UNAUTHENTICATED"), ClientError("bad args", 400, "INVALID_ARGUMENT"),
                ClientError("no model", 404, "NOT_FOUND"), ClientError("forbidden", 403, "PERMISSION_DENIED")):
        a, c = make([exc, response()])
        try:
            a.generate(ModelRequest(prompt="x"))
            raise SystemExit("FAIL 4")
        except FatalModelError as e:
            assert e.attempts == 1 and len(c.calls) == 1
    print("PASS 4: auth / bad-request / not-found fail immediately")

    # 5. retryDelay hint is parsed from error details
    details = {"error": {"code": 429, "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34.5s"}]}}
    e = classify_exception(ClientError("quota", 429, "RESOURCE_EXHAUSTED", details), "m")
    assert isinstance(e, RateLimitError) and e.retry_after == 34.5
    assert classify_exception(ClientError("quota", 429), "m").retry_after is None
    print("PASS 5: retryDelay parsed ->", e.retry_after)

    # 6. Blocked prompt / safety block / exhausted budget are fatal; plain empty is retried
    a, c = make([response(text=None, block="SAFETY"), response("never")])
    try:
        a.generate(ModelRequest(prompt="x"))
        raise SystemExit("FAIL 6a")
    except FatalModelError as e:
        assert "blocked" in str(e) and len(c.calls) == 1
    a, c = make([response(text=None, finish="SAFETY"), response("never")])
    try:
        a.generate(ModelRequest(prompt="x"))
        raise SystemExit("FAIL 6b")
    except FatalModelError:
        assert len(c.calls) == 1
    a, c = make([response(text=None, finish="MAX_TOKENS", thoughts=2000), response("never")])
    try:
        a.generate(ModelRequest(prompt="x"))
        raise SystemExit("FAIL 6c")
    except FatalModelError as e:
        assert "max_output_tokens" in str(e) and len(c.calls) == 1
    a, c = make([response(text="  "), response("ok")])
    assert a.generate(ModelRequest(prompt="x")).text == "ok"
    print("PASS 6: safety blocks and exhausted budget are fatal; empty retried")

    # 7. Missing API key, and usage events reach listeners
    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        GeminiAdapter(cfg, OrchestratorSettings())._call(ModelRequest(prompt="x", max_tokens=10))
        raise SystemExit("FAIL 7")
    except FatalModelError as e:
        assert "GEMINI_API_KEY" in str(e)
    finally:
        if saved:
            os.environ["GEMINI_API_KEY"] = saved
    events = []
    a, c = make([response()])
    a.add_listener(events.append)
    a.generate(ModelRequest(prompt="x", purpose="lite_benchmark"))
    assert events[0].success and events[0].purpose == "lite_benchmark" and events[0].total_tokens == 160
    print("PASS 7: missing key is fatal, usage event emitted")