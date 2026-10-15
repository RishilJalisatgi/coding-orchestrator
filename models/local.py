"""
models/local.py

Adapter for locally hosted models through Ollama's HTTP API (default http://localhost:11434).
Uses only the standard library: nothing to pip install.

Setup:   install Ollama, then e.g.  `ollama pull qwen2.5-coder`
         (set model_name in the config to the exact tag, e.g. "qwen2.5-coder:7b")

Other local servers (LM Studio, vLLM, llama.cpp) speak the OpenAI protocol: use OpenAIAdapter
with base_url set instead of this adapter.

Retries/backoff/usage reporting come from ModelAdapter; this class only performs ONE call
(_call) and maps failures onto the base error types.

Local specifics handled here:
  * num_ctx is sized to the prompt (Ollama's small default context silently truncates prompts)
  * server not running / model not pulled are fatal, with actionable messages
  * <think>...</think> blocks from reasoning models are stripped from the answer
  * timeouts are multiplied, because local inference (and cold model loads) can be slow
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

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
    estimate_tokens,
)

logger = get_logger("models.local")

DEFAULT_BASE_URL = "http://localhost:11434"
LOCAL_TIMEOUT_MULTIPLIER = 3        # local inference is slow, especially on a cold model load
MIN_CTX, MAX_CTX = 2048, 32768
CTX_HEADROOM = 256                  # slack for chat-template tokens

FATAL_STATUSES = {400, 401, 403, 404, 413, 422}
TRANSIENT_STATUSES = {408, 409}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)   # cut off mid-thought


# --------------------------------------------------------------------------- #
# HTTP transport (replaceable in tests)
# --------------------------------------------------------------------------- #

class LocalHTTPError(Exception):
    """The server answered with an HTTP error status."""

    def __init__(self, status_code: int, body: str = ""):
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


Transport = Callable[[str, Optional[dict], float], dict]


def http_json(url: str, payload: Optional[dict], timeout: float) -> dict:
    """POST `payload` as JSON (or GET when payload is None) and return the decoded JSON reply."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise LocalHTTPError(e.code, body) from e


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def choose_num_ctx(prompt_tokens: int, max_tokens: int) -> int:
    """Smallest power-of-two context (2048..32768) that fits prompt + output. Stable steps avoid reloads."""
    needed = prompt_tokens + max_tokens + CTX_HEADROOM
    ctx = MIN_CTX
    while ctx < needed and ctx < MAX_CTX:
        ctx *= 2
    return min(ctx, MAX_CTX)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (and an unterminated trailing one) from reasoning-model output."""
    return _THINK_UNCLOSED.sub("", _THINK_BLOCK.sub("", text)).strip()


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, socket.timeout))


def classify_exception(exc: Exception, model_id: str, base_url: str = DEFAULT_BASE_URL,
                       model_name: str = "") -> Exception:
    """Map a transport failure to RateLimitError / TransientModelError / FatalModelError."""
    msg = f"{type(exc).__name__}: {exc}"

    if isinstance(exc, LocalHTTPError):
        status = exc.status_code
        if status == 404:
            return FatalModelError(
                f"Model {model_name!r} not found on the server. Pull it with: ollama pull {model_name}",
                model_id=model_id)
        if status == 429:
            return RateLimitError(msg, model_id=model_id)
        if status in FATAL_STATUSES:
            return FatalModelError(msg, model_id=model_id)
        if status >= 500 or status in TRANSIENT_STATUSES:
            return TransientModelError(msg, model_id=model_id)
        return TransientModelError(msg, model_id=model_id)

    # URLError wraps the real cause in .reason
    cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc

    if isinstance(cause, ConnectionRefusedError):
        return FatalModelError(
            f"Cannot connect to {base_url}. Is Ollama running? Start it with: ollama serve",
            model_id=model_id)
    if _is_timeout(cause) or _is_timeout(exc):
        return TransientModelError(f"Request timed out: {exc}", model_id=model_id)

    # Resets, DNS blips, malformed JSON, anything else: retrying is the safer default
    return TransientModelError(msg, model_id=model_id)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #

class LocalAdapter(ModelAdapter):
    def __init__(
        self,
        config: ModelConfig,
        settings: OrchestratorSettings,
        listeners: Optional[list[UsageListener]] = None,
        transport: Optional[Transport] = None,      # inject a fake transport (used by tests)
    ):
        super().__init__(config, settings, listeners)
        self._transport: Transport = transport or http_json
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")

    @property
    def timeout(self) -> int:
        return self.settings.request_timeout_seconds * LOCAL_TIMEOUT_MULTIPLIER

    # ---- health check ------------------------------------------------------ #

    def check_server(self) -> tuple:
        """
        Returns (ok, details). ok is True when the server answers AND the configured model is
        installed. details is a human-readable explanation.
        """
        try:
            data = self._transport(f"{self.base_url}/api/tags", None, min(10, self.timeout))
        except Exception as e:
            err = classify_exception(e, self.model_id, self.base_url, self.config.model_name)
            return False, str(err)

        installed = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
        wanted = self.config.model_name
        found = any(name == wanted or name.split(":")[0] == wanted or name == f"{wanted}:latest"
                    for name in installed)
        if not found:
            return False, f"Server is up but model {wanted!r} is not installed (ollama pull {wanted}). " \
                          f"Installed: {', '.join(installed) or 'none'}"
        return True, f"Server is up; model {wanted!r} is installed"

    # ---- the one API call -------------------------------------------------- #

    def _call(self, request: ModelRequest) -> RawResult:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        prompt_tokens = estimate_tokens((request.system or "") + request.prompt)
        options: dict = {
            "num_predict": request.max_tokens,
            "num_ctx": choose_num_ctx(prompt_tokens, request.max_tokens or 0),
        }
        if request.temperature is not None:
            options["temperature"] = request.temperature

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            data = self._transport(f"{self.base_url}/api/chat", payload, self.timeout)
        except Exception as e:
            raise classify_exception(e, self.model_id, self.base_url, self.config.model_name) from e

        if not isinstance(data, dict):
            raise TransientModelError("Unexpected response format from server", model_id=self.model_id)
        if data.get("error"):
            raise TransientModelError(f"Server error: {data['error']}", model_id=self.model_id)

        content = (data.get("message") or {}).get("content") or ""
        text = strip_thinking(content)
        done_reason = data.get("done_reason")

        if not text:
            if done_reason == "length":
                raise FatalModelError(
                    "No visible output: the token budget was used up before the answer "
                    "(raise max_output_tokens for this model)", model_id=self.model_id)
            raise TransientModelError(f"Empty response from model (done_reason={done_reason})",
                                      model_id=self.model_id)

        if done_reason == "length":
            logger.warning("[%s] response was cut off at num_predict=%s (purpose=%s)",
                           self.model_id, request.max_tokens, request.purpose)

        return RawResult(
            text=text,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            raw=data,
        )


# --------------------------------------------------------------------------- #
# Self-test (no server needed): python -m models.local
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from core.logger import setup_logging

    setup_logging("ERROR")

    def reply(text="hello", done="stop", prompt=120, evals=40):
        return {"message": {"role": "assistant", "content": text}, "done": True,
                "done_reason": done, "prompt_eval_count": prompt, "eval_count": evals}

    class FakeTransport:
        def __init__(self, script):
            self.script, self.calls = list(script), []

        def __call__(self, url, payload, timeout):
            self.calls.append((url, payload, timeout))
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    cfg = ModelConfig(id="local-test", provider="local", model_name="qwen2.5-coder",
                      base_url="http://localhost:11434/", max_output_tokens=2000)

    def make(script):
        transport = FakeTransport(script)
        adapter = LocalAdapter(cfg, OrchestratorSettings(max_retries=4, request_timeout_seconds=100),
                               transport=transport)
        adapter._sleep = lambda s: None
        return adapter, transport

    # 1. Happy path: payload shape, token counts, zero cost, tripled timeout
    a, t = make([reply("print(1)")])
    r = a.generate(ModelRequest(prompt="write it", system="be brief", max_tokens=5000, temperature=0.1))
    url, payload, timeout = t.calls[0]
    assert r.text == "print(1)" and (r.input_tokens, r.output_tokens) == (120, 40) and r.cost_usd == 0
    assert url == "http://localhost:11434/api/chat" and timeout == 300
    assert payload["model"] == "qwen2.5-coder" and payload["stream"] is False
    assert payload["messages"] == [{"role": "system", "content": "be brief"},
                                   {"role": "user", "content": "write it"}]
    assert payload["options"]["num_predict"] == 2000 and payload["options"]["temperature"] == 0.1
    print("PASS 1: request/response mapping, tokens, zero cost, extended timeout")

    # 2. Context window sizing
    assert choose_num_ctx(100, 500) == 2048
    assert choose_num_ctx(3000, 2000) == 8192
    assert choose_num_ctx(100_000, 4000) == 32768          # capped
    a, t = make([reply()])
    a.generate(ModelRequest(prompt="x" * 12_000, max_tokens=2000))    # ~3000 prompt tokens
    assert t.calls[0][1]["options"]["num_ctx"] == 8192
    print("PASS 2: num_ctx grows with the prompt (power-of-two steps, capped)")

    # 3. <think> blocks are stripped
    assert strip_thinking("<think>hmm\nlet me see</think>\nthe answer") == "the answer"
    assert strip_thinking("answer<think>never finished") == "answer"
    a, t = make([reply("<think>reasoning</think>final")])
    assert a.generate(ModelRequest(prompt="x")).text == "final"
    print("PASS 3: reasoning blocks stripped")

    # 4. Transient failures are retried
    a, t = make([LocalHTTPError(500, "internal"), TimeoutError("slow"),
                 urllib.error.URLError(socket.timeout("timed out")), json.JSONDecodeError("bad", "", 0),
                 reply()])
    r = a.generate(ModelRequest(prompt="x"))
    assert r.attempts == 5 and len(t.calls) == 5
    print("PASS 4: 5xx / timeouts / bad JSON retried ->", r.attempts, "attempts")

    # 5. Server down: fatal, no retries, actionable message
    for exc in (ConnectionRefusedError("refused"), urllib.error.URLError(ConnectionRefusedError("refused"))):
        a, t = make([exc, reply()])
        try:
            a.generate(ModelRequest(prompt="x"))
            raise SystemExit("FAIL 5")
        except FatalModelError as e:
            assert "ollama serve" in str(e) and e.attempts == 1 and len(t.calls) == 1
    print("PASS 5: server not running is fatal with a clear message")

    # 6. Model not pulled: fatal, tells you the command
    a, t = make([LocalHTTPError(404, "model not found"), reply()])
    try:
        a.generate(ModelRequest(prompt="x"))
        raise SystemExit("FAIL 6")
    except FatalModelError as e:
        assert "ollama pull qwen2.5-coder" in str(e) and len(t.calls) == 1
    print("PASS 6: missing model is fatal ->", "ollama pull qwen2.5-coder")

    # 7. Empty / exhausted budget / error field
    a, t = make([reply(""), reply("ok")])
    assert a.generate(ModelRequest(prompt="x")).text == "ok"
    a, t = make([reply("<think>endless</think>", done="length"), reply("never")])
    try:
        a.generate(ModelRequest(prompt="x"))
        raise SystemExit("FAIL 7")
    except FatalModelError as e:
        assert "max_output_tokens" in str(e) and len(t.calls) == 1
    a, t = make([{"error": "model is loading"}, reply("ok")])
    assert a.generate(ModelRequest(prompt="x")).text == "ok"
    print("PASS 7: empty retried, exhausted budget fatal, error field retried")

    # 8. Health check
    tags = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3:latest"}]}
    a, t = make([tags])
    ok, info = a.check_server()
    assert ok and t.calls[0][0].endswith("/api/tags") and t.calls[0][1] is None
    a, t = make([{"models": [{"name": "llama3:latest"}]}])
    ok, info = a.check_server()
    assert not ok and "ollama pull qwen2.5-coder" in info
    a, t = make([ConnectionRefusedError("refused")])
    ok, info = a.check_server()
    assert not ok and "ollama serve" in info
    print("PASS 8: check_server (ok / model missing / server down)")

    # 9. Usage events reach listeners
    events = []
    a, t = make([reply()])
    a.add_listener(events.append)
    a.generate(ModelRequest(prompt="x", purpose="lite_benchmark"))
    assert events[0].success and events[0].purpose == "lite_benchmark" and events[0].total_tokens == 160
    print("PASS 9: usage event emitted")