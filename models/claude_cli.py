"""
models/claude_cli.py

Adapter that uses your Claude *subscription* (Pro / Max) through the official `claude` CLI
instead of an API key. Provider name in config.json: "claude_cli".

Setup (once):      npm install -g @anthropic-ai/claude-code
Log in / out:      python main.py --login claude | --logout claude | --switch claude

How it behaves like a plain model (so your coder/reviewer parsers keep working):
  * `-p` headless mode, prompt sent on stdin, JSON result read back
  * `--tools=`  -> ALL tools are disabled: the model can only reply with text, it can't edit files
  * `--system-prompt-file` -> replaces Claude Code's own coding-agent prompt with yours
  * runs in an empty temp folder, so it never sees your project
  * NOT using `--bare`: that mode ignores subscription logins
  * ANTHROPIC_API_KEY is removed from the child's environment, otherwise the CLI would bill the key

Not supported by the CLI: temperature (ignored). Output length is set via CLAUDE_CODE_MAX_OUTPUT_TOKENS.

Optional per-model settings in config.json under "options":
    {"cli_path": "C:/path/to/claude.exe", "timeout_seconds": 600, "extra_args": ["--effort", "low"]}

Note: since 2026-06-15, `claude -p` on a subscription draws from a separate monthly Agent SDK
credit, not your normal chat limits. When it runs out you get a clear error (not a retry loop).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Callable, Optional

from config.settings import ModelConfig, OrchestratorSettings
from core.logger import get_logger
from models.base import (
    FatalModelError,
    ModelAdapter,
    ModelRequest,
    RawResult,
    TransientModelError,
    UsageListener,
)
from models.cli_common import classify_cli_error, clean_env, find_binary, run_cli

logger = get_logger("models.claude_cli")

TEXT_ONLY_RULES = (
    "You are a text-in, text-out assistant with NO tools. You cannot create, edit or read files "
    "or run commands. Never say you have saved or created files: put everything, including all "
    "code, directly in your reply, in exactly the format the task asks for."
)

def classify_error(text: str, model_id: str) -> Exception:
    return classify_cli_error(text, model_id, "claude")


def _parse_json(stdout: str) -> Optional[dict]:
    """The CLI prints one JSON object; tolerate stray warning lines around it."""
    try:
        data = json.loads(stdout)
        return data if isinstance(data, dict) else None
    except (ValueError, TypeError):
        pass
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                return data if isinstance(data, dict) else None
            except ValueError:
                continue
    return None


class ClaudeCliAdapter(ModelAdapter):
    def __init__(
        self,
        config: ModelConfig,
        settings: OrchestratorSettings,
        listeners: Optional[list[UsageListener]] = None,
        runner: Optional[Callable] = None,       # inject a fake runner in tests
    ):
        super().__init__(config, settings, listeners)
        self._runner = runner or run_cli
        self._warned_temperature = False

    # ---- helpers ----------------------------------------------------------- #

    @property
    def _options(self) -> dict:
        return getattr(self.config, "options", None) or {}

    @property
    def cli_timeout(self) -> float:
        # Whole-project replies take longer than a quick API call: never below 5 minutes.
        return float(self._options.get("timeout_seconds", max(self.timeout, 300)))

    def build_command(self, binary: str, system_file: str) -> list:
        cmd = [
            binary, "-p",
            "--output-format", "json",
            "--tools=",                          # text-only: disables every tool
            "--no-session-persistence",
            "--max-turns", "1",
            "--system-prompt-file", system_file,
        ]
        if self.config.model_name:
            cmd += ["--model", self.config.model_name]
        cmd += [str(a) for a in self._options.get("extra_args", [])]
        return cmd

    # ---- the one call ------------------------------------------------------ #

    def _call(self, request: ModelRequest) -> RawResult:
        binary = find_binary("claude", self._options.get("cli_path"))
        if not binary:
            raise FatalModelError(
                "The 'claude' CLI is not installed: npm install -g @anthropic-ai/claude-code",
                model_id=self.model_id)

        if request.temperature not in (None, 0.2) and not self._warned_temperature:
            logger.info("[%s] the claude CLI has no temperature setting; ignoring it", self.model_id)
            self._warned_temperature = True

        system = f"{request.system.strip()}\n\n{TEXT_ONLY_RULES}" if request.system else TEXT_ONLY_RULES
        fd, system_file = tempfile.mkstemp(prefix="orch-sys-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(system)
            extra = {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": request.max_tokens} if request.max_tokens else None
            env = clean_env("claude", extra)
            out = self._runner(self.build_command(binary, system_file), request.prompt,
                               self.cli_timeout, env, self.model_id)
        finally:
            try:
                os.remove(system_file)
            except OSError:
                pass

        data = _parse_json(out.stdout)
        if data is None:
            # No JSON at all: crashed, killed, or an old CLI. Classify from whatever it printed.
            raise classify_error((out.stderr or out.stdout or f"exit code {out.returncode}")[:500],
                                 self.model_id)

        text = data.get("result") or ""
        if data.get("is_error") or out.returncode != 0:
            raise classify_error(text or out.stderr or f"exit code {out.returncode}", self.model_id)
        if not text.strip():
            raise TransientModelError(
                f"Empty response from claude CLI (stop_reason={data.get('stop_reason')})",
                model_id=self.model_id)
        if data.get("stop_reason") == "max_tokens":
            logger.warning("[%s] response was cut off at max tokens (purpose=%s)",
                           self.model_id, request.purpose)

        usage = data.get("usage") or {}
        input_tokens = sum(int(usage.get(k) or 0) for k in
                           ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        output_tokens = int(usage.get("output_tokens") or 0)
        return RawResult(text=text,
                         input_tokens=input_tokens or None,    # None -> base class estimates
                         output_tokens=output_tokens or None,
                         raw=data)
