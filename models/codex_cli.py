"""
models/codex_cli.py

Adapter that uses your ChatGPT *subscription* (Plus / Pro / Business) through the official
`codex` CLI instead of an OpenAI API key. Provider name in config.json: "codex_cli".

Setup (once):      npm install -g @openai/codex
Log in / out:      python main.py --login codex | --logout codex | --switch codex

How it behaves like a plain model:
  * `codex exec -` : prompt on stdin, non-interactive
  * `-s read-only` : the sandbox cannot write files (Codex has no switch to remove its shell
                     tool entirely, so read-only + an empty temp folder + instructions is the guard)
  * `--ephemeral`, `--ignore-user-config`, `--ignore-rules`: no saved sessions, no personal
    config/MCP servers/rules leaking in (login is still read from CODEX_HOME)
  * runs in an empty temp folder, so it never sees your project
  * the final message is read from `-o <file>` (falls back to the --json event stream)
  * OPENAI_API_KEY / CODEX_API_KEY are removed from the child's environment so the CLI uses the
    ChatGPT login rather than billing a key

Codex has no system-prompt flag, so the system text is placed at the top of the prompt.
Not supported by the CLI: temperature, max output tokens (ignored).

Optional per-model settings in config.json under "options":
    {"cli_path": "C:/path/to/codex.exe", "timeout_seconds": 600, "extra_args": ["-c", "model_reasoning_effort=low"]}
"""

from __future__ import annotations

import json
import os
import shutil
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

logger = get_logger("models.codex_cli")

TEXT_ONLY_RULES = (
    "You are a text-in, text-out assistant. Do NOT run commands, read the file system, or create "
    "or edit files. Never say you have saved or created files: put everything, including all code, "
    "directly in your reply, in exactly the format the task asks for."
)


def classify_error(text: str, model_id: str) -> Exception:
    return classify_cli_error(text, model_id, "codex")


def parse_events(stdout: str) -> dict:
    """
    Read the `--json` event stream (one JSON object per line; other lines are log noise).
    Returns {"text": last agent message, "usage": {...}, "error": last failure message,
             "completed": bool}.
    Only `turn.failed` counts as a hard failure: plain `error` events include harmless
    "Reconnecting..." notices.
    """
    result = {"text": "", "usage": {}, "error": "", "last_notice": "", "completed": False}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = ev.get("type", "")
        if kind == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") in ("agent_message", "assistant_message") and item.get("text"):
                result["text"] = item["text"]
        elif kind == "turn.completed":
            result["completed"] = True
            result["usage"] = ev.get("usage") or {}
        elif kind == "turn.failed":
            err = ev.get("error")
            result["error"] = (err.get("message") if isinstance(err, dict) else str(err or "")) or "turn failed"
        elif kind == "error":
            result["last_notice"] = str(ev.get("message") or "")
    return result


class CodexCliAdapter(ModelAdapter):
    def __init__(
        self,
        config: ModelConfig,
        settings: OrchestratorSettings,
        listeners: Optional[list[UsageListener]] = None,
        runner: Optional[Callable] = None,       # inject a fake runner in tests
    ):
        super().__init__(config, settings, listeners)
        self._runner = runner or run_cli

    @property
    def _options(self) -> dict:
        return getattr(self.config, "options", None) or {}

    @property
    def cli_timeout(self) -> float:
        return float(self._options.get("timeout_seconds", max(self.timeout, 300)))

    def build_command(self, binary: str, scratch: str, out_file: str) -> list:
        cmd = [
            binary, "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "-s", "read-only",
            "-C", scratch,
            "-o", out_file,
        ]
        if self.config.model_name:
            cmd += ["-m", self.config.model_name]
        cmd += [str(a) for a in self._options.get("extra_args", [])]
        cmd.append("-")                           # read the prompt from stdin
        return cmd

    def _call(self, request: ModelRequest) -> RawResult:
        binary = find_binary("codex", self._options.get("cli_path"))
        if not binary:
            raise FatalModelError("The 'codex' CLI is not installed: npm install -g @openai/codex",
                                  model_id=self.model_id)

        instructions = f"{request.system.strip()}\n\n{TEXT_ONLY_RULES}" if request.system else TEXT_ONLY_RULES
        prompt = f"<instructions>\n{instructions}\n</instructions>\n\n{request.prompt}"

        scratch = tempfile.mkdtemp(prefix="orch-codex-")
        out_file = os.path.join(scratch, "last_message.txt")
        try:
            out = self._runner(self.build_command(binary, scratch, out_file), prompt,
                               self.cli_timeout, clean_env("codex"), self.model_id, cwd=scratch)
            events = parse_events(out.stdout)

            text = ""
            try:
                with open(out_file, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                pass
            if not text.strip():
                text = events["text"]
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        if not text.strip():
            problem = (events["error"] or events["last_notice"]
                       or (out.stderr or "").strip()[-500:] or f"exit code {out.returncode}")
            raise classify_error(problem, self.model_id)
        if events["error"]:                       # got text but the turn was reported failed
            raise classify_error(events["error"], self.model_id)

        usage = events["usage"]
        return RawResult(text=text,
                         input_tokens=int(usage.get("input_tokens") or 0) or None,
                         output_tokens=int(usage.get("output_tokens") or 0) or None,
                         raw=events)
