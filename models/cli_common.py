"""
models/cli_common.py

Shared plumbing for adapters that talk to a model through an official, already-logged-in CLI
(Claude Code's `claude`, OpenAI's `codex`) instead of an API key.

Why a CLI and not the HTTP API?  Your Claude / ChatGPT *subscription* login only works inside the
vendors' own apps. The orchestrator therefore runs the real binary and never touches the stored
tokens itself.

What lives here:
    find_binary()      locate the CLI (env override -> options.cli_path -> PATH)
    clean_env()        child environment WITHOUT API keys (otherwise the CLI silently bills the key)
    run_cli()          run a command with the prompt on stdin, a timeout, and a throw-away cwd
    CLI_PROVIDERS      per-provider login / logout / status commands
    auth_status()      is this CLI logged in?  (cached; cleared on login/logout)
    login() / logout() / switch_account()
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

from models.base import FatalModelError, RateLimitError, TransientModelError

# Variables that would make a CLI bill an API key instead of the subscription login.
API_KEY_ENV_VARS = {
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "codex": ("OPENAI_API_KEY", "CODEX_API_KEY"),
}

# Env var that overrides where the binary is (handy for tests and odd installs).
BIN_ENV_VARS = {"claude": "CLAUDE_BIN", "codex": "CODEX_BIN"}


# --------------------------------------------------------------------------- #
# Binary + environment
# --------------------------------------------------------------------------- #

def find_binary(cli: str, cli_path: Optional[str] = None) -> Optional[str]:
    """Return the full path of the CLI, or None if it is not installed."""
    override = os.environ.get(BIN_ENV_VARS.get(cli, "")) or cli_path
    if override:
        return shutil.which(override) or (override if os.path.exists(override) else None)
    return shutil.which(cli)


def clean_env(cli: str, extra: Optional[dict] = None) -> dict:
    """Copy of os.environ with API keys removed, so the CLI uses the logged-in account."""
    env = dict(os.environ)
    for name in API_KEY_ENV_VARS.get(cli, ()):
        env.pop(name, None)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


@dataclass
class CliOutput:
    returncode: int
    stdout: str
    stderr: str


def run_cli(cmd: list, stdin_text: str, timeout: float, env: dict, model_id: str,
            cwd: Optional[str] = None) -> CliOutput:
    """
    Run one CLI command. The prompt goes in on stdin (no length / quoting problems, also on
    Windows). Runs in an empty temp directory unless `cwd` is given, so a CLI can never see or
    touch your project files.
    """
    scratch = None
    if cwd is None:
        scratch = tempfile.mkdtemp(prefix="orch-cli-")
        cwd = scratch
    try:
        proc = subprocess.run(
            cmd, input=stdin_text, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=env, cwd=cwd,
        )
        return CliOutput(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        raise TransientModelError(f"CLI timed out after {timeout:.0f}s", model_id=model_id) from e
    except FileNotFoundError as e:
        raise FatalModelError(f"CLI not found: {cmd[0]}", model_id=model_id) from e
    except OSError as e:
        raise TransientModelError(f"Could not run CLI: {e}", model_id=model_id) from e
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Account management (login / logout / status)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CliProvider:
    name: str                 # "claude" | "codex"
    label: str                # for messages
    install_hint: str
    login_args: tuple
    logout_args: tuple
    status_args: tuple


CLI_PROVIDERS = {
    "claude": CliProvider(
        name="claude", label="Claude (Anthropic subscription)",
        install_hint="npm install -g @anthropic-ai/claude-code",
        login_args=("auth", "login", "--claudeai"),     # --claudeai = subscription, not API billing
        logout_args=("auth", "logout"),
        status_args=("auth", "status", "--json"),
    ),
    "codex": CliProvider(
        name="codex", label="ChatGPT / Codex (OpenAI subscription)",
        install_hint="npm install -g @openai/codex",
        login_args=("login",),
        logout_args=("logout",),
        status_args=("login", "status"),
    ),
}

_status_cache: dict = {}


@dataclass
class AuthStatus:
    cli: str
    installed: bool
    logged_in: bool
    detail: str = ""


def clear_auth_cache(cli: Optional[str] = None) -> None:
    if cli:
        _status_cache.pop(cli, None)
    else:
        _status_cache.clear()


def _describe_claude_status(stdout: str) -> str:
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return stdout.strip()
    parts = [str(data[k]) for k in ("email", "subscriptionType", "orgName", "authMethod") if data.get(k)]
    return ", ".join(parts)


def auth_status(cli: str, cli_path: Optional[str] = None, use_cache: bool = True) -> AuthStatus:
    """Is this CLI installed and logged in? (cached for the life of the process)"""
    if use_cache and cli in _status_cache:
        return _status_cache[cli]

    spec = CLI_PROVIDERS[cli]
    binary = find_binary(cli, cli_path)
    if not binary:
        result = AuthStatus(cli, False, False, f"not installed ({spec.install_hint})")
    else:
        try:
            proc = subprocess.run([binary, *spec.status_args], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=30,
                                  env=clean_env(cli), stdin=subprocess.DEVNULL)
            text = (proc.stdout or "") + (proc.stderr or "")
            logged_in = proc.returncode == 0
            if cli == "claude":
                try:   # claude exits non-zero when logged out, but trust the JSON too
                    logged_in = bool(json.loads(proc.stdout).get("loggedIn"))
                except (ValueError, TypeError, AttributeError):
                    pass
                detail = _describe_claude_status(proc.stdout) if logged_in else "not logged in"
            else:
                detail = text.strip().splitlines()[0] if (logged_in and text.strip()) else "not logged in"
            result = AuthStatus(cli, True, logged_in, detail)
        except (subprocess.TimeoutExpired, OSError) as e:
            result = AuthStatus(cli, True, False, f"could not check login: {e}")

    _status_cache[cli] = result
    return result


def _run_interactive(cli: str, args: tuple, cli_path: Optional[str] = None) -> int:
    """Run a login/logout command attached to the user's terminal (browser prompts, etc.)."""
    binary = find_binary(cli, cli_path)
    if not binary:
        print(f"{CLI_PROVIDERS[cli].name} CLI not found. Install it with: {CLI_PROVIDERS[cli].install_hint}")
        return 127
    try:
        return subprocess.call([binary, *args], env=clean_env(cli))
    finally:
        clear_auth_cache(cli)


def login(cli: str, cli_path: Optional[str] = None, device_auth: bool = False) -> int:
    spec = CLI_PROVIDERS[cli]
    args = spec.login_args + (("--device-auth",) if (device_auth and cli == "codex") else ())
    print(f"Signing in to {spec.label}. Finish the sign-in in your browser...")
    return _run_interactive(cli, args, cli_path)


def logout(cli: str, cli_path: Optional[str] = None) -> int:
    spec = CLI_PROVIDERS[cli]
    print(f"Signing out of {spec.label}...")
    return _run_interactive(cli, spec.logout_args, cli_path)


def switch_account(cli: str, cli_path: Optional[str] = None, device_auth: bool = False) -> int:
    """Log out of the current account, then sign in with another one."""
    logout(cli, cli_path)          # a failed logout (already logged out) must not stop the login
    return login(cli, cli_path, device_auth)


# --------------------------------------------------------------------------- #
# Error classification (shared by both CLI adapters)
# --------------------------------------------------------------------------- #

AUTH_MARKERS = ("not logged in", "/login", "invalid api key", "oauth token", "authentication",
                "unauthorized", "401", "token has expired", "re-authenticate", "sign in again",
                "log in again", "please log in", "please sign in", "refresh token")
EXHAUSTED_MARKERS = ("usage limit", "limit reached", "hit your", "credit balance", "out of credit",
                     "quota", "agent sdk credit", "monthly", "exceeded your", "upgrade your plan")
RATE_MARKERS = ("rate limit", "rate_limit", "429", "overloaded", "529", "try again", "temporarily",
                "reconnecting", "stream disconnected", "connection", "timed out")


def classify_cli_error(text: str, model_id: str, cli: str) -> Exception:
    """Map a CLI error message onto Fatal / RateLimit / Transient so retries behave sensibly."""
    spec = CLI_PROVIDERS[cli]
    low = (text or "").lower()
    msg = (text or "").strip()[:500] or "unknown CLI error"
    if any(m in low for m in AUTH_MARKERS):
        return FatalModelError(f"{spec.name} is not logged in ({msg}). Run: python main.py --login {spec.name}",
                               model_id=model_id)
    if any(m in low for m in EXHAUSTED_MARKERS):
        # Retrying within a minute will not bring a usage window / credit back.
        return FatalModelError(f"{spec.label} usage exhausted: {msg}", model_id=model_id)
    if any(m in low for m in RATE_MARKERS):
        return RateLimitError(msg, model_id=model_id)
    return TransientModelError(msg, model_id=model_id)
