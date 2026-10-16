"""
config/settings.py

Central configuration for the orchestrator.

- Defines every tunable setting and the model registry (capabilities, costs, quotas).
- Works out of the box with built-in defaults.
- Optionally overridden by config/config.json (see bottom of file for the format).
- API keys are NEVER stored here; they are read from environment variables
  or from a .env file in the project root.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
STATE_DIR = DATA_DIR / "state"
PROJECTS_DIR = BASE_DIR / "projects"
BENCHMARK_DB_PATH = DATA_DIR / "benchmarks.db"
USER_CONFIG_PATH = BASE_DIR / "config" / "config.json"
DOTENV_PATH = BASE_DIR / ".env"


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

# Providers that use a logged-in vendor CLI (subscription) instead of an API key.
CLI_PROVIDER_NAMES = {"claude_cli": "claude", "codex_cli": "codex"}

@dataclass
class ModelConfig:
    """Everything the orchestrator needs to know about one model."""

    id: str                              # unique key, e.g. "claude-sonnet"
    provider: str                        # "anthropic" | "openai" | "gemini" | "local"
    model_name: str                      # the provider's actual model string
    enabled: bool = True

    # Auth / endpoint
    api_key_env: Optional[str] = None    # env var holding the key (None for local)
    base_url: Optional[str] = None       # used by local models (e.g. Ollama)

    # Capability scores, 0-10. Used as priors until real benchmark data exists.
    capabilities: dict = field(default_factory=lambda: {"coding": 5.0, "hld": 5.0, "review": 5.0})

    # Cost in USD per 1k tokens (0 for local)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    # Quota: tokens allowed per rolling window. None = unlimited.
    token_budget_per_window: Optional[int] = None
    window_seconds: int = 3600
    requests_per_minute: Optional[int] = None

    max_output_tokens: int = 4096

    # Provider-specific extras, e.g. for "claude_cli" / "codex_cli":
    #   {"cli_path": "...", "timeout_seconds": 600, "extra_args": ["--effort", "low"]}
    options: dict = field(default_factory=dict)

    @property
    def is_unlimited(self) -> bool:
        return self.token_budget_per_window is None


def default_models() -> list[ModelConfig]:
    """Starter registry. Edit model names / budgets to match what you actually have access to."""
    return [
        ModelConfig(
            id="claude-sonnet",
            provider="anthropic",
            model_name="claude-sonnet-5-5",
            api_key_env="ANTHROPIC_API_KEY",
            capabilities={"coding": 9.0, "hld": 8.5, "review": 9.0},
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            token_budget_per_window=50_000,
            window_seconds=3600,
            requests_per_minute=50,
        ),
        ModelConfig(
            id="gpt",
            provider="openai",
            model_name="gpt-4o",  # change to whichever model you use
            api_key_env="OPENAI_API_KEY",
            capabilities={"coding": 8.0, "hld": 8.5, "review": 8.0},
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
            token_budget_per_window=100_000,
            requests_per_minute=60,
        ),
        ModelConfig(
            id="gemini",
            provider="gemini",
            model_name="gemini-2.5-pro",  # change to whichever model you use
            api_key_env="GEMINI_API_KEY",
            capabilities={"coding": 7.5, "hld": 7.5, "review": 7.5},
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.01,
            token_budget_per_window=200_000,
            requests_per_minute=60,
        ),
        ModelConfig(
            id="local-qwen",
            provider="local",
            model_name="qwen2.5-coder",
            base_url="http://localhost:11434",  # Ollama default
            capabilities={"coding": 6.0, "hld": 5.0, "review": 5.0},
            token_budget_per_window=None,       # unlimited
            max_output_tokens=4096,
        ),
    ]


# --------------------------------------------------------------------------- #
# Orchestrator behaviour settings
# --------------------------------------------------------------------------- #

# Default metric weights for scoring candidates (must cover the same keys everywhere).
DEFAULT_SCORE_WEIGHTS = {
    "correctness": 0.40,
    "test_pass_rate": 0.20,
    "runtime": 0.10,
    "memory": 0.05,
    "security": 0.05,
    "code_quality": 0.10,
    "architecture": 0.05,
    "documentation": 0.05,
}

# Task-type specific overrides. Missing metrics fall back to 0; weights are normalised on use.
TASK_TYPE_WEIGHTS = {
    "competitive_programming": {"correctness": 0.60, "test_pass_rate": 0.25, "runtime": 0.15},
    "backend_api": {"correctness": 0.30, "test_pass_rate": 0.20, "security": 0.20,
                    "architecture": 0.15, "code_quality": 0.10, "documentation": 0.05},
    "ml_project": {"correctness": 0.35, "test_pass_rate": 0.15, "runtime": 0.15,
                   "memory": 0.15, "code_quality": 0.10, "documentation": 0.10},
    "desktop_application": {"correctness": 0.35, "test_pass_rate": 0.20, "code_quality": 0.15,
                            "architecture": 0.15, "runtime": 0.10, "documentation": 0.05},
}


@dataclass
class OrchestratorSettings:
    # Workflow
    max_review_iterations: int = 3          # review/revise loop cap
    min_complexity_for_orchestration: float = 4.0   # below this, use a single model directly (0-10)
    max_benchmark_candidates: int = 3       # how many models compete on the lite project
    lite_task_token_budget: int = 8_000     # per-candidate cap for the lite benchmark

    # Learning from history
    min_history_to_skip_benchmark: int = 10  # similar past runs needed before skipping benchmark
    history_confidence_margin: float = 0.05  # winner must lead by this to skip benchmarking

    # Reliability
    max_retries: int = 3
    request_timeout_seconds: int = 120
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0

    # Quota safety: never plan to spend the last X% of a model's window
    quota_safety_margin: float = 0.10

    # Evaluation
    sandbox_timeout_seconds: int = 60

    # Logging
    log_level: str = "INFO"

    # Scoring
    score_weights: dict = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    task_type_weights: dict = field(default_factory=lambda: {k: dict(v) for k, v in TASK_TYPE_WEIGHTS.items()})

    def weights_for(self, task_type: Optional[str]) -> dict:
        """Return normalised weights (sum = 1) for a given task type."""
        weights = self.task_type_weights.get(task_type or "", self.score_weights)
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}


# --------------------------------------------------------------------------- #
# Top-level settings object
# --------------------------------------------------------------------------- #

@dataclass
class Settings:
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)
    models: list[ModelConfig] = field(default_factory=default_models)

    def get_model(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"Unknown model id: {model_id!r}")

    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled and self.has_credentials(m)]

    @staticmethod
    def get_api_key(model: ModelConfig) -> Optional[str]:
        return os.environ.get(model.api_key_env) if model.api_key_env else None

    def has_credentials(self, model: ModelConfig) -> bool:
        """Local models need nothing; CLI providers need a logged-in CLI; hosted models need a key."""
        return self.credential_problem(model) is None

    def credential_problem(self, model: ModelConfig) -> Optional[str]:
        """Why this model cannot be used right now (None = it can)."""
        if model.provider == "local":
            return None
        cli = CLI_PROVIDER_NAMES.get((model.provider or "").lower())
        if cli:
            from models.cli_common import CLI_PROVIDERS, auth_status   # lazy: avoids a circular import
            status = auth_status(cli, (model.options or {}).get("cli_path"))
            if status.logged_in:
                return None
            if not status.installed:
                return f"{cli} CLI {status.detail}"
            return f"not logged in: run `python main.py --login {CLI_PROVIDERS[cli].name}`"
        if self.get_api_key(model):
            return None
        return f"missing API key (set {model.api_key_env})"


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #

def ensure_dirs() -> None:
    for d in (DATA_DIR, LOG_DIR, STATE_DIR, PROJECTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path = DOTENV_PATH) -> None:
    """Minimal .env reader (KEY=VALUE per line). Does not overwrite existing env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _apply_user_config(settings: Settings, data: dict) -> None:
    """Merge config.json into the defaults."""
    # 1. orchestrator settings: simple key overrides
    for key, value in data.get("orchestrator", {}).items():
        if hasattr(settings.orchestrator, key):
            setattr(settings.orchestrator, key, value)

    # 2. models: update existing by id, or add new ones
    existing = {m.id: m for m in settings.models}
    for entry in data.get("models", []):
        model_id = entry.get("id")
        if not model_id:
            continue
        if model_id in existing:
            for key, value in entry.items():
                if hasattr(existing[model_id], key):
                    setattr(existing[model_id], key, value)
        else:
            settings.models.append(ModelConfig(**entry))


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Build the Settings object: defaults -> .env -> config.json overrides."""
    ensure_dirs()
    _load_dotenv()

    settings = Settings()
    path = config_path or USER_CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            _apply_user_config(settings, json.load(f))
    return settings


# --------------------------------------------------------------------------- #
# Example config/config.json (optional)
# --------------------------------------------------------------------------- #
# {
#   "orchestrator": {
#     "max_review_iterations": 4,
#     "log_level": "DEBUG"
#   },
#   "models": [
#     {"id": "claude-sonnet", "token_budget_per_window": 30000},
#     {"id": "gpt", "model_name": "gpt-4.1"},
#     {"id": "gemini", "enabled": false}
#   ]
# }
#
# Example .env (project root):
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...


if __name__ == "__main__":
    # Quick sanity check: python -m config.settings
    s = load_settings()
    print(f"Loaded {len(s.models)} models, {len(s.enabled_models())} usable:")
    for m in s.models:
        status = "OK" if s.has_credentials(m) and m.enabled else "no key / disabled"
        print(f"  - {m.id:14} {m.provider:10} {m.model_name:22} [{status}]")
