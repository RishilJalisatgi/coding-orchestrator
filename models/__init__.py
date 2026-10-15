"""
models/__init__.py

Model factory and registry. The rest of the code never imports a provider adapter directly:

    from models import ModelRegistry, ModelRequest

    registry = ModelRegistry(settings, resource_manager)      # wires usage tracking automatically
    response = registry.generate("claude-sonnet", ModelRequest(prompt="...", purpose="hld"))

Provider names (the `provider` field of a model config):
    anthropic          -> ClaudeAdapter
    openai             -> OpenAIAdapter
    openai_compatible  -> OpenAIAdapter  (LM Studio, vLLM, OpenRouter...: set base_url, and set
                                          api_key_env to an env var holding any non-empty value)
    gemini             -> GeminiAdapter
    local              -> LocalAdapter   (Ollama)

Adapter modules are imported lazily, so a problem with one provider's module can't break the others.

NOTE: core.resource_manager imports models.base, so importing it here would be circular.
The registry only needs an object with a .record(event) method (duck-typed).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Optional

from config.settings import ModelConfig, OrchestratorSettings, Settings
from core.logger import get_logger
from models.base import (
    FatalModelError,
    ModelAdapter,
    ModelError,
    ModelRequest,
    ModelResponse,
    RateLimitError,
    RawResult,
    TransientModelError,
    UsageEvent,
    UsageListener,
)

if TYPE_CHECKING:                                  # type hints only; avoids a circular import
    from core.resource_manager import ResourceManager

logger = get_logger("models")

# provider name -> (module path, class name)
PROVIDERS: dict = {
    "anthropic": ("models.claude", "ClaudeAdapter"),
    "openai": ("models.openai", "OpenAIAdapter"),
    "openai_compatible": ("models.openai", "OpenAIAdapter"),
    "gemini": ("models.gemini", "GeminiAdapter"),
    "local": ("models.local", "LocalAdapter"),
}

__all__ = [
    "PROVIDERS", "UnknownProviderError", "ModelUnavailable", "adapter_class", "create_adapter",
    "ModelRegistry",
    # re-exported from models.base for convenience
    "ModelAdapter", "ModelRequest", "ModelResponse", "RawResult", "UsageEvent", "UsageListener",
    "ModelError", "TransientModelError", "RateLimitError", "FatalModelError",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class UnknownProviderError(ValueError):
    """A model config names a provider that has no adapter."""


class ModelUnavailable(Exception):
    """The requested model is not usable (unknown, disabled, or missing credentials)."""


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def adapter_class(provider: str) -> type:
    """Resolve a provider name to its adapter class (imports the adapter module lazily)."""
    key = (provider or "").lower()
    if key not in PROVIDERS:
        raise UnknownProviderError(
            f"Unknown provider {provider!r}. Valid providers: {', '.join(sorted(PROVIDERS))}")
    module_name, class_name = PROVIDERS[key]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_adapter(
    config: ModelConfig,
    settings: OrchestratorSettings,
    listeners: Optional[list] = None,
    **kwargs: Any,
) -> ModelAdapter:
    """
    Build the right adapter for a model config. Extra keyword arguments are passed to the
    adapter (api_key=..., client=..., transport=... for tests).
    """
    cls = adapter_class(config.provider)
    return cls(config, settings, listeners=listeners, **kwargs)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class ModelRegistry:
    """
    Holds one adapter per usable model and wires usage tracking to the resource manager.

    Models that are disabled, lack credentials, or fail to construct are NOT in the registry;
    the reason is kept in `.skipped` so errors and decision logs can explain why.
    """

    def __init__(self, settings: Settings, resource_manager: Optional["ResourceManager"] = None):
        self.settings = settings
        self.resource_manager = resource_manager
        self._adapters: dict = {}
        self.skipped: dict = {}                      # model_id -> reason

        for cfg in settings.models:
            if not cfg.enabled:
                self.skipped[cfg.id] = "disabled in config"
                continue
            if not settings.has_credentials(cfg):
                self.skipped[cfg.id] = f"missing API key (set {cfg.api_key_env})"
                continue
            try:
                self._adapters[cfg.id] = create_adapter(cfg, settings.orchestrator, listeners=self._listeners())
            except Exception as e:                   # one bad config must not take down the rest
                self.skipped[cfg.id] = f"could not create adapter: {e}"
                logger.error("Could not create adapter for %s: %s", cfg.id, e)

        logger.info("Model registry ready: %d usable (%s); %d skipped",
                    len(self._adapters), ", ".join(self._adapters) or "none", len(self.skipped))
        for model_id, reason in self.skipped.items():
            logger.info("  skipped %s: %s", model_id, reason)

    def _listeners(self) -> list:
        return [self.resource_manager.record] if self.resource_manager else []

    # ---- lookup ------------------------------------------------------------ #

    def ids(self) -> list:
        return list(self._adapters)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)

    def get(self, model_id: str) -> ModelAdapter:
        adapter = self._adapters.get(model_id)
        if adapter is None:
            if model_id in self.skipped:
                raise ModelUnavailable(f"Model {model_id!r} is unavailable: {self.skipped[model_id]}")
            raise ModelUnavailable(
                f"Unknown model {model_id!r}. Known models: {', '.join(self._adapters) or 'none'}")
        return adapter

    def config(self, model_id: str) -> ModelConfig:
        return self.get(model_id).config

    # ---- use --------------------------------------------------------------- #

    def generate(self, model_id: str, request: ModelRequest) -> ModelResponse:
        """Call a model by id (retries, usage tracking and cost are handled by the adapter)."""
        return self.get(model_id).generate(request)

    def try_generate(self, model_id: str, request: ModelRequest) -> Optional[ModelResponse]:
        """Like generate(), but returns None on any model failure or unavailability."""
        try:
            return self.get(model_id).generate(request)
        except (ModelUnavailable, ModelError):
            return None

    # ---- extension --------------------------------------------------------- #

    def add_adapter(self, adapter: ModelAdapter) -> None:
        """Register a ready-made adapter (custom providers, tests). Usage tracking is attached."""
        if self.resource_manager and self.resource_manager.record not in adapter.listeners:
            adapter.add_listener(self.resource_manager.record)
        self._adapters[adapter.model_id] = adapter
        self.skipped.pop(adapter.model_id, None)