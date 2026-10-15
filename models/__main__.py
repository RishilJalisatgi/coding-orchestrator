"""
models/__main__.py -- self-test for the model registry (no network, no API keys).

Run from the project root:   python -m models
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from config.settings import ModelConfig, OrchestratorSettings, Settings
from core.resource_manager import ResourceManager
from models import (FatalModelError, ModelAdapter, ModelRegistry, ModelRequest, ModelUnavailable,
                    RawResult, UnknownProviderError, adapter_class, create_adapter)


class FakeAdapter(ModelAdapter):
    def _call(self, request: ModelRequest) -> RawResult:
        return RawResult(text="ok", input_tokens=100, output_tokens=50)


def main() -> None:
    saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")}
    try:
        orch = OrchestratorSettings()
        cfgs = [
            ModelConfig(id="hosted", provider="anthropic", model_name="m", api_key_env="ANTHROPIC_API_KEY",
                        token_budget_per_window=10_000, window_seconds=3600),
            ModelConfig(id="off", provider="local", model_name="q", enabled=False),
            ModelConfig(id="local", provider="local", model_name="q"),
        ]
        settings = Settings(orchestrator=orch, models=cfgs)

        # 1. Models without keys / disabled are skipped with a reason; the rest load
        with tempfile.TemporaryDirectory() as tmp:
            rm = ResourceManager(cfgs, orch, state_path=Path(tmp) / "res.json")
            reg = ModelRegistry(settings, rm)
            assert reg.ids() == ["local"] and len(reg) == 1 and "local" in reg
            assert "ANTHROPIC_API_KEY" in reg.skipped["hosted"] and "disabled" in reg.skipped["off"]
            print("PASS 1: registry loads usable models, records why others were skipped")

            # 2. Unknown / unavailable models raise ModelUnavailable with a useful message
            for mid, word in (("hosted", "unavailable"), ("nope", "Unknown")):
                try:
                    reg.get(mid)
                    raise SystemExit("FAIL 2")
                except ModelUnavailable as e:
                    assert word in str(e)
            assert reg.try_generate("nope", ModelRequest(prompt="x")) is None
            print("PASS 2: unknown / skipped models raise ModelUnavailable; try_generate -> None")

            # 3. Usage from an added adapter flows into the resource manager
            fake = FakeAdapter(cfgs[0], orch)
            reg.add_adapter(fake)
            assert "hosted" in reg and "hosted" not in reg.skipped
            r = reg.generate("hosted", ModelRequest(prompt="hi", purpose="coding"))
            assert r.text == "ok" and r.total_tokens == 150
            st = rm.status("hosted")
            assert st.remaining_tokens == 10_000 - 150, st.remaining_tokens
            print("PASS 3: usage tracked in ResourceManager ->", st.remaining_tokens, "tokens left")

            # 4. Factory: provider classes resolve lazily; bad provider is rejected
            assert adapter_class("local").__name__ == "LocalAdapter"
            assert adapter_class("openai_compatible").__name__ == "OpenAIAdapter"
            try:
                adapter_class("bogus")
                raise SystemExit("FAIL 4")
            except UnknownProviderError:
                pass
            assert create_adapter(cfgs[2], orch).model_id == "local"
            print("PASS 4: provider factory")

            # 5. Missing key surfaces as a fatal error from the real adapter
            try:
                create_adapter(cfgs[0], orch).generate(ModelRequest(prompt="x"))
                raise SystemExit("FAIL 5")
            except FatalModelError as e:
                assert "ANTHROPIC_API_KEY" in str(e) or "anthropic" in str(e).lower()
            print("PASS 5: missing API key is a FatalModelError")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


if __name__ == "__main__":
    main()
