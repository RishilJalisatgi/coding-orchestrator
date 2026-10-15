"""
core/logger.py

Logging for the orchestrator.

- setup_logging(): configures console + rotating file logging (call once, in main.py).
- get_logger(name): returns a namespaced logger for any module.
- DecisionLogger: appends structured, human-inspectable decisions to a JSONL file
  (model selection, workflow choice, benchmark skipped, retries, etc.).
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from config.settings import LOG_DIR, ensure_dirs

ROOT_LOGGER_NAME = "orchestrator"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


# --------------------------------------------------------------------------- #
# Standard logging
# --------------------------------------------------------------------------- #

def setup_logging(level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """Configure the root 'orchestrator' logger. Safe to call more than once."""
    global _configured
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _configured:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_to_file:
        ensure_dirs()
        file_handler = RotatingFileHandler(
            LOG_DIR / "orchestrator.log",
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger, e.g. get_logger('core.scheduler') -> 'orchestrator.core.scheduler'."""
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


# --------------------------------------------------------------------------- #
# Decision logging
# --------------------------------------------------------------------------- #

class DecisionLogger:
    """
    Records every important orchestration decision as one JSON line.

    Each record contains:
        timestamp, run_id, project_id, category, decision, reasoning, context

    Example:
        dl = DecisionLogger(project_id="downloads-organizer")
        dl.log(
            category="model_selection",
            decision="run benchmark with gemini, gpt, local-qwen",
            reasoning="claude has only 14k tokens left; reserving it for final review",
            context={"claude_remaining": 14000},
        )
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        log_path: Optional[Path] = None,
    ):
        ensure_dirs()
        self.project_id = project_id
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.log_path = Path(log_path) if log_path else LOG_DIR / "decisions.jsonl"
        self._lock = threading.Lock()
        self._logger = get_logger("decisions")

    def log(
        self,
        category: str,
        decision: str,
        reasoning: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Write one decision record and return it."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "project_id": self.project_id,
            "category": category,
            "decision": decision,
            "reasoning": reasoning,
            "context": context or {},
        }
        line = json.dumps(record, ensure_ascii=False, default=str)

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        self._logger.info("[%s] %s | %s", category, decision, reasoning)
        return record

    def read_all(self, only_this_run: bool = False) -> list[dict]:
        """Load decisions back from disk (useful for inspection / debugging)."""
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip a corrupted line instead of crashing
                if only_this_run and rec.get("run_id") != self.run_id:
                    continue
                records.append(rec)
        return records


# --------------------------------------------------------------------------- #
# Self-test: python -m core.logger
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    setup_logging("DEBUG")
    log = get_logger("core.logger_test")
    log.info("Standard logging works.")

    dl = DecisionLogger(project_id="demo")
    dl.log(
        category="workflow_selection",
        decision="use workflow B (competitive benchmark)",
        reasoning="complexity 7.5 >= threshold 4.0",
        context={"complexity": 7.5},
    )
    dl.log(
        category="model_selection",
        decision="benchmark with gemini, gpt, local-qwen",
        reasoning="claude quota low; reserved for final review",
        context={"claude_remaining": 14000},
    )

    print(f"\nDecisions written to: {dl.log_path}")
    print(f"This run recorded {len(dl.read_all(only_this_run=True))} decisions.")