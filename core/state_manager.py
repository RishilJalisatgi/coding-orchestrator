"""
core/state_manager.py

Persistent project state, artifacts and checkpoints.

Typical use (done by the orchestrator):

    sm = StateManager()
    state = sm.create_project("Build a desktop app that categorises my downloads")
    # ... or after a crash:
    state = sm.load_project(project_id)

    stage = sm.next_stage(state)                  # where to resume
    with sm.stage(state, "task_analysis") as out:
        out["complexity"] = 7.5                   # stored as the stage's data
        sm.write_artifact(state, "task_spec.json", {"complexity": 7.5})

Anything not marked completed/skipped is re-run on resume. A stage that was 'running'
when the process died is detected on load and marked pending again.

Project layout:
    projects/<project_id>/
        state.json
        requirements.md, task_spec.json, architecture.md, architecture_review.md, ...
        implementation/  tests/  benchmark/  decisions/
        history/         (older versions of overwritten artifacts)
        checkpoints/     (named state snapshots)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from config.settings import PROJECTS_DIR, ensure_dirs
from core.logger import get_logger

logger = get_logger("core.state_manager")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default pipeline, in order. The orchestrator may pass its own list to create_project().
STAGES = (
    "task_analysis",       # complexity, tech, difficulty
    "workflow_decision",   # orchestrate or use a single model directly
    "lite_generation",     # build the lite benchmark task
    "benchmark",           # run candidates and evaluate them
    "selection",           # choose the coding model
    "hld",                 # architecture / design
    "coding",              # full implementation
    "review",              # review / revise loop
)

PENDING, RUNNING, COMPLETED, FAILED, SKIPPED = "pending", "running", "completed", "failed", "skipped"
DONE_STATUSES = (COMPLETED, SKIPPED)

PROJECT_RUNNING, PROJECT_PAUSED, PROJECT_COMPLETED, PROJECT_FAILED = "running", "paused", "completed", "failed"

SUBDIRS = ("implementation", "tests", "benchmark", "decisions", "history", "checkpoints")
RESERVED_NAMES = ("state.json",)
RESERVED_DIRS = ("checkpoints", "history")

_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class StateError(Exception):
    """Base class for state manager errors."""


class ProjectNotFound(StateError):
    pass


class InvalidArtifactPath(StateError):
    pass


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #

@dataclass
class StageRecord:
    status: str = PENDING
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None
    data: dict = field(default_factory=dict)


@dataclass
class ProjectState:
    project_id: str
    requirement: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    status: str = PROJECT_RUNNING
    current_stage: Optional[str] = None
    stages: dict = field(default_factory=dict)            # stage name -> StageRecord
    stage_order: list = field(default_factory=list)
    data: dict = field(default_factory=dict)              # small shared values (selected model, etc.)
    artifacts: dict = field(default_factory=dict)         # artifact name -> relative path
    review_iteration: int = 0
    review_history: list = field(default_factory=list)    # one entry per review round
    checkpoints: list = field(default_factory=list)       # checkpoint file names

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ProjectState":
        raw = dict(raw)
        stages = {name: StageRecord(**rec) for name, rec in raw.pop("stages", {}).items()}
        return cls(stages=stages, **raw)


# --------------------------------------------------------------------------- #
# State manager
# --------------------------------------------------------------------------- #

class StateManager:
    def __init__(self, projects_dir: Optional[Path] = None):
        ensure_dirs()
        self.projects_dir = Path(projects_dir) if projects_dir else PROJECTS_DIR
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Project lifecycle
    # ------------------------------------------------------------------ #

    def project_dir(self, project_id: str) -> Path:
        if not _ID_PATTERN.match(project_id):
            raise StateError(f"Invalid project id: {project_id!r}")
        return self.projects_dir / project_id

    def exists(self, project_id: str) -> bool:
        return (self.project_dir(project_id) / "state.json").exists()

    @staticmethod
    def _make_project_id(requirement: str) -> str:
        words = re.sub(r"[^a-z0-9]+", " ", requirement.lower()).split()[:4]
        slug = "-".join(words)[:30].strip("-") or "project"
        return f"{slug}-{uuid.uuid4().hex[:6]}"

    def create_project(
        self,
        requirement: str,
        project_id: Optional[str] = None,
        stages: tuple = STAGES,
    ) -> ProjectState:
        """Create a new project: folders, state.json and requirements.md."""
        with self._lock:
            project_id = project_id or self._make_project_id(requirement)
            if self.exists(project_id):
                raise StateError(f"Project {project_id!r} already exists")

            root = self.project_dir(project_id)
            root.mkdir(parents=True, exist_ok=True)
            for sub in SUBDIRS:
                (root / sub).mkdir(exist_ok=True)

            state = ProjectState(
                project_id=project_id,
                requirement=requirement,
                stage_order=list(stages),
                stages={name: StageRecord() for name in stages},
            )
            self.save(state)
            self.write_artifact(state, "requirements.md", f"# Requirements\n\n{requirement}\n")
            logger.info("Created project %s", project_id)
            return state

    def load_project(self, project_id: str) -> ProjectState:
        """Load a project. Stages left 'running' by a crash are reset to pending."""
        with self._lock:
            path = self.project_dir(project_id) / "state.json"
            if not path.exists():
                raise ProjectNotFound(f"No project {project_id!r} in {self.projects_dir}")
            try:
                state = ProjectState.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, TypeError) as e:
                raise StateError(f"state.json for {project_id!r} is corrupted: {e}") from e

            recovered = False
            for name, rec in state.stages.items():
                if rec.status == RUNNING:
                    rec.status = PENDING
                    rec.error = "interrupted (process stopped while this stage was running)"
                    recovered = True
                    logger.warning("[%s] stage %r was interrupted; will re-run", project_id, name)
            if recovered:
                self.save(state)
            return state

    def list_projects(self) -> list[dict]:
        """Short summary of every project, newest first."""
        summaries = []
        for path in self.projects_dir.glob("*/state.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                summaries.append({
                    "project_id": raw["project_id"],
                    "status": raw.get("status"),
                    "current_stage": raw.get("current_stage"),
                    "updated_at": raw.get("updated_at"),
                    "requirement": raw.get("requirement", "")[:80],
                })
            except (OSError, ValueError, KeyError):
                logger.warning("Skipping unreadable project state: %s", path)
        return sorted(summaries, key=lambda s: s.get("updated_at") or "", reverse=True)

    def delete_project(self, project_id: str) -> None:
        with self._lock:
            root = self.project_dir(project_id)
            if root.exists():
                shutil.rmtree(root)
                logger.info("Deleted project %s", project_id)

    def set_status(self, state: ProjectState, status: str) -> None:
        with self._lock:
            state.status = status
            self.save(state)

    def save(self, state: ProjectState) -> None:
        """Atomic save: write to a temp file, then replace."""
        with self._lock:
            state.updated_at = _now()
            path = self.project_dir(state.project_id) / "state.json"
            self._atomic_write(path, json.dumps(state.to_dict(), indent=2, ensure_ascii=False, default=str))

    # ------------------------------------------------------------------ #
    # Stage tracking
    # ------------------------------------------------------------------ #

    def _rec(self, state: ProjectState, stage: str) -> StageRecord:
        if stage not in state.stages:
            state.stages[stage] = StageRecord()      # allow ad-hoc stages
            state.stage_order.append(stage)
        return state.stages[stage]

    def start_stage(self, state: ProjectState, stage: str) -> None:
        with self._lock:
            rec = self._rec(state, stage)
            rec.status = RUNNING
            rec.started_at = _now()
            rec.finished_at = None
            rec.error = None
            rec.attempts += 1
            state.current_stage = stage
            state.status = PROJECT_RUNNING
            self.save(state)
            logger.info("[%s] stage %r started (attempt %d)", state.project_id, stage, rec.attempts)

    def complete_stage(self, state: ProjectState, stage: str, data: Optional[dict] = None) -> None:
        with self._lock:
            rec = self._rec(state, stage)
            rec.status = COMPLETED
            rec.finished_at = _now()
            rec.error = None
            if data:
                rec.data.update(data)
            if self.next_stage(state) is None:
                state.status = PROJECT_COMPLETED
                state.current_stage = None
            self.save(state)
            logger.info("[%s] stage %r completed", state.project_id, stage)

    def fail_stage(self, state: ProjectState, stage: str, error: str) -> None:
        with self._lock:
            rec = self._rec(state, stage)
            rec.status = FAILED
            rec.finished_at = _now()
            rec.error = error
            state.status = PROJECT_FAILED
            self.save(state)
            logger.error("[%s] stage %r failed: %s", state.project_id, stage, error)

    def skip_stage(self, state: ProjectState, stage: str, reason: str = "") -> None:
        with self._lock:
            rec = self._rec(state, stage)
            rec.status = SKIPPED
            rec.finished_at = _now()
            rec.data["skip_reason"] = reason
            if self.next_stage(state) is None:
                state.status = PROJECT_COMPLETED
                state.current_stage = None
            self.save(state)
            logger.info("[%s] stage %r skipped: %s", state.project_id, stage, reason)

    def next_stage(self, state: ProjectState) -> Optional[str]:
        """First stage (in order) that is not completed/skipped. None when everything is done."""
        for name in state.stage_order:
            if state.stages[name].status not in DONE_STATUSES:
                return name
        return None

    def is_complete(self, state: ProjectState) -> bool:
        return self.next_stage(state) is None

    def stage_data(self, state: ProjectState, stage: str) -> dict:
        return dict(state.stages[stage].data) if stage in state.stages else {}

    @contextmanager
    def stage(self, state: ProjectState, name: str) -> Iterator[dict]:
        """
        Run a stage with automatic bookkeeping:
            with sm.stage(state, "hld") as out:
                out["model"] = "gpt"
        Marks it completed on success (saving `out` as stage data), or failed (and re-raises).
        """
        self.start_stage(state, name)
        out: dict = {}
        try:
            yield out
        except BaseException as e:      # includes KeyboardInterrupt: still record, then re-raise
            self.fail_stage(state, name, f"{type(e).__name__}: {e}")
            raise
        else:
            self.complete_stage(state, name, out or None)

    # ------------------------------------------------------------------ #
    # Shared data and review loop
    # ------------------------------------------------------------------ #

    def set_data(self, state: ProjectState, key: str, value: Any) -> None:
        with self._lock:
            state.data[key] = value
            self.save(state)

    def get_data(self, state: ProjectState, key: str, default: Any = None) -> Any:
        return state.data.get(key, default)

    def record_review(
        self,
        state: ProjectState,
        passed: bool,
        findings: Optional[list] = None,
        reviewer_model: Optional[str] = None,
        coder_model: Optional[str] = None,
    ) -> int:
        """Log one review round and return the new iteration number."""
        with self._lock:
            state.review_iteration += 1
            state.review_history.append({
                "iteration": state.review_iteration,
                "timestamp": _now(),
                "passed": passed,
                "findings": findings or [],
                "reviewer_model": reviewer_model,
                "coder_model": coder_model,
            })
            self.save(state)
            return state.review_iteration

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #

    def _resolve(self, state: ProjectState, name: str) -> Path:
        """Map an artifact name to a path inside the project folder (blocks path traversal)."""
        root = self.project_dir(state.project_id).resolve()
        path = (root / name).resolve()
        if not name or not path.is_relative_to(root) or path == root:
            raise InvalidArtifactPath(f"Artifact path escapes the project folder: {name!r}")
        rel = path.relative_to(root)
        if rel.as_posix() in RESERVED_NAMES or rel.parts[0] in RESERVED_DIRS:
            raise InvalidArtifactPath(f"Reserved artifact path: {name!r}")
        return path

    def artifact_path(self, state: ProjectState, name: str) -> Path:
        return self._resolve(state, name)

    def write_artifact(
        self,
        state: ProjectState,
        name: str,
        content: Union[str, dict, list],
        keep_history: bool = True,
    ) -> Path:
        """
        Write an artifact. dict/list content is saved as JSON. If the artifact already
        exists and keep_history is True, the old version is archived under history/.
        """
        with self._lock:
            path = self._resolve(state, name)
            path.parent.mkdir(parents=True, exist_ok=True)

            if keep_history and path.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                archive = self.project_dir(state.project_id) / "history" / f"{stamp}__{name.replace('/', '__')}"
                shutil.copy2(path, archive)

            text = content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False, default=str)
            self._atomic_write(path, text)

            state.artifacts[name] = path.relative_to(self.project_dir(state.project_id).resolve()).as_posix()
            self.save(state)
            return path

    def read_artifact(self, state: ProjectState, name: str, default: Optional[str] = None) -> Optional[str]:
        path = self._resolve(state, name)
        return path.read_text(encoding="utf-8") if path.exists() else default

    def read_json_artifact(self, state: ProjectState, name: str, default: Any = None) -> Any:
        text = self.read_artifact(state, name)
        if text is None:
            return default
        try:
            return json.loads(text)
        except ValueError:
            logger.warning("Artifact %r is not valid JSON", name)
            return default

    def list_artifacts(self, state: ProjectState) -> list[str]:
        return sorted(state.artifacts)

    def artifact_history(self, state: ProjectState, name: str) -> list[Path]:
        """Archived older versions of an artifact, oldest first."""
        hist = self.project_dir(state.project_id) / "history"
        return sorted(hist.glob(f"*__{name.replace('/', '__')}"))

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #

    def checkpoint(self, state: ProjectState, label: str) -> str:
        """Snapshot the full state under a label. Returns the checkpoint file name."""
        with self._lock:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "checkpoint"
            file_name = f"{len(state.checkpoints) + 1:03d}_{safe}.json"
            path = self.project_dir(state.project_id) / "checkpoints" / file_name
            self._atomic_write(path, json.dumps(state.to_dict(), indent=2, ensure_ascii=False, default=str))
            state.checkpoints.append(file_name)
            self.save(state)
            logger.info("[%s] checkpoint saved: %s", state.project_id, file_name)
            return file_name

    def restore_checkpoint(self, project_id: str, file_name: str) -> ProjectState:
        """Roll the project's state back to a checkpoint (artifacts on disk are left as they are)."""
        with self._lock:
            path = self.project_dir(project_id) / "checkpoints" / file_name
            if not path.exists():
                raise StateError(f"No checkpoint {file_name!r} for project {project_id!r}")
            state = ProjectState.from_dict(json.loads(path.read_text(encoding="utf-8")))
            all_checkpoints = sorted(p.name for p in (self.project_dir(project_id) / "checkpoints").glob("*.json"))
            state.checkpoints = all_checkpoints
            self.save(state)
            logger.info("[%s] state restored from %s", project_id, file_name)
            return state

    def list_checkpoints(self, state: ProjectState) -> list[str]:
        return list(state.checkpoints)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Self-test: python -m core.state_manager
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile

    from core.logger import setup_logging

    setup_logging("WARNING")

    with tempfile.TemporaryDirectory() as tmp:
        sm = StateManager(projects_dir=Path(tmp))

        # 1. Create
        state = sm.create_project("Build a desktop app that categorises my downloads")
        pid = state.project_id
        root = sm.project_dir(pid)
        assert (root / "state.json").exists() and (root / "benchmark").is_dir()
        assert sm.read_artifact(state, "requirements.md").startswith("# Requirements")
        assert sm.next_stage(state) == "task_analysis"
        print("PASS 1: project created ->", pid)

        # 2. Stage context manager: success
        with sm.stage(state, "task_analysis") as out:
            out["complexity"] = 7.5
        assert state.stages["task_analysis"].status == COMPLETED
        assert sm.stage_data(state, "task_analysis")["complexity"] == 7.5
        print("PASS 2: stage completes with data")

        # 3. Stage context manager: failure, then successful retry
        try:
            with sm.stage(state, "workflow_decision"):
                raise ValueError("model unavailable")
        except ValueError:
            pass
        rec = state.stages["workflow_decision"]
        assert rec.status == FAILED and "model unavailable" in rec.error and state.status == PROJECT_FAILED
        with sm.stage(state, "workflow_decision"):
            pass
        assert rec.status == COMPLETED and rec.attempts == 2
        print("PASS 3: failed stage recorded, retry succeeded (attempts =", rec.attempts, ")")

        # 4. Crash recovery: a stage left 'running' is picked up again after reload
        sm.skip_stage(state, "lite_generation", "task too small")
        sm.start_stage(state, "benchmark")        # ... process "dies" here
        sm2 = StateManager(projects_dir=Path(tmp))
        reloaded = sm2.load_project(pid)
        assert reloaded.stages["benchmark"].status == PENDING
        assert "interrupted" in reloaded.stages["benchmark"].error
        assert sm2.next_stage(reloaded) == "benchmark"
        assert reloaded.stages["task_analysis"].data["complexity"] == 7.5
        state = reloaded
        print("PASS 4: crash recovery resumes at 'benchmark'")

        # 5. Artifacts: text, JSON, history
        sm.write_artifact(state, "architecture.md", "v1 design")
        sm.write_artifact(state, "architecture.md", "v2 design")
        assert sm.read_artifact(state, "architecture.md") == "v2 design"
        assert len(sm.artifact_history(state, "architecture.md")) == 1
        sm.write_artifact(state, "benchmark/results.json", {"gpt": 0.91})
        assert sm.read_json_artifact(state, "benchmark/results.json") == {"gpt": 0.91}
        assert "benchmark/results.json" in sm.list_artifacts(state)
        print("PASS 5: artifacts (text, JSON, history)")

        # 6. Path safety
        for bad in ("../escape.txt", "/etc/passwd", "state.json", "checkpoints/x.json"):
            try:
                sm.write_artifact(state, bad, "x")
                raise SystemExit(f"FAIL 6: {bad!r} was accepted")
            except InvalidArtifactPath:
                pass
        print("PASS 6: unsafe artifact paths rejected")

        # 7. Checkpoint and restore
        cp = sm.checkpoint(state, "before coding")
        sm.set_data(state, "selected_model", "gpt")
        restored = sm.restore_checkpoint(pid, cp)
        assert "selected_model" not in restored.data and cp in restored.checkpoints
        state = restored
        print("PASS 7: checkpoint restore ->", cp)

        # 8. Review loop tracking
        assert sm.record_review(state, False, ["missing input validation"], "claude-sonnet", "gpt") == 1
        assert sm.record_review(state, True, [], "claude-sonnet", "gpt") == 2
        assert state.review_history[0]["findings"] == ["missing input validation"]
        print("PASS 8: review rounds recorded")

        # 9. Project completes when every stage is done
        for name in ("benchmark", "selection", "hld", "coding", "review"):
            sm.complete_stage(state, name)
        assert sm.is_complete(state) and state.status == PROJECT_COMPLETED
        print("PASS 9: project marked completed")

        # 10. Listing and unknown projects
        assert sm.list_projects()[0]["project_id"] == pid
        try:
            sm.load_project("does-not-exist")
            raise SystemExit("FAIL 10")
        except ProjectNotFound:
            pass
        print("PASS 10: list_projects and ProjectNotFound")