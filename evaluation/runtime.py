"""
evaluation/runtime.py

Sandboxed execution of a candidate: how long it takes, how much memory it uses, whether it imports and
runs at all.

    from context.repository import Repository
    from evaluation.runtime import RuntimeRunner, RunSpec

    rt = RuntimeRunner(settings.orchestrator)
    repo = Repository(candidate_dir)

    # 1. Does every module at least import? (feeds scoring.score_correctness(runs=...))
    check = rt.import_check(repo)
    check.runs                  # True / False / None (nothing to import)
    check.failed                # [(module, "ValueError: ...")]
    check.missing_dependencies  # packages the candidate DECLARES but that are not installed here (environment)

    # 2. Time and memory of a workload (median of N fresh runs)
    spec = RunSpec(command=["python", "main.py", "--n", "100000"], stdin=None, expected_stdout="42\\n")
    result = rt.measure(repo, spec, repeats=3)
    result.wall_seconds         # median wall-clock seconds
    result.peak_rss_mb          # peak memory of the whole process tree (None where unsupported)
    result.output_ok            # True/False if expected_stdout was given, else None
    result.metrics(reference_seconds=1.5, reference_mb=200)   # {"runtime": 0.8, "memory": 1.0}

Without absolute references, compare candidates with scoring.relative_to_best() on the raw numbers.

How it measures: the command runs inside a tiny wrapper process (a fresh interpreter) that applies resource
limits (CPU seconds, address space, file size, no core dumps), runs the command, and reports wall time, CPU
time and the peak resident memory of everything the command spawned (`RUSAGE_CHILDREN`, so a fresh wrapper
gives a clean per-run maximum). Wall time is measured around the command only, not interpreter start-up.
On Windows there is no `resource` module: time is still measured, memory is None.

Isolation (same rules as evaluation/tests.py): every repeat runs in a fresh COPY of the workspace; hard timeout
with the whole process group killed; scrubbed environment (no API keys); HOME/TMPDIR in a temp directory.
That limits accidents, not a determined attacker: run the orchestrator in a container for untrusted code.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import OrchestratorSettings
from core.logger import get_logger
from evaluation.scoring import score_memory, score_runtime
from evaluation.static_analysis import CommandResult, Runner, is_test_path, run_command, safe_env

logger = get_logger("evaluation.runtime")

DEFAULT_MEMORY_LIMIT_MB = 4096
OUTPUT_TAIL_CHARS = 2000
MAX_IMPORT_MODULES = 25
PER_MODULE_TIMEOUT = 10.0

# The wrapper runs in its own interpreter so RUSAGE_CHILDREN holds only this one command's numbers.
WRAPPER_SOURCE = r'''
import json, os, subprocess, sys, time
try:
    import resource
except ImportError:
    resource = None
result_path, cpu, mem_mb, stdin_path, out_path, err_path = sys.argv[1:7]
cmd = sys.argv[8:]
cpu, mem_mb = int(cpu), int(mem_mb)

def limits():
    if resource is None:
        return
    try:
        if cpu > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if mem_mb > 0:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 20, 256 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass

info = {"error": "", "returncode": None, "cpu": None, "maxrss": None}
stdin = open(stdin_path, "rb") if stdin_path != "-" else subprocess.DEVNULL
t0 = time.perf_counter()
try:
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.run(cmd, stdin=stdin, stdout=out, stderr=err,
                              preexec_fn=limits if (resource is not None and os.name == "posix") else None)
    info["returncode"] = proc.returncode
except OSError as e:
    info["error"] = "%s: %s" % (type(e).__name__, e)
info["wall"] = time.perf_counter() - t0
if resource is not None:
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    info["cpu"] = ru.ru_utime + ru.ru_stime
    info["maxrss"] = ru.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
with open(result_path, "w") as handle:
    json.dump(info, handle)
'''

IMPORT_SNIPPET = r'''
import importlib, sys
try:
    importlib.import_module(sys.argv[1])
except SystemExit as e:
    sys.stderr.write("ERR:SystemExit||exit %s\n" % (e.code,)); sys.exit(3)
except BaseException as e:
    missing = getattr(e, "name", "") if isinstance(e, ModuleNotFoundError) else ""
    msg = (str(e).splitlines() or [""])[0][:200]
    sys.stderr.write("ERR:%s|%s|%s\n" % (type(e).__name__, missing or "", msg)); sys.exit(3)
'''


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class RunSpec:
    command: list                          # argv; a leading "python" is replaced by the current interpreter
    name: str = "run"
    stdin: Optional[str] = None
    expected_stdout: Optional[str] = None  # compared after stripping trailing whitespace on both sides
    expected_exit: int = 0
    env: Optional[dict] = None             # extra (non-secret) variables
    timeout: Optional[float] = None


@dataclass
class RunResult:
    name: str
    runs: int = 0                          # repeats actually executed
    ran_ok: bool = False                   # every repeat exited with expected_exit
    timed_out: bool = False
    exit_code: Optional[int] = None
    wall_seconds: Optional[float] = None   # median over repeats
    wall_all: list = field(default_factory=list)
    cpu_seconds: Optional[float] = None
    peak_rss_mb: Optional[float] = None    # median of the per-run peaks
    output_ok: Optional[bool] = None       # None when no expected_stdout was given
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""                        # could not run at all
    notes: list = field(default_factory=list)

    @property
    def completed(self) -> bool:
        """Finished with the right exit code and (if checked) the right output."""
        return self.ran_ok and self.output_ok is not False

    def metrics(self, reference_seconds: Optional[float] = None, reference_mb: Optional[float] = None) -> dict:
        """runtime / memory in 0..1. A failed, timed-out or wrong-output run scores 0.0; not run at all -> None."""
        if self.error and not self.runs:
            return {"runtime": None, "memory": None}
        done = self.completed
        return {"runtime": score_runtime(self.wall_seconds, reference_seconds, completed=done),
                "memory": score_memory(self.peak_rss_mb, reference_mb, completed=done)}

    def to_dict(self) -> dict:
        return dict(self.__dict__, completed=self.completed)


@dataclass
class ImportCheck:
    ok: list = field(default_factory=list)
    failed: list = field(default_factory=list)                 # [(module, "ErrorType: message")]
    missing_dependencies: list = field(default_factory=list)   # [(module, name)]: DECLARED by the candidate, not installed here
    timed_out: list = field(default_factory=list)
    skipped: int = 0                                           # modules not checked (over the cap)

    @property
    def runs(self) -> Optional[bool]:
        """True if everything importable imported. False on any real failure/hang. None: nothing to judge."""
        if self.failed or self.timed_out:
            return False
        return True if self.ok else None

    @property
    def ratio(self) -> Optional[float]:
        judged = len(self.ok) + len(self.failed) + len(self.timed_out)
        return len(self.ok) / judged if judged else None

    def summary(self) -> str:
        parts = [f"{len(self.ok)} imported"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed ({self.failed[0][0]}: {self.failed[0][1]})")
        if self.timed_out:
            parts.append(f"{len(self.timed_out)} hung")
        if self.missing_dependencies:
            parts.append(f"{len(self.missing_dependencies)} need uninstalled packages "
                         f"({', '.join(sorted({d for _, d in self.missing_dependencies}))})")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return dict(self.__dict__, runs=self.runs, ratio=self.ratio)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def module_names(files: list, has_src: bool) -> list:
    """Importable module names for repo files, shallowest first. Tests, setup scripts and odd names are skipped."""
    names = []
    for path in files:
        if not path.endswith(".py") or is_test_path(path):
            continue
        parts = path[:-3].split("/")
        if parts[0] == "src" and has_src and len(parts) > 1:
            parts = parts[1:]
        if parts[-1] in ("setup", "conftest", "__main__") or any(p.startswith(".") for p in parts):
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or not all(p.isidentifier() for p in parts):
            continue
        names.append(".".join(parts))
    return sorted(set(names), key=lambda n: (n.count("."), n))


def _local_top_levels(repo_root: Path) -> set:
    """Top-level module/package names that exist inside the workspace (root and src/)."""
    found = set()
    for base in (repo_root, repo_root / "src"):
        if base.is_dir():
            for entry in base.iterdir():
                if entry.suffix == ".py":
                    found.add(entry.stem)
                elif entry.is_dir() and not entry.name.startswith("."):
                    found.add(entry.name)
    return found


IMPORT_ALIASES = {"yaml": "pyyaml", "pil": "pillow", "cv2": "opencv-python", "sklearn": "scikit-learn",
                  "bs4": "beautifulsoup4", "dotenv": "python-dotenv", "jwt": "pyjwt", "dateutil": "python-dateutil",
                  "attr": "attrs", "serial": "pyserial", "magic": "python-magic", "docx": "python-docx"}


def _normalise_package(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(".", "_")


def declared_dependencies(root: Path) -> set:
    """
    Package names the candidate declares (requirements*.txt lines; plus every word in pyproject.toml /
    setup.py / setup.cfg / Pipfile, which is cruder but format-independent). Normalised: lowercase, `_`.
    """
    declared: set = set()
    try:
        for path in sorted(root.glob("requirements*.txt")) + sorted(root.glob("requirements/*.txt")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.split("#", 1)[0].strip()
                if line and not line.startswith("-"):
                    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
                    if match:
                        declared.add(_normalise_package(match.group(0)))
        for name in ("pyproject.toml", "setup.py", "setup.cfg", "Pipfile"):
            path = root / name
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                declared.update(_normalise_package(w) for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", text))
    except OSError:
        pass
    return declared


def is_declared(import_name: str, declared: set) -> bool:
    key = _normalise_package(import_name)
    alias = _normalise_package(IMPORT_ALIASES.get(key, ""))
    return key in declared or bool(alias and alias in declared)


def _tail(path: Path) -> str:
    try:
        return path.read_bytes()[-OUTPUT_TAIL_CHARS:].decode("utf-8", "replace")
    except OSError:
        return ""


def _same_output(actual: str, expected: str) -> bool:
    return actual.rstrip() == expected.rstrip()


def _call(runner: Runner, argv: list, cwd: str, timeout: float, env: dict) -> CommandResult:
    try:
        return runner(argv, cwd, timeout, env)           # type: ignore[call-arg]
    except TypeError:
        return runner(argv, cwd, timeout)


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #

class RuntimeRunner:
    def __init__(self, settings: Optional[OrchestratorSettings] = None, runner: Optional[Runner] = None,
                 memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB):
        self.settings = settings or OrchestratorSettings()
        self.runner: Runner = runner or run_command
        self.memory_limit_mb = memory_limit_mb

    # ---- environment ----------------------------------------------------------- #

    @staticmethod
    def _env(root: Path, scratch: Path, extra: Optional[dict] = None) -> dict:
        paths = [str(root)] + ([str(root / "src")] if (root / "src").is_dir() else [])
        env = {"PYTHONPATH": os.pathsep.join(paths), "HOME": str(scratch), "TMPDIR": str(scratch),
               "TEMP": str(scratch), "TMP": str(scratch)}
        env.update(extra or {})
        return safe_env(env)

    # ---- import check ------------------------------------------------------------ #

    def import_check(self, repo: Any, modules: Optional[list] = None,
                     per_module_timeout: float = PER_MODULE_TIMEOUT) -> ImportCheck:
        """Import each module in its own subprocess (in a copy of the workspace). Never raises."""
        check = ImportCheck()
        try:
            with tempfile.TemporaryDirectory(prefix="orch-run-") as tmp:
                work, scratch = Path(tmp) / "work", Path(tmp) / "scratch"
                scratch.mkdir()
                copy = repo.copy_to(work)
                root = Path(copy.root)
                env = self._env(root, scratch)
                names = modules if modules is not None else module_names(
                    copy.list_files(include=["*.py"]), (root / "src").is_dir())
                check.skipped = max(0, len(names) - MAX_IMPORT_MODULES)
                local = _local_top_levels(root)
                declared = declared_dependencies(root)
                for name in names[:MAX_IMPORT_MODULES]:
                    run = _call(self.runner, [sys.executable, "-B", "-c", IMPORT_SNIPPET, name], str(root),
                                min(per_module_timeout, float(self.settings.sandbox_timeout_seconds)), env)
                    self._classify_import(check, name, run, local, declared)
        except Exception as e:                           # harness problems must not crash the orchestrator
            logger.error("Import check failed: %s: %s", type(e).__name__, e)
            check.failed.append(("<harness>", f"{type(e).__name__}: {e}"[:200]))
        return check

    @staticmethod
    def _classify_import(check: ImportCheck, name: str, run: CommandResult, local: set, declared: set) -> None:
        if run.error:
            check.failed.append((name, f"could not start: {run.error}"[:200]))
        elif run.timed_out:
            check.timed_out.append(name)
        elif run.returncode == 0:
            check.ok.append(name)
        else:
            line = next((ln for ln in reversed((run.stderr or "").splitlines()) if ln.startswith("ERR:")), "")
            kind, _, rest = line[4:].partition("|")
            missing, _, message = rest.partition("|")
            top = missing.split(".")[0]
            if kind == "ModuleNotFoundError" and top and top not in local and is_declared(top, declared):
                check.missing_dependencies.append((name, top))        # their requirement, our environment
            elif kind == "ModuleNotFoundError" and top and top not in local:
                check.failed.append((name, f"ModuleNotFoundError: {top!r} is not installed and not declared "
                                           f"in requirements (hallucinated module or missing requirement)"))
            else:
                text = f"{kind}: {message}" if kind else f"exit {run.returncode}: {(run.stderr or '').strip()[-150:]}"
                check.failed.append((name, text[:200]))

    # ---- measuring a workload ----------------------------------------------------- #

    def measure(self, repo: Any, spec: RunSpec, repeats: int = 3) -> RunResult:
        """Run the command `repeats` times (fresh copy each time) and report medians. Never raises."""
        result = RunResult(name=spec.name)
        timeout = float(spec.timeout or self.settings.sandbox_timeout_seconds)
        walls, cpus, rss = [], [], []
        try:
            command = [sys.executable if c == "python" and i == 0 else c for i, c in enumerate(spec.command)]
            for attempt in range(max(1, repeats)):
                one = self._one_run(repo, spec, command, timeout)
                if one["error"]:
                    result.error = one["error"]
                    break
                result.runs += 1
                result.exit_code = one["returncode"]
                result.stdout_tail, result.stderr_tail = one["stdout"], one["stderr"]
                if one["timed_out"]:
                    result.timed_out = True
                    result.notes.append(f"run {attempt + 1} exceeded {timeout:.0f}s and was killed")
                    break
                walls.append(one["wall"])
                if one["cpu"] is not None:
                    cpus.append(one["cpu"])
                if one["maxrss"] is not None:
                    rss.append(one["maxrss"] / (1024 * 1024))
                if one["returncode"] != spec.expected_exit:
                    if one["returncode"] is not None and one["returncode"] < 0:
                        result.notes.append(f"killed by signal {-one['returncode']} (a resource limit?)")
                    break                                   # a failing command will not get better on repeat
                if spec.expected_stdout is not None and not _same_output(one["stdout"], spec.expected_stdout):
                    result.output_ok = False
                    break
                if spec.expected_stdout is not None:
                    result.output_ok = True
        except Exception as e:
            logger.error("Measuring %s failed: %s: %s", spec.name, type(e).__name__, e)
            result.error = f"harness error: {type(e).__name__}: {e}"[:200]

        result.wall_all = [round(w, 4) for w in walls]
        if walls:
            result.wall_seconds = round(statistics.median(walls), 4)
        if cpus:
            result.cpu_seconds = round(statistics.median(cpus), 4)
        if rss:
            result.peak_rss_mb = round(statistics.median(rss), 2)
        elif walls and os.name != "posix":
            result.notes.append("memory not measured on this platform")
        result.ran_ok = (bool(walls) and not result.timed_out and not result.error
                         and result.exit_code == spec.expected_exit and len(walls) == max(1, repeats))
        return result

    def _one_run(self, repo: Any, spec: RunSpec, command: list, timeout: float) -> dict:
        out = {"error": "", "timed_out": False, "returncode": None, "wall": None, "cpu": None, "maxrss": None,
               "stdout": "", "stderr": ""}
        with tempfile.TemporaryDirectory(prefix="orch-run-") as tmp:
            work, scratch = Path(tmp) / "work", Path(tmp) / "scratch"
            scratch.mkdir()
            copy = repo.copy_to(work)
            root = Path(copy.root)
            wrapper = scratch / "wrapper.py"
            wrapper.write_text(WRAPPER_SOURCE, encoding="utf-8")
            stdin_path = "-"
            if spec.stdin is not None:
                stdin_file = scratch / "stdin.txt"
                stdin_file.write_text(spec.stdin, encoding="utf-8")
                stdin_path = str(stdin_file)
            result_file, out_file, err_file = scratch / "result.json", scratch / "out.txt", scratch / "err.txt"
            argv = [sys.executable, "-B", str(wrapper), str(result_file), str(int(timeout) + 2),
                    str(self.memory_limit_mb), stdin_path, str(out_file), str(err_file), "--"] + command
            run = _call(self.runner, argv, str(root), timeout + 5, self._env(root, scratch, spec.env))
            out["stdout"], out["stderr"] = _tail(out_file), _tail(err_file)
            if run.error:
                out["error"] = f"could not start: {run.error}"[:200]
            elif run.timed_out:
                out["timed_out"] = True
            else:
                try:
                    info = json.loads(result_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    out["error"] = f"measurement wrapper failed (exit {run.returncode}): {(run.stderr or '').strip()[-150:]}"
                    return out
                if info.get("error"):
                    out["error"] = f"could not run command: {info['error']}"[:200]
                out["returncode"] = info.get("returncode")
                out["wall"], out["cpu"], out["maxrss"] = info.get("wall"), info.get("cpu"), info.get("maxrss")
        return out


# --------------------------------------------------------------------------- #
# Self-test:  python -m evaluation.runtime      (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from context.repository import Repository
    from core.logger import setup_logging

    setup_logging("ERROR")
    tmp = tempfile.TemporaryDirectory()
    POSIX = os.name == "posix"

    def make_repo(name, files):
        repo = Repository(Path(tmp.name) / name)
        if files:
            repo.write_files(files)
        return repo

    # 1. Module-name discovery
    names = module_names(["main.py", "pkg/__init__.py", "pkg/util.py", "src/lib/core.py", "tests/test_x.py",
                          "setup.py", "bad-name.py", "pkg/__main__.py", "conftest.py", ".hidden/x.py", "a/b/c.py"], True)
    assert names == ["main", "pkg", "lib.core", "pkg.util", "a.b.c"], names
    assert module_names(["src/lib/core.py"], False) == ["src.lib.core"]
    print("PASS 1: module names from files (src/ layout, tests/setup/odd names skipped) ->", names)

    # 2. Import check: good, broken, missing dependency, hang, syntax error, SystemExit
    repo = make_repo("imports", {
        "good.py": "VALUE = 1\n",
        "pkg/__init__.py": "from .util import helper\n",
        "pkg/util.py": "def helper():\n    return 1\n",
        "src/lib/__init__.py": "",
        "src/lib/core.py": "from lib import missing_name_here\n",
        "needs_dep.py": "import some_third_party_pkg_xyz\n",
        "requirements.txt": "# deps\nsome-third-party-pkg-xyz>=1.0  # pinned\n",
        "halluc.py": "from utils.helpers import f\n",
        "boom.py": "raise ValueError('bad config')\n",
        "syntax.py": "def (:\n",
        "exits.py": "import sys\nsys.exit(2)\n",
        "hang.py": "import time\ntime.sleep(60)\n",
        "tests/test_good.py": "raise RuntimeError('tests are not imported')\n",
    })
    t0 = time.monotonic()
    chk = RuntimeRunner().import_check(repo, per_module_timeout=2)
    failed = dict(chk.failed)
    assert set(chk.ok) == {"good", "lib", "pkg", "pkg.util"}, chk.ok
    assert "ValueError: bad config" in failed["boom"] and "SyntaxError" in failed["syntax"]
    assert "SystemExit" in failed["exits"] and "ImportError" in failed["lib.core"]
    assert chk.missing_dependencies == [("needs_dep", "some_third_party_pkg_xyz")]      # declared -> environment issue
    assert "not declared" in failed["halluc"]                                           # undeclared -> candidate's bug
    assert chk.timed_out == ["hang"] and chk.runs is False and 0 < chk.ratio < 1
    assert time.monotonic() - t0 < 30
    print("PASS 2: import check ->", chk.summary())

    # 3. A clean project: runs True; empty project: None; local broken import is a failure, not a 'missing dependency'
    chk = RuntimeRunner().import_check(make_repo("clean", {"a.py": "x = 1\n", "b.py": "import a\n"}))
    assert chk.runs is True and chk.ratio == 1.0 and chk.failed == []
    assert RuntimeRunner().import_check(make_repo("none", {"README.md": "hi\n"})).runs is None
    chk = RuntimeRunner().import_check(make_repo("localmiss", {"a.py": "import b\n", "b.py": "import a_missing_local\n"}))
    assert dict(chk.failed).keys() == {"a", "b"} and chk.runs is False and not chk.missing_dependencies
    assert is_declared("yaml", {"pyyaml"}) and is_declared("PIL", {"pillow"}) and not is_declared("flask", {"django"})
    assert declared_dependencies(make_repo("deps", {"requirements.txt": "Flask==2.0\n-r other.txt\nrequests[socks]>=2\n"}).root) \
        == {"flask", "requests"}
    print("PASS 3: clean -> True, nothing to import -> None, broken chain -> False")

    # 4. measure(): wall time, CPU, memory, medians, fresh copies
    repo = make_repo("work", {
        "hello.py": "print('hi')\n",
        "sleep.py": "import time\ntime.sleep(0.4)\nprint('done')\n",
        "hog.py": "data = bytearray(150 * 1024 * 1024)\nfor i in range(0, len(data), 4096):\n    data[i] = 1\nprint('ok')\n",
        "stateful.py": "import os\nif os.path.exists('marker'):\n    raise SystemExit(7)\nopen('marker', 'w').write('x')\nprint('first')\n",
        "echo.py": "import sys\nprint(sys.stdin.read().upper(), end='')\n",
        "fail.py": "import sys\nprint('about to fail')\nsys.stderr.write('trace here')\nsys.exit(3)\n",
        "env.py": "import os\nprint(os.environ.get('ANTHROPIC_API_KEY', 'absent'))\n",
    })
    rt = RuntimeRunner()
    r = rt.measure(repo, RunSpec(["python", "sleep.py"], expected_stdout="done\n"), repeats=3)
    assert r.ran_ok and r.output_ok is True and r.runs == 3 and len(r.wall_all) == 3
    assert 0.35 <= r.wall_seconds < 2.0, r.wall_seconds
    assert r.cpu_seconds is not None or not POSIX
    print("PASS 4: wall time (median of 3) ->", r.wall_seconds, "s; cpu", r.cpu_seconds)

    # 5. Memory: a 150 MB allocation shows up; a trivial script does not
    if POSIX:
        hog = rt.measure(repo, RunSpec(["python", "hog.py"]), repeats=2)
        hello = rt.measure(repo, RunSpec(["python", "hello.py"]), repeats=2)
        assert hog.peak_rss_mb >= 140 and hello.peak_rss_mb < 60, (hog.peak_rss_mb, hello.peak_rss_mb)
        assert hog.metrics(reference_mb=200)["memory"] == 1.0
        assert hog.metrics(reference_mb=70)["memory"] < 0.6
        print(f"PASS 5: memory measured -> hog {hog.peak_rss_mb} MB vs hello {hello.peak_rss_mb} MB")
    else:
        print("PASS 5: memory check skipped on this platform")

    # 6. Failures: exit code, wrong output, stdin, no state leaking between repeats, timeouts
    f = rt.measure(repo, RunSpec(["python", "fail.py"]), repeats=3)
    assert not f.ran_ok and f.exit_code == 3 and f.runs == 1 and "trace here" in f.stderr_tail
    assert f.metrics(reference_seconds=1, reference_mb=100) == {"runtime": 0.0, "memory": 0.0}
    w = rt.measure(repo, RunSpec(["python", "hello.py"], expected_stdout="bye\n"), repeats=2)
    assert w.output_ok is False and not w.completed and w.metrics(reference_seconds=1)["runtime"] == 0.0
    e = rt.measure(repo, RunSpec(["python", "echo.py"], stdin="hello\n", expected_stdout="HELLO\n"), repeats=1)
    assert e.ran_ok and e.output_ok is True
    s = rt.measure(repo, RunSpec(["python", "stateful.py"]), repeats=3)
    assert s.ran_ok and s.runs == 3                                    # each repeat starts from a clean copy
    assert not repo.exists("marker")
    slow = rt.measure(make_repo("slow", {"s.py": "import time\ntime.sleep(60)\n"}), RunSpec(["python", "s.py"], timeout=2))
    assert slow.timed_out and not slow.ran_ok and slow.metrics(reference_seconds=1)["runtime"] == 0.0
    print("PASS 6: exit codes, wrong output, stdin, clean copy per repeat, timeout all handled")

    # 7. Memory limit is enforced (POSIX): a 600 MB allocation under a 300 MB cap fails instead of eating the host
    if POSIX:
        big = make_repo("big", {"b.py": "x = bytearray(600 * 1024 * 1024)\nprint('allocated')\n"})
        capped = RuntimeRunner(memory_limit_mb=300).measure(big, RunSpec(["python", "b.py"]), repeats=1)
        assert not capped.ran_ok and "allocated" not in capped.stdout_tail, capped.to_dict()
        print("PASS 7: memory cap enforced -> exit", capped.exit_code)
    else:
        print("PASS 7: memory cap check skipped on this platform")

    # 8. Environment: API keys never reach the candidate
    os.environ["ANTHROPIC_API_KEY"] = "sk-secret"
    try:
        k = rt.measure(repo, RunSpec(["python", "env.py"], expected_stdout="absent\n"), repeats=1)
    finally:
        del os.environ["ANTHROPIC_API_KEY"]
    assert k.output_ok is True
    print("PASS 8: candidate cannot see API keys")

    # 9. Start failures and harness problems become values, not exceptions; temp dirs cleaned
    nope = rt.measure(repo, RunSpec(["definitely-not-a-binary-xyz"]), repeats=1)
    assert not nope.ran_ok and nope.error and nope.metrics() == {"runtime": None, "memory": None}

    class BadRepo:
        root = Path(tmp.name)

        def copy_to(self, dest):
            raise OSError("disk full")

    bad = rt.measure(BadRepo(), RunSpec(["python", "x.py"]), repeats=1)
    assert "harness error" in bad.error or "disk full" in bad.error
    assert rt.import_check(BadRepo()).runs is False
    assert not list(Path(tempfile.gettempdir()).glob("orch-run-*"))
    json.dumps(bad.to_dict())
    json.dumps(chk.to_dict())
    print("PASS 9: errors are values, temp directories cleaned, results are JSON")