"""
context/repository.py

The project workspace: safe file operations, model-output parsing, snapshots and diffs.

    repo = Repository(project_dir / "implementation")
    repo.write_file("src/app.py", code)
    repo.read_file("src/app.py")
    repo.list_files(include=["*.py"])
    print(repo.tree())

Turning a model's reply into files (see FILE_FORMAT_INSTRUCTIONS for the prompt text):

    result = repo.apply_model_output(response.text)
    result.written        # {"src/app.py": "created", ...}
    result.incomplete     # files cut off mid-way (reply hit the token limit): NOT written
    result.rejected       # [(path, reason)] unsafe or invalid paths: NOT written
    result.placeholders   # {"src/app.py": [(line_no, text)]}: "...rest of code unchanged" style gaps

Review loop support:

    before = repo.snapshot("revision-1")
    ...apply a revision...
    print(repo.diff(before).summary())     # "1 added, 2 modified, 0 deleted"
    repo.restore(before)                   # roll back if the revision made things worse

Safety rules (enforced for every path): relative only, forward slashes, no '..', no drive letters,
no characters Windows forbids, must stay inside the root even through symlinks, and never inside an
ignored location (.git, __pycache__, node_modules, virtualenvs, caches...).
"""

from __future__ import annotations

import codecs
import difflib
import fnmatch
import hashlib
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

from core.logger import get_logger

logger = get_logger("context.repository")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_IGNORE_DIRS = (
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
)
DEFAULT_IGNORE_FILES = ("*.pyc", "*.pyo", ".DS_Store", "Thumbs.db")

MAX_FILE_BYTES = 1_000_000
MAX_FILES = 500
MAX_PATH_LENGTH = 240
SNIFF_BYTES = 8192

_EXTENSIONLESS_NAMES = {"Makefile", "Dockerfile", "Procfile", "LICENSE", "Gemfile", "Jenkinsfile"}

FILE_FORMAT_INSTRUCTIONS = """Return every file in full using exactly this format:

<file path="relative/path/to/file.py">
complete file contents here
</file>

Rules:
- Paths are relative to the project root and use forward slashes.
- Include the COMPLETE contents of every file you create or change. Never write placeholders such as
  "rest of the code unchanged" or "...".
- Do not wrap the <file> blocks in markdown code fences.
- Put any explanation OUTSIDE the <file> blocks."""


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class RepoError(Exception):
    """Base class for repository errors."""


class InvalidPath(RepoError):
    """The path is unsafe or malformed."""


class FileNotInRepo(RepoError):
    pass


class NotTextFile(RepoError):
    pass


class RepoLimitError(RepoError):
    """A size or file-count limit would be exceeded."""


# --------------------------------------------------------------------------- #
# Path handling
# --------------------------------------------------------------------------- #

def normalize_path(raw: str) -> str:
    """Validate a repo-relative path and return it normalised ('a/b/c.py'). Raises InvalidPath."""
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidPath("empty path")
    text = raw.strip().replace("\\", "/")
    if any(ord(c) < 32 for c in text):
        raise InvalidPath("control characters in path")
    if text.endswith("/"):
        raise InvalidPath("path names a directory, not a file")
    if text.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", text):
        raise InvalidPath("absolute paths are not allowed")

    parts = []
    for part in text.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise InvalidPath("'..' is not allowed")
        parts.append(part)
    if not parts:
        raise InvalidPath("empty path")

    normalized = "/".join(parts)
    if len(normalized) > MAX_PATH_LENGTH:
        raise InvalidPath(f"path is longer than {MAX_PATH_LENGTH} characters")
    if re.search(r'[<>:"|?*]', normalized):
        raise InvalidPath("path contains characters that are not allowed (< > : \" | ? *)")
    return normalized


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_text_bytes(data: bytes) -> bool:
    sample = data[:SNIFF_BYTES]
    if b"\0" in sample:
        return False
    try:
        codecs.getincrementaldecoder("utf-8")().decode(sample, final=False)   # tolerates a cut multibyte tail
    except UnicodeDecodeError:
        return False
    return True


def _estimate_tokens(chars: int) -> int:
    return chars // 4


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int
    is_text: bool

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix.lower()


@dataclass
class RepoStats:
    files: int
    text_files: int
    total_bytes: int
    total_lines: int
    estimated_tokens: int
    by_extension: dict = field(default_factory=dict)      # ".py" -> {"files": n, "lines": m}


@dataclass
class Snapshot:
    """Point-in-time copy of the workspace. Text contents are kept (for diffs/restore); binaries only as hashes."""
    label: str
    created_at: str
    hashes: dict                          # path -> sha256 (every file)
    files: dict                           # path -> text content, or None for binary / oversized files

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Snapshot":
        return cls(label=raw["label"], created_at=raw["created_at"],
                   hashes=dict(raw["hashes"]), files=dict(raw["files"]))


@dataclass
class RepoDiff:
    added: list
    modified: list
    deleted: list
    unified: str                          # unified diff of text changes (possibly truncated)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def summary(self) -> str:
        return f"{len(self.added)} added, {len(self.modified)} modified, {len(self.deleted)} deleted"


@dataclass
class ParseResult:
    files: dict = field(default_factory=dict)            # normalised path -> content
    rejected: list = field(default_factory=list)         # [(raw path, reason)]
    incomplete: list = field(default_factory=list)       # paths whose block was cut off
    placeholders: dict = field(default_factory=dict)     # path -> [(line_no, text)]
    warnings: list = field(default_factory=list)
    format: str = "none"                                 # "tags" | "fences" | "none"


@dataclass
class ApplyResult:
    written: dict = field(default_factory=dict)          # path -> "created" | "updated" | "unchanged"
    rejected: list = field(default_factory=list)
    incomplete: list = field(default_factory=list)
    placeholders: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    format: str = "none"

    @property
    def ok(self) -> bool:
        """True when files were written and nothing was cut off or rejected."""
        return bool(self.written) and not self.incomplete and not self.rejected


# --------------------------------------------------------------------------- #
# Model output parsing
# --------------------------------------------------------------------------- #

_OPEN_TAG = re.compile(r"""<file\s+path\s*=\s*(["'])(.+?)\1\s*>""", re.IGNORECASE)
_CLOSE_TAG = re.compile(r"</file\s*>", re.IGNORECASE)
_FENCE_OPEN = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*([\w+.#-]*)\s*$")
_PATH_COMMENT_KEYED = re.compile(
    r"^\s*(?:#|//|--|;|/\*|<!--)\s*(?:file(?:name|path)?|path)\s*[:=]\s*(\S+?)\s*(?:\*/|-->)?\s*$", re.IGNORECASE)
_PATH_COMMENT_BARE = re.compile(r"^\s*(?:#|//|--)\s*([\w.\-/]+\.\w{1,8})\s*$")
_COMMENT_LINE = re.compile(r"^\s*(?:#|//|/\*|\*|--|<!--)")
_PLACEHOLDER_PHRASES = re.compile(
    r"rest of (the )?(file|code|implementation|class|function)|remaining (code|implementation)|existing code|"
    r"(code|file) (remains )?unchanged|unchanged code|same as (before|above)|(implementation|code) (goes )?here|"
    r"your code here|\.\.\. ?(rest|remaining|existing|other|previous)", re.IGNORECASE)


def find_placeholders(content: str) -> list:
    """Comment lines that look like 'the rest of the code goes here': [(line_no, text)]."""
    found = []
    for number, line in enumerate(content.splitlines(), 1):
        if _COMMENT_LINE.match(line) and _PLACEHOLDER_PHRASES.search(line):
            found.append((number, line.strip()[:120]))
    return found


def _unwrap_single_fence(content: str) -> str:
    """If the whole content is ONE fenced block (models often do this inside tags), return its inside."""
    lines = content.strip("\n").split("\n")
    if len(lines) < 2:
        return content
    opening = _FENCE_OPEN.match(lines[0])
    if not opening:
        return content
    fence = opening.group(1)
    closing = re.match(r"^\s{0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", lines[-1])
    if not closing:
        return content
    if any(re.match(r"^\s{0,3}(`{3,}|~{3,})", inner) for inner in lines[1:-1]):
        return content                                      # nested fences: leave untouched
    return "\n".join(lines[1:-1])


def _clean_content(raw: str) -> str:
    content = raw.replace("\r\n", "\n")
    if content.startswith("\n"):
        content = content[1:]
    content = _unwrap_single_fence(content).rstrip()
    return content + "\n" if content else ""


def _label_to_path(line: str) -> Optional[str]:
    """'### src/app.py', '**File: src/app.py**', '`src/app.py`:' -> 'src/app.py' (else None)."""
    text = re.sub(r"^#{1,6}\s*", "", line.strip()).strip("*_` \t:")
    text = re.sub(r"^(?:file(?:name|path)?|path)\s*[:=]\s*", "", text, flags=re.IGNORECASE).strip("*_` \t:")
    if re.fullmatch(r"[\w.\-/\\]+\.[A-Za-z0-9]{1,8}", text) or text in _EXTENSIONLESS_NAMES:
        return text
    return None


def _parse_tags(text: str, result: ParseResult) -> None:
    opens = list(_OPEN_TAG.finditer(text))
    closes = [m.start() for m in _CLOSE_TAG.finditer(text)]
    for index, match in enumerate(opens):
        raw_path = match.group(2).strip()
        next_open = opens[index + 1].start() if index + 1 < len(opens) else len(text)
        close = next((c for c in closes if match.end() <= c < next_open), None)
        if close is None:
            result.incomplete.append(raw_path)             # reply was cut off (or the tag was never closed)
            continue
        _add_file(result, raw_path, text[match.end():close])


def _parse_fences(text: str, result: ParseResult) -> None:
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        opening = _FENCE_OPEN.match(lines[i])
        if not opening:
            i += 1
            continue
        fence = opening.group(1)
        closer = re.compile(r"^\s{0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$")
        j = i + 1
        while j < len(lines) and not closer.match(lines[j]):
            j += 1
        closed = j < len(lines)
        body = lines[i + 1:j]

        raw_path = None
        k = i - 1
        while k >= 0 and not lines[k].strip() and i - k <= 2:
            k -= 1
        if k >= 0 and i - k <= 3:
            raw_path = _label_to_path(lines[k])
        if raw_path is None and body:
            comment = _PATH_COMMENT_KEYED.match(body[0]) or _PATH_COMMENT_BARE.match(body[0])
            if comment:
                raw_path, body = comment.group(1), body[1:]

        if raw_path:
            if closed:
                _add_file(result, raw_path, "\n".join(body))
            else:
                result.incomplete.append(raw_path)
        i = j + 1


def _add_file(result: ParseResult, raw_path: str, raw_content: str) -> None:
    try:
        path = normalize_path(raw_path)
    except InvalidPath as e:
        result.rejected.append((raw_path, str(e)))
        return
    if path in result.files:
        result.warnings.append(f"{path} appears more than once: the last version is used")
    content = _clean_content(raw_content)
    result.files[path] = content
    placeholders = find_placeholders(content)
    if placeholders:
        result.placeholders[path] = placeholders
    else:
        result.placeholders.pop(path, None)


def parse_model_output(text: str) -> ParseResult:
    """
    Extract files from a model reply. Prefers the canonical <file path="..."> format; if none is
    found, falls back to fenced code blocks labelled with a path (a heading line above the fence, or
    a path comment on the block's first line).
    """
    result = ParseResult()
    if not text or not text.strip():
        result.warnings.append("the reply was empty")
        return result

    if _OPEN_TAG.search(text):
        result.format = "tags"
        _parse_tags(text, result)
    else:
        _parse_fences(text, result)
        if result.files or result.incomplete:
            result.format = "fences"
            result.warnings.append("the reply did not use the <file path=...> format: parsed labelled code blocks")

    if result.incomplete:
        result.warnings.append(
            f"{len(result.incomplete)} file(s) were cut off or never closed ({', '.join(result.incomplete)}): "
            "the reply probably hit the token limit; these files were NOT used")
    if result.placeholders:
        result.warnings.append(
            f"placeholder comments found in {', '.join(result.placeholders)}: parts of the code may be missing")
    if not result.files and not result.incomplete and not result.rejected:
        result.warnings.append("no files were found in the reply")
    return result


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #

class Repository:
    def __init__(
        self,
        root: Union[str, Path],
        *,
        ignore_dirs: Iterable = (),
        ignore_files: Iterable = (),
        max_file_bytes: int = MAX_FILE_BYTES,
        max_files: int = MAX_FILES,
        create: bool = True,
    ):
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise RepoError(f"Repository root does not exist: {self.root}")
        self._root = self.root.resolve()
        self._ignore_dirs = set(DEFAULT_IGNORE_DIRS) | set(ignore_dirs)
        self._ignore_files = tuple(DEFAULT_IGNORE_FILES) + tuple(ignore_files)
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"Repository({str(self.root)!r})"

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def _dir_ignored(self, name: str) -> bool:
        return name in self._ignore_dirs or name.endswith(".egg-info")

    def is_ignored(self, rel: str) -> bool:
        parts = rel.split("/")
        if any(self._dir_ignored(p) for p in parts[:-1]):
            return True
        name = parts[-1]
        return name in self._ignore_dirs or any(fnmatch.fnmatch(name, pat) for pat in self._ignore_files)

    def _resolve(self, path: str) -> tuple:
        """Validate a path and return (normalised relative path, absolute Path inside the root)."""
        rel = normalize_path(path)
        if self.is_ignored(rel):
            raise InvalidPath(f"{rel!r} is in an ignored location")
        absolute = (self._root / rel).resolve()           # follows symlinks of existing parts
        if not absolute.is_relative_to(self._root):
            raise InvalidPath(f"{rel!r} escapes the repository")
        return rel, absolute

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def exists(self, path: str) -> bool:
        try:
            return self._resolve(path)[1].is_file()
        except InvalidPath:
            return False

    def _read_bytes(self, path: str) -> tuple:
        rel, absolute = self._resolve(path)
        if not absolute.is_file():
            raise FileNotInRepo(f"No such file: {rel}")
        size = absolute.stat().st_size
        if size > self.max_file_bytes:
            raise RepoLimitError(f"{rel} is {size} bytes (limit {self.max_file_bytes})")
        return rel, absolute.read_bytes()

    def read_file(self, path: str, errors: str = "strict") -> str:
        """Read a text file. Raises FileNotInRepo, NotTextFile (binary or not UTF-8) or RepoLimitError."""
        rel, data = self._read_bytes(path)
        if not _is_text_bytes(data):
            raise NotTextFile(f"{rel} is a binary file")
        try:
            return data.decode("utf-8", errors=errors)
        except UnicodeDecodeError as e:
            raise NotTextFile(f"{rel} is not valid UTF-8") from e

    def read_text_files(self, paths: Optional[Iterable] = None) -> dict:
        """{path: content} for text files (all of them by default). Binary/oversized files are skipped."""
        contents = {}
        for rel in (list(paths) if paths is not None else self.list_files()):
            try:
                contents[normalize_path(rel)] = self.read_file(rel)
            except (RepoError, OSError):
                continue
        return contents

    # ------------------------------------------------------------------ #
    # Listing
    # ------------------------------------------------------------------ #

    def _iter_paths(self):
        for dirpath, dirnames, filenames in os.walk(self._root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames
                                 if not self._dir_ignored(d) and not os.path.islink(os.path.join(dirpath, d)))
            rel_dir = os.path.relpath(dirpath, self._root)
            for name in sorted(filenames):
                if os.path.islink(os.path.join(dirpath, name)):
                    continue
                rel = name if rel_dir == "." else f"{rel_dir}/{name}".replace(os.sep, "/")
                if not self.is_ignored(rel):
                    yield rel

    @staticmethod
    def _matches(rel: str, pattern: str) -> bool:
        """Patterns without '/' match the file name; patterns with '/' match the whole relative path."""
        target = rel if "/" in pattern else rel.rsplit("/", 1)[-1]
        return fnmatch.fnmatch(target, pattern)

    def list_files(self, include: Optional[Iterable] = None, exclude: Optional[Iterable] = None,
                   text_only: bool = False) -> list:
        include, exclude = list(include or []), list(exclude or [])
        result = []
        for rel in self._iter_paths():
            if include and not any(self._matches(rel, p) for p in include):
                continue
            if any(self._matches(rel, p) for p in exclude):
                continue
            if text_only and not self._sniff_text(rel):
                continue
            result.append(rel)
        return sorted(result)

    def _sniff_text(self, rel: str) -> bool:
        try:
            with open(self._root / rel, "rb") as handle:
                return _is_text_bytes(handle.read(SNIFF_BYTES))
        except OSError:
            return False

    def file_infos(self, **filters) -> list:
        infos = []
        for rel in self.list_files(**filters):
            try:
                infos.append(FileInfo(rel, (self._root / rel).stat().st_size, self._sniff_text(rel)))
            except OSError:
                continue
        return infos

    def stats(self) -> RepoStats:
        files = text_files = total_bytes = total_lines = 0
        by_extension: dict = {}
        for info in self.file_infos():
            files += 1
            total_bytes += info.size
            if not info.is_text:
                continue
            text_files += 1
            try:
                lines = len(self.read_file(info.path).splitlines())
            except RepoError:
                lines = 0
            total_lines += lines
            bucket = by_extension.setdefault(info.suffix or "(none)", {"files": 0, "lines": 0})
            bucket["files"] += 1
            bucket["lines"] += lines
        return RepoStats(files, text_files, total_bytes, total_lines,
                         _estimate_tokens(total_bytes), by_extension)

    def tree(self, max_depth: Optional[int] = None, max_entries: int = 300, root_label: Optional[str] = None) -> str:
        """A directory tree as text, directories first."""
        nested: dict = {}
        for rel in self.list_files():
            node = nested
            parts = rel.split("/")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = None

        lines = [f"{root_label or self.root.name or '.'}/"]

        def render(node: dict, prefix: str, depth: int) -> None:
            entries = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))
            for index, (name, child) in enumerate(entries):
                last = index == len(entries) - 1
                branch = "└── " if last else "├── "
                if child is None:
                    lines.append(f"{prefix}{branch}{name}")
                elif max_depth is not None and depth >= max_depth:
                    lines.append(f"{prefix}{branch}{name}/ ...")
                else:
                    lines.append(f"{prefix}{branch}{name}/")
                    render(child, prefix + ("    " if last else "│   "), depth + 1)

        render(nested, "", 1)
        if len(lines) > max_entries:
            omitted = len(lines) - max_entries
            lines = lines[:max_entries] + [f"... ({omitted} more entries)"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def _validate_write(self, path: str, content: str) -> tuple:
        rel, absolute = self._resolve(path)
        data = content.encode("utf-8")
        if len(data) > self.max_file_bytes:
            raise RepoLimitError(f"{rel} would be {len(data)} bytes (limit {self.max_file_bytes})")
        if absolute.is_dir():
            raise InvalidPath(f"{rel!r} is a directory")
        return rel, absolute, data

    def _write_validated(self, absolute: Path, data: bytes) -> str:
        existed = absolute.is_file()
        if existed and absolute.read_bytes() == data:
            return "unchanged"
        try:
            absolute.parent.mkdir(parents=True, exist_ok=True)
            tmp = absolute.with_name(f"{absolute.name}.tmp{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, absolute)
        except (FileExistsError, NotADirectoryError) as e:
            raise InvalidPath(f"cannot write {absolute.name}: a parent path is a file") from e
        except OSError as e:
            raise RepoError(f"cannot write {absolute.name}: {e}") from e
        return "updated" if existed else "created"

    def write_file(self, path: str, content: str, overwrite: bool = True) -> str:
        """Write one file atomically. Returns 'created', 'updated' or 'unchanged'."""
        return self.write_files({path: content}, overwrite=overwrite)[normalize_path(path)]

    def write_files(self, files: Mapping, overwrite: bool = True) -> dict:
        """
        Write several files. Everything is validated first, so a bad path, an oversized file or a
        limit violation raises before ANY file is touched.
        """
        with self._lock:
            prepared = {}
            for path, content in files.items():
                rel, absolute, data = self._validate_write(path, content)
                if not overwrite and absolute.exists():
                    raise RepoError(f"{rel} already exists")
                prepared[rel] = (absolute, data)

            new_files = [rel for rel, (absolute, _) in prepared.items() if not absolute.is_file()]
            if new_files:
                existing = sum(1 for _ in self._iter_paths())
                if existing + len(new_files) > self.max_files:
                    raise RepoLimitError(f"writing {len(new_files)} new file(s) would exceed the "
                                         f"limit of {self.max_files} files (currently {existing})")

            return {rel: self._write_validated(absolute, data) for rel, (absolute, data) in prepared.items()}

    def delete_file(self, path: str) -> bool:
        """Delete a file (and any directories left empty). Returns False if it did not exist."""
        with self._lock:
            rel, absolute = self._resolve(path)
            if not absolute.is_file():
                return False
            absolute.unlink()
            parent = absolute.parent
            while parent != self._root and parent.is_relative_to(self._root):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
            return True

    # ------------------------------------------------------------------ #
    # Model output
    # ------------------------------------------------------------------ #

    def apply_model_output(self, text: str, overwrite: bool = True) -> ApplyResult:
        """Parse a model reply and write its (complete, valid) files. See the module docstring."""
        parsed = parse_model_output(text)
        result = ApplyResult(rejected=list(parsed.rejected), incomplete=list(parsed.incomplete),
                             placeholders=dict(parsed.placeholders), warnings=list(parsed.warnings),
                             format=parsed.format)

        acceptable = {}
        for path, content in parsed.files.items():
            try:
                self._validate_write(path, content)
            except RepoError as e:
                result.rejected.append((path, str(e)))
                result.placeholders.pop(path, None)
                continue
            acceptable[path] = content

        if acceptable:
            result.written = self.write_files(acceptable, overwrite=overwrite)
        return result

    # ------------------------------------------------------------------ #
    # Snapshots, diffs, restore, copies
    # ------------------------------------------------------------------ #

    def snapshot(self, label: str = "") -> Snapshot:
        hashes, contents = {}, {}
        for rel in self.list_files():
            try:
                data = (self._root / rel).read_bytes()
            except OSError:
                continue
            hashes[rel] = _sha256(data)
            text = None
            if len(data) <= self.max_file_bytes and _is_text_bytes(data):
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
            contents[rel] = text
        return Snapshot(label=label, created_at=datetime.now(timezone.utc).isoformat(),
                        hashes=hashes, files=contents)

    def diff(self, since: Snapshot, context_lines: int = 3, max_chars: int = 20_000) -> RepoDiff:
        """What changed in the workspace compared with an earlier snapshot."""
        now = self.snapshot()
        added = sorted(p for p in now.hashes if p not in since.hashes)
        deleted = sorted(p for p in since.hashes if p not in now.hashes)
        modified = sorted(p for p in now.hashes if p in since.hashes and now.hashes[p] != since.hashes[p])

        chunks = []
        for path in sorted(added + deleted + modified):
            old, new = since.files.get(path), now.files.get(path)
            if path in added:
                old = ""
            if path in deleted:
                new = ""
            if old is None or new is None:
                chunks.append(f"Binary or oversized file changed: {path}\n")
                continue
            chunks.extend(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{path}" if path not in added else "/dev/null",
                tofile=f"b/{path}" if path not in deleted else "/dev/null", n=context_lines))
        unified = "".join(c if c.endswith("\n") else c + "\n" for c in chunks)
        if len(unified) > max_chars:
            unified = unified[:max_chars] + "\n... (diff truncated)\n"
        return RepoDiff(added, modified, deleted, unified)

    def restore(self, snapshot: Snapshot, delete_extra: bool = True) -> dict:
        """
        Roll the workspace back to a snapshot. Text files are restored exactly; files added since are
        deleted (when delete_extra). Binary/oversized files can't be restored and are reported as skipped.
        """
        with self._lock:
            restored, deleted, skipped = 0, 0, []
            for path, content in snapshot.files.items():
                if content is None:
                    if not self.exists(path) or _sha256((self._root / path).read_bytes()) != snapshot.hashes[path]:
                        skipped.append(path)
                    continue
                if self.write_file(path, content) != "unchanged":
                    restored += 1
            if delete_extra:
                for path in self.list_files():
                    if path not in snapshot.hashes:
                        self.delete_file(path)
                        deleted += 1
            return {"restored": restored, "deleted": deleted, "skipped": skipped}

    def copy_to(self, destination: Union[str, Path], overwrite: bool = False, **kwargs) -> "Repository":
        """Copy every (non-ignored) file into a new workspace, e.g. one per benchmark candidate."""
        destination = Path(destination)
        if destination.exists() and any(destination.iterdir()) and not overwrite:
            raise RepoError(f"Destination is not empty: {destination}")
        target = Repository(destination, max_file_bytes=self.max_file_bytes, max_files=self.max_files, **kwargs)
        for rel in self.list_files():
            data = (self._root / rel).read_bytes()
            out = target._root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        return target


# --------------------------------------------------------------------------- #
# Self-test (no network, no API keys): python -m context.repository
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile

    from core.logger import setup_logging

    setup_logging("ERROR")

    def raises(exc, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except exc:
            return True
        return False

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        repo = Repository(base / "proj")

        # 1. Path validation
        assert normalize_path("./src\\app.py") == "src/app.py" and normalize_path("a//b/c.py") == "a/b/c.py"
        for bad in ("", "  ", "../x.py", "a/../../x", "/etc/passwd", "C:/x.py", "C:\\x.py", "~/x", "dir/",
                    "a\0b", 'we"ird.py', "a" * 300):
            assert raises(InvalidPath, normalize_path, bad), bad
        print("PASS 1: path validation")

        # 2. Write / read / statuses / delete with directory pruning
        assert repo.write_file("src/app.py", "print(1)\n") == "created"
        assert repo.write_file("src/app.py", "print(1)\n") == "unchanged"
        assert repo.write_file("src/app.py", "print(2)\n") == "updated"
        assert repo.read_file("src/app.py") == "print(2)\n" and repo.exists("src/app.py")
        assert not repo.exists("../nope") and not repo.exists("missing.py")
        assert raises(FileNotInRepo, repo.read_file, "missing.py")
        assert raises(RepoError, repo.write_file, "src/app.py", "x", overwrite=False)
        assert repo.delete_file("src/app.py") and not (repo.root / "src").exists()
        assert repo.delete_file("src/app.py") is False
        print("PASS 2: write/read/delete, statuses, empty directories pruned")

        # 3. Safety: ignored locations, escapes, symlinks, file-as-directory, atomic validation
        assert raises(InvalidPath, repo.write_file, ".git/config", "x")
        assert raises(InvalidPath, repo.write_file, "node_modules/x/index.js", "x")
        assert raises(InvalidPath, repo.write_file, "pkg/__pycache__/a.pyc", "x")
        assert raises(InvalidPath, repo.write_file, "../outside.txt", "x")
        repo.write_file("a.txt", "x")
        assert raises(InvalidPath, repo.write_file, "a.txt/b.txt", "x")
        assert not (base / "outside.txt").exists()
        before = set(repo.list_files())
        assert raises(InvalidPath, repo.write_files, {"ok.py": "x", "../bad.py": "y"})
        assert set(repo.list_files()) == before, "a bad path must not leave partial writes behind"
        outside = base / "secret"
        outside.mkdir()
        (outside / "s.txt").write_text("top secret")
        try:
            os.symlink(outside, repo.root / "link", target_is_directory=True)
            assert raises(InvalidPath, repo.read_file, "link/s.txt")
            assert raises(InvalidPath, repo.write_file, "link/new.txt", "x")
            assert "link/s.txt" not in repo.list_files()
            print("PASS 3: ignored dirs, '..', file-as-directory, symlink escape, all-or-nothing batches")
        except OSError:
            print("PASS 3: ignored dirs, '..', file-as-directory, all-or-nothing batches (symlinks unavailable here)")
        repo.delete_file("a.txt")

        # 4. Limits
        small = Repository(base / "small", max_file_bytes=100, max_files=3)
        assert raises(RepoLimitError, small.write_file, "big.txt", "x" * 101)
        small.write_files({"a.py": "1", "b.py": "2", "c.py": "3"})
        assert raises(RepoLimitError, small.write_file, "d.py", "4")
        assert small.write_file("a.py", "updated") == "updated"          # overwriting is not "a new file"
        print("PASS 4: size and file-count limits")

        # 5. Binary files and stats
        r5 = Repository(base / "r5")
        r5.write_files({"main.py": "a = 1\nb = 2\n", "docs/readme.md": "# hi\n"})
        (r5.root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
        assert raises(NotTextFile, r5.read_file, "logo.png")
        assert "logo.png" in r5.list_files() and "logo.png" not in r5.list_files(text_only=True)
        st = r5.stats()
        assert st.files == 3 and st.text_files == 2 and st.total_lines == 3 and st.by_extension[".py"]["lines"] == 2
        assert set(r5.read_text_files()) == {"main.py", "docs/readme.md"}
        print("PASS 5: binary detection and stats ->", st.by_extension)

        # 6. Glob filters and tree
        r5.write_files({"core/a.py": "", "core/b.py": "x\n", "tests/test_a.py": "t\n"})
        assert r5.list_files(include=["*.py"], exclude=["test_*.py"]) == ["core/a.py", "core/b.py", "main.py"]
        assert r5.list_files(include=["core/*.py"]) == ["core/a.py", "core/b.py"]
        expected = ("proj/\n├── core/\n│   ├── a.py\n│   └── b.py\n├── docs/\n│   └── readme.md\n"
                    "├── tests/\n│   └── test_a.py\n├── logo.png\n└── main.py")
        assert r5.tree(root_label="proj") == expected, r5.tree(root_label="proj")
        assert "core/ ..." in r5.tree(max_depth=1) and "a.py" not in r5.tree(max_depth=1)
        print("PASS 6: include/exclude globs and tree")

        # 7. Snapshot, diff, restore
        snap = r5.snapshot("v1")
        r5.write_file("main.py", "a = 1\nb = 3\n")
        r5.write_file("new.py", "n = 1\n")
        r5.delete_file("core/b.py")
        d = r5.diff(snap)
        assert d.added == ["new.py"] and d.modified == ["main.py"] and d.deleted == ["core/b.py"]
        assert "-b = 2" in d.unified and "+b = 3" in d.unified and "+n = 1" in d.unified and d.summary() == "1 added, 1 modified, 1 deleted"
        info = r5.restore(snap)
        assert info == {"restored": 2, "deleted": 1, "skipped": []}, info
        assert not r5.diff(snap).changed and r5.read_file("main.py") == "a = 1\nb = 2\n"
        assert Snapshot.from_dict(snap.to_dict()).hashes == snap.hashes
        print("PASS 7: snapshot / diff / restore")

        # 8. Parsing the canonical tag format (fences inside tags, CRLF, duplicates, unsafe paths)
        reply = (
            'Here is the project.\n\n'
            '<file path="src/app.py">\n```python\nprint("hi")\n```\n</file>\n\n'
            '<file path="README.md">\r\n# Title\r\n\r\nSome text\r\n</file>\n'
            '<file path="../evil.py">\nimport os\n</file>\n'
            '<file path="src/empty/__init__.py"></file>\n'
            '<file path="src/app.py">\nprint("second")\n</file>\n'
            'Done!')
        p = parse_model_output(reply)
        assert p.format == "tags" and p.files["src/app.py"] == 'print("second")\n'
        assert p.files["README.md"] == "# Title\n\nSome text\n" and p.files["src/empty/__init__.py"] == ""
        assert p.rejected and p.rejected[0][0] == "../evil.py" and not p.incomplete
        assert any("more than once" in w for w in p.warnings)
        print("PASS 8: tag parsing (fence unwrapping, CRLF, empty file, duplicate, unsafe path)")

        # 9. Cut-off replies: unclosed files are reported, never used
        cut = '<file path="a.py">\nx = 1\n</file>\n<file path="b.py">\ndef f():\n    return 1\n<file path="c.py">\ny = 2\n</file>\n<file path="d.py">\ndef g(:\n    ret'
        p = parse_model_output(cut)
        assert set(p.files) == {"a.py", "c.py"} and p.incomplete == ["b.py", "d.py"]
        assert any("token limit" in w for w in p.warnings)
        print("PASS 9: truncated replies ->", p.incomplete, "flagged as incomplete")

        # 10. Fenced fallback: heading labels, path comments, nested fences, unlabelled blocks ignored
        md = (
            "Run it with:\n\n```bash\npython main.py\n```\n\n"
            "### src/main.py\n\n```python\nprint('x')\n```\n\n"
            "**File: utils/helpers.py**\n```python\ndef h():\n    return 1\n```\n\n"
            "```python\n# core/engine.py\nclass Engine: ...\n```\n\n"
            "`docs/guide.md`:\n````markdown\n# Guide\n\n```python\nprint(1)\n```\n````\n")
        p = parse_model_output(md)
        assert p.format == "fences"
        assert set(p.files) == {"src/main.py", "utils/helpers.py", "core/engine.py", "docs/guide.md"}, set(p.files)
        assert p.files["core/engine.py"] == "class Engine: ...\n"
        assert p.files["docs/guide.md"] == "# Guide\n\n```python\nprint(1)\n```\n"
        assert parse_model_output("Just chatting, no code.").files == {}
        assert parse_model_output("").warnings == ["the reply was empty"]
        print("PASS 10: fenced-block fallback (labels, path comments, nested fences)")

        # 11. apply_model_output end to end, with placeholder warnings
        r11 = Repository(base / "r11")
        reply = (
            '<file path="app.py">\ndef main():\n    # ... rest of the code unchanged\n    pass\n</file>\n'
            '<file path="ok.py">\nx = 1\n</file>\n'
            '<file path=".git/hooks/pre-commit">\nrm -rf /\n</file>\n'
            '<file path="tail.py">\ndef broken(')
        res = r11.apply_model_output(reply)
        assert res.written == {"app.py": "created", "ok.py": "created"}
        assert res.incomplete == ["tail.py"] and res.rejected[0][0] == ".git/hooks/pre-commit"
        assert "app.py" in res.placeholders and not res.ok
        assert not (r11.root / ".git").exists() and not r11.exists("tail.py")
        assert Repository(base / "r11b").apply_model_output('<file path="a.py">\nx = 1\n</file>').ok
        print("PASS 11: apply_model_output ->", res.written, "| warnings:", len(res.warnings))

        # 12. copy_to
        copy = r5.copy_to(base / "copy")
        assert copy.list_files() == r5.list_files() and copy.read_file("main.py") == r5.read_file("main.py")
        assert raises(RepoError, r5.copy_to, base / "copy")
        assert (copy.root / "logo.png").read_bytes() == (r5.root / "logo.png").read_bytes()
        print("PASS 12: copy_to (including binaries), refuses a non-empty destination")

        assert "<file path=" in FILE_FORMAT_INSTRUCTIONS