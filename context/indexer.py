"""
context/indexer.py

A searchable map of a repository: symbols, imports and the file-level dependency graph.
No third-party packages. Python files are parsed with `ast`; if a file has a syntax error (common
in half-finished model output) the index falls back to regexes instead of failing. Other languages
(JS/TS, Go, Java, Rust, C/C++) use regex tables, so they are approximate but never crash.

    from context.repository import Repository
    from context.indexer import RepoIndex, build_index

    repo = Repository(project_dir / "implementation")
    index = build_index(repo, index_path=project_dir / "index.json")   # loads, updates, saves

    index.update(repo)                                  # incremental: only changed files are re-parsed
    index.find_symbol("Scheduler")                      # -> [SymbolHit(path, symbol)]
    index.search("jwt token refresh", limit=8)          # -> [(path, score)] ranked
    index.dependencies("core/scheduler.py")             # files it imports
    index.dependents("core/scheduler.py", transitive=True)   # files that would be affected by a change
    index.related_files(["auth/jwt.py"], depth=1)       # neighbours in the graph, both directions
    index.tests_for("auth/jwt.py")                      # likely test files
    print(index.outline(max_chars=4000))                # compact map to put in a prompt

The index is plain JSON (`to_dict()` / `save()` / `load()`), written atomically. Indexing problems
never raise: a file that cannot be read or parsed is still listed, with `parse_error` explaining why.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import posixpath
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from core.logger import get_logger

logger = get_logger("context.indexer")

INDEX_VERSION = 1
MAX_SIGNATURE_CHARS = 160
MAX_DOC_CHARS = 200

LANGUAGE_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".java": "java", ".rs": "rust",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
}
JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json")
STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "for", "with", "on", "is", "it", "that", "this",
             "be", "as", "by", "at", "from", "my", "me", "i", "app", "build", "create", "make", "file", "code"}

try:
    STDLIB = set(sys.stdlib_module_names)                     # Python 3.10+
except AttributeError:                                       # pragma: no cover
    STDLIB = {"os", "sys", "re", "json", "math", "time", "typing", "pathlib", "dataclasses", "collections",
              "itertools", "functools", "subprocess", "threading", "logging", "unittest", "argparse"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Symbol:
    name: str
    kind: str                       # class | function | method | variable | interface | type | enum | module
    line: int
    end_line: int = 0
    signature: str = ""
    doc: str = ""
    parent: str = ""                # enclosing class for methods
    decorators: list = field(default_factory=list)
    bases: list = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in ("", [], 0) or k in ("name", "kind", "line")}

    @classmethod
    def from_dict(cls, raw: dict) -> "Symbol":
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


@dataclass
class ImportRef:
    module: str                     # 'os.path', './utils', 'a.b' (for relative python: module without dots)
    names: list = field(default_factory=list)      # python 'from x import a, b'
    level: int = 0                  # python relative-import level
    line: int = 0
    lazy: bool = False              # imported inside a function (not at load time)
    kind: str = "import"            # import | include | mod

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in ("", [], 0, False, "import") or k == "module"}

    @classmethod
    def from_dict(cls, raw: dict) -> "ImportRef":
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


@dataclass
class FileIndex:
    path: str
    language: str
    sha256: str
    lines: int = 0
    size: int = 0
    summary: str = ""               # first line of the module docstring (python)
    symbols: list = field(default_factory=list)
    imports: list = field(default_factory=list)
    resolved: list = field(default_factory=list)       # repo files this file imports (filled by relink)
    external: list = field(default_factory=list)       # third-party top-level packages
    stdlib: list = field(default_factory=list)
    is_test: bool = False
    is_entry: bool = False          # python: has `if __name__ == "__main__"`
    parse_error: str = ""
    approximate: bool = False       # regex-based result (non-python or python with syntax errors)

    def to_dict(self) -> dict:
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        data["symbols"] = [s.to_dict() for s in self.symbols]
        data["imports"] = [i.to_dict() for i in self.imports]
        return data

    @classmethod
    def from_dict(cls, raw: dict) -> "FileIndex":
        fi = cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw and k not in ("symbols", "imports")})
        fi.symbols = [Symbol.from_dict(s) for s in raw.get("symbols", [])]
        fi.imports = [ImportRef.from_dict(i) for i in raw.get("imports", [])]
        return fi


@dataclass
class SymbolHit:
    path: str
    symbol: Symbol


@dataclass
class UpdateResult:
    added: list = field(default_factory=list)
    changed: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    unchanged: int = 0

    @property
    def modified(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    def summary(self) -> str:
        return (f"{len(self.added)} added, {len(self.changed)} changed, {len(self.removed)} removed, "
                f"{self.unchanged} unchanged")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _first_line(text: Optional[str], limit: int = MAX_DOC_CHARS) -> str:
    if not text:
        return ""
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


def _is_test_path(path: str) -> bool:
    name = posixpath.basename(path)
    stem = name.rsplit(".", 1)[0]
    parts = path.split("/")
    return (stem.startswith("test_") or stem.endswith("_test") or stem.endswith(".test") or ".spec." in name
            or ".test." in name or stem == "conftest" or any(p in ("tests", "test", "__tests__") for p in parts[:-1]))


def _tokens(text: str) -> list:
    """Split snake_case, camelCase, paths and words into lowercase tokens (stopwords removed)."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    words = re.split(r"[^A-Za-z0-9]+", text.lower())
    return [w for w in words if len(w) >= 2 and w not in STOPWORDS]


def _short(text: str, limit: int = MAX_SIGNATURE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# Python parsing
# --------------------------------------------------------------------------- #

def _py_signature(node: ast.AST) -> str:
    try:
        args = ast.unparse(node.args)                                     # type: ignore[attr-defined]
        returns = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
        return _short(f"({args}){returns}")
    except Exception:
        return "(...)"


def _py_decorators(node: ast.AST) -> list:
    out = []
    for d in getattr(node, "decorator_list", []):
        try:
            out.append(_short(ast.unparse(d), 60))
        except Exception:
            pass
    return out


def _py_end(node: ast.AST) -> int:
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)


def _parse_python_ast(source: str) -> tuple:
    """Returns (summary, symbols, imports, is_entry). Raises SyntaxError/ValueError on bad source."""
    tree = ast.parse(source)
    symbols: list = []
    imports: list = []
    entry = False

    def visit(body: list, parent: str, lazy: bool, top: bool) -> None:
        nonlocal entry
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                kind = "method" if parent else "function"
                if top or parent:                    # skip functions nested inside functions
                    symbols.append(Symbol(node.name, kind, node.lineno, _py_end(node),
                                          prefix + _py_signature(node), _first_line(ast.get_docstring(node)),
                                          parent, _py_decorators(node)))
                visit(node.body, "", True, False)
            elif isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    try:
                        bases.append(_short(ast.unparse(b), 60))
                    except Exception:
                        pass
                if top or parent:
                    symbols.append(Symbol(node.name, "class", node.lineno, _py_end(node), "",
                                          _first_line(ast.get_docstring(node)), parent,
                                          _py_decorators(node), bases))
                visit(node.body, node.name if (top or parent) else "", lazy, False)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportRef(alias.name, [], 0, node.lineno, lazy))
            elif isinstance(node, ast.ImportFrom):
                imports.append(ImportRef(node.module or "", [a.name for a in node.names], node.level or 0,
                                         node.lineno, lazy))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and top:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and (t.id.isupper() or t.id == "__all__"):
                        symbols.append(Symbol(t.id, "variable", node.lineno, _py_end(node)))
            elif isinstance(node, ast.If):
                if top and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) \
                        and node.test.left.id == "__name__":
                    entry = True
                    continue                          # the self-test block is not part of the API
                visit(node.body, parent, lazy, top)
                visit(node.orelse, parent, lazy, top)
            elif isinstance(node, (ast.Try, ast.With, ast.AsyncWith, ast.For, ast.While)):
                for attr in ("body", "orelse", "finalbody"):
                    visit(getattr(node, attr, []) or [], parent, lazy, top)
                for handler in getattr(node, "handlers", []) or []:
                    visit(handler.body, parent, lazy, top)

    visit(tree.body, "", False, True)
    return _first_line(ast.get_docstring(tree)), symbols, imports, entry


_PY_DEF = re.compile(r"^(\s*)(async\s+)?def\s+([A-Za-z_]\w*)\s*\((.*)$")
_PY_CLASS = re.compile(r"^(\s*)class\s+([A-Za-z_]\w*)\s*(?:\((.*?)\))?\s*:?")
_PY_IMPORT = re.compile(r"^\s*import\s+([\w.,\s]+?)(?:\s+as\s+\w+)?\s*$")
_PY_FROM = re.compile(r"^\s*from\s+(\.*)([\w.]*)\s+import\s+(.+)$")
_PY_CONST = re.compile(r"^([A-Z][A-Z0-9_]*)\s*[:=]")


def _parse_python_regex(source: str) -> tuple:
    """Line-based fallback used when `ast` fails (syntax errors). Approximate but never raises."""
    symbols: list = []
    imports: list = []
    entry = False
    class_stack: list = []                            # (indent, name)
    for number, line in enumerate(source.splitlines(), 1):
        if re.match(r"^if\s+__name__\s*==", line):
            entry = True
        m = _PY_CLASS.match(line)
        if m:
            indent = len(m.group(1))
            while class_stack and class_stack[-1][0] >= indent:
                class_stack.pop()
            parent = class_stack[-1][1] if class_stack else ""
            bases = [b.strip() for b in (m.group(3) or "").split(",") if b.strip()]
            symbols.append(Symbol(m.group(2), "class", number, number, "", "", parent, [], bases))
            class_stack.append((indent, m.group(2)))
            continue
        m = _PY_DEF.match(line)
        if m:
            indent = len(m.group(1))
            while class_stack and class_stack[-1][0] >= indent:
                class_stack.pop()
            parent = class_stack[-1][1] if class_stack else ""
            if indent == 0 or parent:
                symbols.append(Symbol(m.group(3), "method" if parent else "function", number, number,
                                      _short(("async " if m.group(2) else "") + "(" + m.group(4).rstrip(": ")),
                                      "", parent))
            continue
        m = _PY_FROM.match(line)
        if m:
            names = [n.strip().split(" as ")[0].strip() for n in m.group(3).strip("() ").split(",") if n.strip()]
            imports.append(ImportRef(m.group(2), names, len(m.group(1)), number, line.startswith((" ", "\t"))))
            continue
        m = _PY_IMPORT.match(line)
        if m:
            for name in m.group(1).split(","):
                if name.strip():
                    imports.append(ImportRef(name.strip().split(" as ")[0], [], 0, number,
                                             line.startswith((" ", "\t"))))
            continue
        m = _PY_CONST.match(line)
        if m:
            symbols.append(Symbol(m.group(1), "variable", number, number))
    return "", symbols, imports, entry


# --------------------------------------------------------------------------- #
# Other languages (regex tables): (pattern, kind) with group 1 = name
# --------------------------------------------------------------------------- #

_JS_SYMBOLS = [
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*"
                r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>"), "function"),
    (re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)"), "interface"),
    (re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*="), "type"),
    (re.compile(r"^\s*(?:export\s+)?(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)"), "enum"),
]
_JS_IMPORTS = [
    re.compile(r"""^\s*(?:import|export)\s[^'"]*?\sfrom\s*['"]([^'"]+)['"]"""),
    re.compile(r"""^\s*import\s*['"]([^'"]+)['"]"""),
    re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""\bimport\(\s*['"]([^'"]+)['"]\s*\)"""),
]
_GO_SYMBOLS = [
    (re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"), "function"),
    (re.compile(r"^type\s+([A-Za-z_]\w*)\s+struct\b"), "class"),
    (re.compile(r"^type\s+([A-Za-z_]\w*)\s+interface\b"), "interface"),
    (re.compile(r"^type\s+([A-Za-z_]\w*)\s+"), "type"),
]
_JAVA_SYMBOLS = [
    (re.compile(r"^\s*(?:public\s+|protected\s+|private\s+|abstract\s+|final\s+|static\s+)*class\s+(\w+)"), "class"),
    (re.compile(r"^\s*(?:public\s+|protected\s+|private\s+)*interface\s+(\w+)"), "interface"),
    (re.compile(r"^\s*(?:public\s+|protected\s+|private\s+)*enum\s+(\w+)"), "enum"),
    (re.compile(r"^\s*(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\],\s?]+?\s+(\w+)\s*\([^;]*$"),
     "method"),
]
_RUST_SYMBOLS = [
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)"), "function"),
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)"), "class"),
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)"), "enum"),
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)"), "interface"),
    (re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?(\w+)"), "class"),
]
_C_SYMBOLS = [
    (re.compile(r"^\s*(?:typedef\s+)?(?:class|struct)\s+(\w+)\s*(?::[^{;]*)?\{?\s*$"), "class"),
    (re.compile(r"^[A-Za-z_][\w\s\*:<>,&]*?[\s\*&]([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{?\s*$"), "function"),
]
_SYMBOL_TABLES = {
    "javascript": _JS_SYMBOLS, "typescript": _JS_SYMBOLS, "go": _GO_SYMBOLS, "java": _JAVA_SYMBOLS,
    "rust": _RUST_SYMBOLS, "c": _C_SYMBOLS, "cpp": _C_SYMBOLS,
}
_C_CONTROL = {"if", "for", "while", "switch", "return", "else", "do", "sizeof", "catch"}
_JAVA_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+?)(?:\.\*)?\s*;")
_RUST_USE = re.compile(r"^\s*(?:pub\s+)?use\s+([\w:]+)")
_RUST_MOD = re.compile(r"^\s*(?:pub\s+)?mod\s+(\w+)\s*;")
_C_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
_GO_IMPORT_BLOCK = re.compile(r"^import\s*\(([^)]*)\)", re.M | re.S)
_GO_IMPORT_ONE = re.compile(r'^import\s+(?:\w+\s+)?"([^"]+)"', re.M)
_QUOTED = re.compile(r'"([^"]+)"')


def _parse_generic(source: str, language: str) -> tuple:
    symbols: list = []
    imports: list = []
    table = _SYMBOL_TABLES.get(language, [])
    in_block_comment = False
    for number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*") and "*/" not in stripped:
            in_block_comment = True
            continue
        if stripped.startswith(("//", "#!")) or (stripped.startswith("#") and language not in ("c", "cpp")):
            continue
        for pattern, kind in table:
            m = pattern.match(line)
            if m:
                name = m.group(1)
                if language in ("c", "cpp") and (name in _C_CONTROL or kind == "function" and line.startswith(" ")):
                    break
                symbols.append(Symbol(name, kind, number, number, _short(stripped, 100)))
                break
        if language in ("javascript", "typescript"):
            for pattern in _JS_IMPORTS:
                for m in pattern.finditer(line):
                    imports.append(ImportRef(m.group(1), [], 0, number))
                    break
        elif language == "java":
            m = _JAVA_IMPORT.match(line)
            if m:
                imports.append(ImportRef(m.group(1), [], 0, number))
        elif language == "rust":
            m = _RUST_MOD.match(line)
            if m:
                imports.append(ImportRef(m.group(1), [], 0, number, kind="mod"))
            else:
                m = _RUST_USE.match(line)
                if m:
                    imports.append(ImportRef(m.group(1), [], 0, number))
        elif language in ("c", "cpp"):
            m = _C_INCLUDE.match(line)
            if m:
                imports.append(ImportRef(m.group(2), [], 0 if m.group(1) == '"' else -1, number, kind="include"))
    if language == "go":
        for m in _GO_IMPORT_BLOCK.finditer(source):
            for q in _QUOTED.finditer(m.group(1)):
                imports.append(ImportRef(q.group(1), [], 0, source[: m.start()].count("\n") + 1))
        for m in _GO_IMPORT_ONE.finditer(source):
            imports.append(ImportRef(m.group(1), [], 0, source[: m.start()].count("\n") + 1))
    return "", symbols, imports, False


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #

class RepoIndex:
    def __init__(self, files: Optional[dict] = None):
        self.files: dict = files or {}            # path -> FileIndex
        self._dependents: dict = {}
        self.parse_count = 0                      # how many files were parsed (tests use this to prove incrementality)
        self._relink()

    # ---- building ---------------------------------------------------------- #

    @staticmethod
    def parse_source(path: str, source: str) -> FileIndex:
        """Index one file's text. Never raises."""
        language = LANGUAGE_BY_EXT.get(posixpath.splitext(path)[1].lower(), "other")
        fi = FileIndex(path=path, language=language, sha256=_sha256(source), size=len(source.encode("utf-8", "replace")),
                       lines=source.count("\n") + (1 if source and not source.endswith("\n") else 0),
                       is_test=_is_test_path(path))
        try:
            if language == "python":
                try:
                    fi.summary, fi.symbols, fi.imports, fi.is_entry = _parse_python_ast(source)
                except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
                    fi.parse_error = f"{type(e).__name__}: {e}"[:200]
                    fi.approximate = True
                    fi.summary, fi.symbols, fi.imports, fi.is_entry = _parse_python_regex(source)
            elif language != "other":
                fi.approximate = True
                fi.summary, fi.symbols, fi.imports, fi.is_entry = _parse_generic(source, language)
        except Exception as e:                    # a parser bug must not break indexing
            fi.parse_error = f"{type(e).__name__}: {e}"[:200]
            logger.warning("Indexing %s failed: %s", path, fi.parse_error)
        return fi

    def update(self, repo: Any, paths: Optional[Iterable] = None) -> UpdateResult:
        """
        Bring the index in line with the repository. Only files whose content hash changed are
        re-parsed. `paths` limits the update to those files (deletions are then detected only for them).
        """
        result = UpdateResult()
        wanted = sorted(paths) if paths is not None else repo.list_files(text_only=True)
        seen = set()
        for rel in wanted:
            seen.add(rel)
            try:
                source = repo.read_file(rel, errors="replace")
            except Exception as e:                # unreadable / binary / vanished: list it, don't crash
                logger.debug("Skipping %s: %s", rel, e)
                if rel in self.files:
                    del self.files[rel]
                    result.removed.append(rel)
                continue
            digest = _sha256(source)
            old = self.files.get(rel)
            if old and old.sha256 == digest:
                result.unchanged += 1
                continue
            self.files[rel] = self.parse_source(rel, source)
            self.parse_count += 1
            (result.changed if old else result.added).append(rel)

        stale = [p for p in self.files if p not in seen] if paths is None else \
                [p for p in (paths or []) if p in self.files and not repo.exists(p)]
        for rel in stale:
            del self.files[rel]
            result.removed.append(rel)

        if result.modified or not self._dependents:
            self._relink()
        logger.debug("Index updated: %s", result.summary())
        return result

    # ---- dependency graph --------------------------------------------------- #

    def _relink(self) -> None:
        """Resolve imports to repo files (depends on the whole file set, so redone after any change)."""
        py_paths = [p for p, f in self.files.items() if f.language == "python"]
        suffixes: dict = {}
        for p in py_paths:
            parts = p[:-3 if p.endswith(".py") else -4].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1] or ["__init__"]
            for i in range(len(parts)):
                suffixes.setdefault("/".join(parts[i:]), []).append(p)
        all_paths = set(self.files)

        for fi in self.files.values():
            targets: list = []
            external: set = set()
            stdlib: set = set()
            for ref in fi.imports:
                found = self._resolve(fi, ref, all_paths, suffixes)
                if found:
                    targets.extend(found)
                    continue
                top = self._top_package(fi, ref)
                if top:
                    if fi.language == "python" and top in STDLIB:
                        stdlib.add(top)
                    elif fi.language == "python" or not top.startswith("."):
                        external.add(top)
            fi.resolved = sorted({t for t in targets if t != fi.path})
            fi.external = sorted(external)
            fi.stdlib = sorted(stdlib)

        self._dependents = {}
        for fi in self.files.values():
            for target in fi.resolved:
                self._dependents.setdefault(target, set()).add(fi.path)

    @staticmethod
    def _top_package(fi: FileIndex, ref: ImportRef) -> str:
        mod = ref.module
        if fi.language == "python":
            return "" if ref.level else mod.split(".")[0]
        if fi.language in ("javascript", "typescript"):
            if mod.startswith("."):
                return ""
            parts = mod.split("/")
            return "/".join(parts[:2]) if mod.startswith("@") and len(parts) > 1 else parts[0]
        if fi.language in ("c", "cpp"):
            return mod if ref.level == -1 else ""
        return mod.split(".")[0].split("::")[0] if mod else ""

    def _resolve(self, fi: FileIndex, ref: ImportRef, all_paths: set, suffixes: dict) -> list:
        lang = fi.language
        here = posixpath.dirname(fi.path)
        if lang == "python":
            return self._resolve_python(fi, ref, all_paths, suffixes, here)
        if lang in ("javascript", "typescript"):
            if not ref.module.startswith("."):
                return []
            base = posixpath.normpath(posixpath.join(here, ref.module))
            candidates = [base] + [base + e for e in JS_EXTS] + [posixpath.join(base, "index" + e) for e in JS_EXTS]
            return [c for c in candidates if c in all_paths][:1]
        if lang in ("c", "cpp") and ref.kind == "include" and ref.level == 0:
            for c in (posixpath.normpath(posixpath.join(here, ref.module)), ref.module):
                if c in all_paths:
                    return [c]
            tail = "/" + ref.module
            return [p for p in sorted(all_paths) if p.endswith(tail)][:1]
        if lang == "java":
            tail = "/".join(ref.module.split(".")) + ".java"
            return [p for p in sorted(all_paths) if p == tail or p.endswith("/" + tail)][:1]
        if lang == "rust" and ref.kind == "mod":
            for c in (posixpath.join(here, ref.module + ".rs"), posixpath.join(here, ref.module, "mod.rs")):
                if c in all_paths:
                    return [c]
        return []

    @staticmethod
    def _resolve_python(fi: FileIndex, ref: ImportRef, all_paths: set, suffixes: dict, here: str) -> list:
        def lookup(parts: list, relative_to: Optional[str]) -> Optional[str]:
            if not parts:
                if relative_to is None:
                    return None
                init = posixpath.join(relative_to, "__init__.py") if relative_to else "__init__.py"
                return init if init in all_paths else None
            joined = "/".join(parts)
            if relative_to is not None:
                base = posixpath.join(relative_to, joined) if relative_to else joined
                for cand in (base + ".py", base + "/__init__.py"):
                    if cand in all_paths:
                        return cand
                return None
            for cand in (joined + ".py", joined + "/__init__.py"):
                if cand in all_paths:
                    return cand
            hits = suffixes.get(joined)               # src/ layouts and nested packages
            if hits:
                return sorted(hits, key=lambda p: (len(p), p))[0]
            return None

        module_parts = [p for p in ref.module.split(".") if p]
        anchor: Optional[str] = None
        if ref.level:
            anchor = here
            for _ in range(ref.level - 1):
                anchor = posixpath.dirname(anchor)
        results: list = []
        for name in ref.names:                        # `from pkg import submodule`
            hit = lookup(module_parts + [name], anchor)
            if hit:
                results.append(hit)
        if not results or not ref.names or any(n == "*" for n in ref.names):
            base = lookup(module_parts, anchor)
            if base:
                results.append(base)
        elif module_parts or anchor is not None:      # also depend on the package/module itself if it defines the names
            base = lookup(module_parts, anchor)
            if base and len(results) < len(ref.names):
                results.append(base)
        return results

    # ---- queries ------------------------------------------------------------ #

    def __contains__(self, path: str) -> bool:
        return path in self.files

    def __len__(self) -> int:
        return len(self.files)

    def get(self, path: str) -> Optional[FileIndex]:
        return self.files.get(path)

    def find_symbol(self, name: str, kind: Optional[str] = None, exact: bool = True) -> list:
        """Symbols by name (exact, case-sensitive; or substring, case-insensitive). `Class.method` also works."""
        hits = []
        needle = name.lower()
        for path in sorted(self.files):
            for sym in self.files[path].symbols:
                if kind and sym.kind != kind:
                    continue
                ok = name in (sym.name, sym.qualified_name) if exact else needle in sym.qualified_name.lower()
                if ok:
                    hits.append(SymbolHit(path, sym))
        return hits

    def dependencies(self, path: str, transitive: bool = False) -> list:
        """Repo files that `path` imports."""
        return self._walk(path, lambda p: self.files[p].resolved if p in self.files else [], transitive)

    def dependents(self, path: str, transitive: bool = False) -> list:
        """Repo files that import `path` (what a change to it can break)."""
        return self._walk(path, lambda p: self._dependents.get(p, ()), transitive)

    @staticmethod
    def _walk(start: str, neighbours, transitive: bool) -> list:
        seen: set = set()
        frontier = [start]
        while frontier:
            nxt = []
            for node in frontier:
                for n in neighbours(node):
                    if n != start and n not in seen:
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt if transitive else []
        return sorted(seen)

    def related_files(self, paths: Iterable, depth: int = 1) -> list:
        """The given files plus their neighbours (imports and importers) up to `depth` hops."""
        result = set(p for p in paths if p in self.files)
        frontier = set(result)
        for _ in range(max(0, depth)):
            nxt = set()
            for p in frontier:
                nxt.update(self.dependencies(p))
                nxt.update(self.dependents(p))
            frontier = nxt - result
            result |= nxt
        return sorted(result)

    def tests_for(self, path: str) -> list:
        """Test files that import `path` or follow the naming convention (test_x.py / x_test.py / x.test.ts)."""
        found = {p for p in self.dependents(path) if self.files[p].is_test}
        stem = posixpath.basename(path).split(".")[0]
        for p, f in self.files.items():
            if f.is_test and posixpath.basename(p).split(".")[0] in (f"test_{stem}", f"{stem}_test", f"{stem}.test",
                                                                      f"{stem}.spec", f"test{stem}"):
                found.add(p)
        for p, f in self.files.items():               # x.test.ts / x.spec.ts
            if f.is_test and re.match(rf"^{re.escape(stem)}\.(test|spec)\.", posixpath.basename(p)):
                found.add(p)
        return sorted(found)

    def importance(self, path: str) -> float:
        """Cheap centrality: files many others import matter more. Entry points get a boost."""
        fi = self.files.get(path)
        if not fi:
            return 0.0
        return math.log1p(len(self._dependents.get(path, ()))) + (0.5 if fi.is_entry else 0.0)

    def ranked_files(self, include_tests: bool = True) -> list:
        paths = [p for p, f in self.files.items() if include_tests or not f.is_test]
        return sorted(paths, key=lambda p: (-self.importance(p), p))

    def entry_points(self) -> list:
        return sorted(p for p, f in self.files.items() if f.is_entry and not f.is_test)

    def search(self, query: str, limit: int = 10, include_tests: bool = True) -> list:
        """
        Rank files for a natural-language query by matching its words against path, symbol names and
        docstrings. Returns [(path, score)], best first. Deterministic; no embeddings needed.
        """
        words = set(_tokens(query))
        if not words:
            return []
        scored = []
        for path, fi in self.files.items():
            if fi.is_test and not include_tests:
                continue
            path_t = set(_tokens(path))
            sym_t: set = set()
            doc_t = set(_tokens(fi.summary))
            for s in fi.symbols:
                sym_t.update(_tokens(s.qualified_name))
                doc_t.update(_tokens(s.doc))
            score = 3.0 * len(words & path_t) + 2.0 * len(words & sym_t) + 1.0 * len(words & doc_t)
            if score <= 0:
                continue
            scored.append((path, round(score + 0.25 * self.importance(path), 4)))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:limit]

    def outline(self, paths: Optional[Iterable] = None, max_chars: int = 6000, include_methods: bool = True) -> str:
        """
        A compact text map, suitable for a prompt:

            core/scheduler.py  (761 lines)  Decides who does which stage.
              class Scheduler  L120
                def plan(self, analysis)  L150
        """
        order = [p for p in (paths if paths is not None else self.ranked_files()) if p in self.files]
        chunks, used = [], 0
        for i, path in enumerate(order):
            fi = self.files[path]
            lines = [f"{path}  ({fi.lines} lines){'  ' + fi.summary if fi.summary else ''}"]
            for s in fi.symbols:
                if s.kind == "method" and not include_methods:
                    continue
                indent = "    " if s.parent else "  "
                label = {"function": "def", "method": "def"}.get(s.kind, s.kind)
                tail = s.signature if s.kind in ("function", "method") else (f"({', '.join(s.bases)})" if s.bases else "")
                lines.append(f"{indent}{label} {s.name}{tail}  L{s.line}")
            chunk = "\n".join(lines)
            if used + len(chunk) + 1 > max_chars:
                chunks.append(f"... ({len(order) - i} more files)")
                break
            chunks.append(chunk)
            used += len(chunk) + 1
        return "\n".join(chunks)

    def stats(self) -> dict:
        langs: dict = {}
        for f in self.files.values():
            langs[f.language] = langs.get(f.language, 0) + 1
        return {
            "files": len(self.files),
            "languages": dict(sorted(langs.items())),
            "symbols": sum(len(f.symbols) for f in self.files.values()),
            "edges": sum(len(f.resolved) for f in self.files.values()),
            "approximate_files": sorted(p for p, f in self.files.items() if f.approximate and f.parse_error),
            "external_packages": sorted({e for f in self.files.values() for e in f.external}),
        }

    # ---- persistence --------------------------------------------------------- #

    def to_dict(self) -> dict:
        return {"version": INDEX_VERSION, "files": {p: f.to_dict() for p, f in sorted(self.files.items())}}

    @classmethod
    def from_dict(cls, raw: dict) -> "RepoIndex":
        if not isinstance(raw, dict) or raw.get("version") != INDEX_VERSION:
            raise ValueError("unsupported or corrupt index data")
        return cls({p: FileIndex.from_dict(f) for p, f in raw.get("files", {}).items()})

    def save(self, path) -> None:
        """Atomic write (temp file + replace)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=1)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path) -> "RepoIndex":
        """Load a saved index; a missing or corrupt file gives an empty index (it will simply be rebuilt)."""
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except FileNotFoundError:
            return cls()
        except (ValueError, OSError, TypeError, KeyError) as e:
            logger.warning("Index file %s unusable (%s); starting from an empty index", path, e)
            return cls()


def build_index(repo: Any, index_path=None) -> RepoIndex:
    """Load the saved index (if any), update it from the repository, save it, and return it."""
    index = RepoIndex.load(index_path) if index_path else RepoIndex()
    result = index.update(repo)
    if index_path and (result.modified or not Path(index_path).exists()):
        try:
            index.save(index_path)
        except OSError as e:                          # bookkeeping must never crash a run
            logger.warning("Could not save index to %s: %s", index_path, e)
    return index


# --------------------------------------------------------------------------- #
# Self-test:  python -m context.indexer     (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from context.repository import Repository
    from core.logger import setup_logging

    setup_logging("ERROR")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj"
        repo = Repository(root)
        repo.write_files({
            "main.py": '"""Entry point."""\nimport os\nimport requests\nfrom core import scheduler\nfrom core.models import Job\n\n'
                       'def run():\n    import json\n    return scheduler.plan()\n\nif __name__ == "__main__":\n    run()\n',
            "core/__init__.py": "",
            "core/scheduler.py": '"""Plans which model does which stage."""\nfrom __future__ import annotations\n'
                                 'from .models import Job\nfrom . import helpers\nMAX_JOBS = 3\n\n'
                                 'class Scheduler(Base):\n    """Does the planning."""\n'
                                 '    def plan(self, analysis: int = 1) -> str:\n        """Make a plan."""\n        return "x"\n'
                                 '    async def wait(self): ...\n\n'
                                 '@decorator\ndef helper_fn(a, *, b=2): return a\n',
            "core/models.py": "class Job:\n    pass\n",
            "core/helpers.py": "def util():\n    pass\n",
            "src/pkg/__init__.py": "",
            "src/pkg/auth.py": '"""JWT token handling."""\ndef create_token(user): return "t"\ndef refresh_token(t): return t\n',
            "uses_src.py": "from pkg.auth import create_token\nimport pkg\n",
            "tests/test_scheduler.py": "from core.scheduler import Scheduler\ndef test_plan(): pass\n",
            "tests/test_auth.py": "def test_auth(): pass\n",
            "broken.py": "import os\nclass Half(Base):\n    def ok(self):\n        pass\n    def bad(self:\n        return (\n",
            "web/app.ts": "import { x } from './util';\nimport React from 'react';\nimport '@scope/pkg/sub';\n"
                          "export async function load() {}\nexport const go = async (a: number) => a;\n"
                          "export interface User { id: number }\nexport class Store {}\n",
            "web/util.ts": "export function x() {}\nconst y = require('./missing');\n",
            "svc/main.go": 'package main\n\nimport (\n\t"fmt"\n\t"github.com/x/y"\n)\n\ntype Server struct{}\n\nfunc (s *Server) Run() {}\nfunc main() {}\n',
            "lib/Thing.java": "import com.acme.Helper;\npublic class Thing {\n    public void go(int a) {\n    }\n}\n",
            "com/acme/Helper.java": "public class Helper {}\n",
            "notes.txt": "just some notes\n",
            "empty.py": "",
        })

        index = RepoIndex()
        res = index.update(repo)
        assert len(res.added) == len(index) == 18 and res.unchanged == 0, (len(res.added), len(index))
        print("PASS 1: indexed", len(index), "files ->", index.stats()["languages"])

        # 2. Python symbols: kinds, parents, signatures, docstrings, decorators, bases, constants, entry point
        sch = index.get("core/scheduler.py")
        names = {s.qualified_name: s for s in sch.symbols}
        assert set(names) == {"MAX_JOBS", "Scheduler", "Scheduler.plan", "Scheduler.wait", "helper_fn"}, set(names)
        assert names["Scheduler"].bases == ["Base"] and names["Scheduler"].doc == "Does the planning."
        assert names["Scheduler.plan"].kind == "method" and "analysis: int=1" in names["Scheduler.plan"].signature
        assert names["Scheduler.plan"].signature.endswith("-> str") and names["Scheduler.plan"].doc == "Make a plan."
        assert names["Scheduler.wait"].signature.startswith("async ") and names["helper_fn"].decorators == ["decorator"]
        assert sch.summary == "Plans which model does which stage." and not sch.approximate
        assert index.get("main.py").is_entry and index.entry_points() == ["main.py"]
        assert [s.name for s in index.get("main.py").symbols] == ["run"]          # __main__ block not an API symbol
        print("PASS 2: python symbols, signatures, docstrings, decorators, entry points")

        # 3. Import resolution (absolute, relative, `from pkg import module`, src/ layout) + external/stdlib split
        assert index.dependencies("main.py") == ["core/models.py", "core/scheduler.py"], index.dependencies("main.py")
        assert index.dependencies("core/scheduler.py") == ["core/helpers.py", "core/models.py"]
        assert index.dependencies("uses_src.py") == ["src/pkg/__init__.py", "src/pkg/auth.py"]
        main = index.get("main.py")
        assert main.external == ["requests"] and set(main.stdlib) == {"json", "os"}, (main.external, main.stdlib)
        assert [i.lazy for i in main.imports if i.module == "json"] == [True]
        print("PASS 3: imports resolved (relative, src layout, lazy), stdlib vs external classified")

        # 4. Reverse dependencies, transitive closure, neighbours, test mapping
        assert index.dependents("core/models.py") == ["core/scheduler.py", "main.py"]
        assert index.dependents("core/models.py", transitive=True) == ["core/scheduler.py", "main.py",
                                                                        "tests/test_scheduler.py"]
        assert "core/helpers.py" in index.related_files(["main.py"], depth=2)
        assert index.tests_for("core/scheduler.py") == ["tests/test_scheduler.py"]
        assert index.tests_for("src/pkg/auth.py") == ["tests/test_auth.py"]      # by naming convention
        print("PASS 4: dependents (direct + transitive), related files, tests_for")

        # 5. Syntax errors: regex fallback keeps what it can, records the error, does not raise
        broken = index.get("broken.py")
        assert broken.approximate and "SyntaxError" in broken.parse_error
        assert {s.qualified_name for s in broken.symbols} == {"Half", "Half.ok", "Half.bad"}
        assert [i.module for i in broken.imports] == ["os"]
        assert index.stats()["approximate_files"] == ["broken.py"]
        assert index.get("empty.py").symbols == [] and index.get("notes.txt").language == "other"
        print("PASS 5: syntax-error file indexed via fallback ->", sorted(s.qualified_name for s in broken.symbols))

        # 6. Other languages
        ts = {s.name: s.kind for s in index.get("web/app.ts").symbols}
        assert ts == {"load": "function", "go": "function", "User": "interface", "Store": "class"}, ts
        assert index.dependencies("web/app.ts") == ["web/util.ts"]
        assert index.get("web/app.ts").external == ["@scope/pkg", "react"], index.get("web/app.ts").external
        assert index.get("web/util.ts").external == ["./missing"] or index.get("web/util.ts").external == []
        go = {s.name: s.kind for s in index.get("svc/main.go").symbols}
        assert go == {"Server": "class", "Run": "function", "main": "function"}, go
        assert sorted(i.module for i in index.get("svc/main.go").imports) == ["fmt", "github.com/x/y"]
        assert index.dependencies("lib/Thing.java") == ["com/acme/Helper.java"]
        assert any(s.name == "Thing" and s.kind == "class" for s in index.get("lib/Thing.java").symbols)
        assert index.get("web/app.ts").approximate
        print("PASS 6: TypeScript / Go / Java symbols and imports")

        # 7. find_symbol and search
        hit = index.find_symbol("plan")
        assert [(h.path, h.symbol.kind) for h in hit] == [("core/scheduler.py", "method")]   # bare name matches methods
        assert [(h.path) for h in index.find_symbol("Scheduler.plan")] == ["core/scheduler.py"]
        assert [h.path for h in index.find_symbol("sched", exact=False)] == ["core/scheduler.py"] * 3
        assert [h.path for h in index.find_symbol("Job", kind="class")] == ["core/models.py"]
        top = index.search("refresh the jwt token")
        assert top[0][0] == "src/pkg/auth.py", top
        assert index.search("scheduler planning")[0][0] == "core/scheduler.py"
        assert index.search("the and of") == [] and index.search("zzzqqq") == []
        assert all(p != "tests/test_auth.py" for p, _ in index.search("auth", include_tests=False))
        print("PASS 7: find_symbol and ranked search ->", top[0])

        # 8. Incremental update: only changed files are re-parsed; related state is relinked
        parsed_before = index.parse_count
        res = index.update(repo)
        assert not res.modified and res.unchanged == 18 and index.parse_count == parsed_before
        repo.write_file("core/models.py", "class Job:\n    pass\nclass Task:\n    pass\n")
        repo.write_file("core/new.py", "from core.models import Task\n")
        repo.delete_file("core/helpers.py")
        res = index.update(repo)
        assert res.changed == ["core/models.py"] and res.added == ["core/new.py"] and res.removed == ["core/helpers.py"]
        assert index.parse_count == parsed_before + 2
        assert "core/new.py" in index.dependents("core/models.py")
        assert index.dependencies("core/scheduler.py") == ["core/__init__.py", "core/models.py"]   # helpers.py gone: `from . import helpers` now falls back to the package
        assert [h.path for h in index.find_symbol("Task")] == ["core/models.py"]
        print("PASS 8: incremental update ->", res.summary())

        # 9. Partial update with `paths`
        repo.write_file("core/models.py", "class Job:\n    pass\n")
        repo.write_file("notes.txt", "changed\n")
        res = index.update(repo, paths=["core/models.py"])
        assert res.changed == ["core/models.py"] and index.get("notes.txt").sha256 != "" and res.unchanged == 0
        assert index.find_symbol("Task") == []
        print("PASS 9: partial update limited to the given paths")

        # 10. Persistence: JSON round trip, atomic save, corrupt file tolerated, build_index reuses the file
        path = Path(tmp) / "state" / "index.json"
        index.save(path)
        json.loads(path.read_text(encoding="utf-8"))                              # valid JSON
        loaded = RepoIndex.load(path)
        assert loaded.to_dict() == index.to_dict() and loaded.dependents("core/models.py") == \
            index.dependents("core/models.py")
        assert not list(path.parent.glob("*.tmp"))
        path.write_text("{not json", encoding="utf-8")
        assert len(RepoIndex.load(path)) == 0 and len(RepoIndex.load(Path(tmp) / "none.json")) == 0
        built = build_index(repo, path)
        assert len(built) == len(index) and RepoIndex.load(path).parse_count == 0
        again = build_index(repo, path)
        assert again.parse_count == 0                                              # nothing re-parsed from a saved index
        print("PASS 10: save/load round trip, corrupt file tolerated, build_index reuses saved state")

        # 11. Outline fits its budget and lists the important files first
        text = index.outline(max_chars=900)
        assert len(text) <= 900 + 40 and "more files" in text
        full = index.outline()
        assert "class Scheduler(Base)" in full and "def plan(self, analysis: int=1) -> str  L" in full
        assert full.index("core/models.py") < full.index("notes.txt")             # imported files rank higher
        assert "def plan" not in index.outline(include_methods=False)
        assert index.outline(paths=["main.py"]).startswith("main.py  (")
        print("PASS 11: outline respects the character budget and ranks by importance")

        # 12. Robustness: CRLF source, unreadable/binary files, unicode, huge nesting
        repo.write_file("crlf.py", "def a():\r\n    pass\r\n\r\nclass B:\r\n    pass\r\n")
        (root / "blob.bin").write_bytes(b"\x00\x01\x02" * 50)
        repo.write_file("uni.py", '"""Café ☕ module"""\ndef größe(): pass\n')
        deep = "x = " + "(" * 400 + "1" + ")" * 400 + "\ndef after(): pass\n"
        repo.write_file("deep.py", deep)
        index.update(repo)
        assert [s.name for s in index.get("crlf.py").symbols] == ["a", "B"]
        assert "blob.bin" not in index and index.get("uni.py").symbols[0].name == "größe"
        assert index.get("deep.py") is not None                                    # parsed or fell back, but never raised
        print("PASS 12: CRLF, binary files, unicode and deeply nested code handled")
