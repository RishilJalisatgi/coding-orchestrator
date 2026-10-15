"""
context/selector.py

Chooses WHAT a model gets to see: the most relevant files (or parts of files) for a task, within a
token budget. Models receive this instead of the whole repository.

    from context.repository import Repository
    from context.indexer import RepoIndex
    from context.selector import ContextSelector

    repo = Repository(project_dir / "implementation")
    index = RepoIndex(); index.update(repo)
    selector = ContextSelector(repo, index)

    ctx = selector.select("fix the token refresh bug", budget_tokens=6000)
    prompt = f"{task}\\n\\n{ctx.render()}"
    ctx.tokens        # never more than budget_tokens
    ctx.files         # [SelectedFile(path, mode, tokens, score, reason)]
    ctx.omitted       # [(path, why)]  relevant files that did not fit

    # review / revision: the files that changed + what depends on them + their tests
    ctx = selector.select("review the changes", focus_files=diff_paths, include_tests=True)

How the budget is spent (best candidates first):
  1. an optional project map (compact outline of every file, so the model knows what exists)
  2. each candidate gets the richest form that fits: full file -> relevant snippets -> signatures only
Candidates are: focus files (forced to the top), files matching the query (indexer.search), and the
neighbours of the best matches in the import graph (imports and importers), plus related tests.
Everything is deterministic: same repo + same query = same selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from context.indexer import RepoIndex, FileIndex, Symbol, _tokens
from core.logger import get_logger

logger = get_logger("context.selector")

FULL, SNIPPETS, OUTLINE = "full", "snippets", "outline"
DEFAULT_BUDGET = 6000
MAP_SHARE = 0.12                    # at most this share of the budget goes to the project map
MAX_FILE_SHARE = 0.45               # a single file in full may use at most this share of the budget
NEIGHBOUR_DECAY = 0.5               # neighbour score = decay * score of the file it came from
MIN_USEFUL_TOKENS = 40              # don't bother with leftovers smaller than this
NON_PY_SNIPPET_LINES = 40           # regex-indexed symbols have no end line: show this many lines


def estimate_tokens(text: str) -> int:
    """~4 characters per token, same rule as the rest of the project."""
    return max(1, len(text) // 4) if text else 0


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class SelectedFile:
    path: str
    mode: str                       # full | snippets | outline
    content: str
    tokens: int
    score: float
    reason: str                     # why it was chosen, for the decision log

    def to_dict(self) -> dict:
        return {"path": self.path, "mode": self.mode, "tokens": self.tokens,
                "score": round(self.score, 3), "reason": self.reason}


@dataclass
class SelectedContext:
    files: list = field(default_factory=list)
    project_map: str = ""
    omitted: list = field(default_factory=list)       # [(path, reason)]
    budget_tokens: int = 0
    query: str = ""

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.project_map) + sum(f.tokens for f in self.files)

    @property
    def paths(self) -> list:
        return [f.path for f in self.files]

    def render(self) -> str:
        """Text to put in a prompt. Uses '=== path [mode] ===' headers (not the <file> output format)."""
        parts = []
        if self.project_map:
            parts.append("=== PROJECT MAP ===\n" + self.project_map)
        for f in self.files:
            parts.append(f"=== {f.path} [{f.mode}] ===\n{f.content}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {"query": self.query, "budget_tokens": self.budget_tokens, "tokens": self.tokens,
                "files": [f.to_dict() for f in self.files],
                "omitted": [list(o) for o in self.omitted],
                "map_tokens": estimate_tokens(self.project_map)}

    def summary(self) -> str:
        modes: dict = {}
        for f in self.files:
            modes[f.mode] = modes.get(f.mode, 0) + 1
        mix = ", ".join(f"{n} {m}" for m, n in sorted(modes.items())) or "nothing"
        return f"{self.tokens}/{self.budget_tokens} tokens: {mix}; {len(self.omitted)} omitted"


# --------------------------------------------------------------------------- #
# Selector
# --------------------------------------------------------------------------- #

class ContextSelector:
    def __init__(self, repo: Any, index: RepoIndex):
        self.repo = repo
        self.index = index

    # ---- public ------------------------------------------------------------ #

    def select(
        self,
        query: str = "",
        budget_tokens: int = DEFAULT_BUDGET,
        focus_files: Optional[Iterable] = None,
        include_tests: bool = False,
        include_map: bool = True,
        neighbour_depth: int = 1,
        max_candidates: int = 12,
        exclude: Optional[Iterable] = None,
        refresh: bool = False,
    ) -> SelectedContext:
        """
        Pick context for `query` within `budget_tokens` (a hard limit on the estimated total).

        focus_files    files that MUST come first (e.g. the files a revision touches)
        include_tests  also consider test files (and tests for the selected files)
        include_map    prepend a compact outline of the whole project
        refresh        re-sync the index with the repository first (cheap: only changed files re-parse)
        """
        if refresh:
            self.index.update(self.repo)
        budget = max(0, int(budget_tokens))
        ctx = SelectedContext(budget_tokens=budget, query=query)
        excluded = set(exclude or [])
        focus = [p for p in dict.fromkeys(focus_files or []) if p in self.index and p not in excluded]

        candidates = self._candidates(query, focus, include_tests, neighbour_depth, max_candidates, excluded)

        remaining = budget
        if include_map and len(self.index) and budget >= 400:
            ctx.project_map = self._project_map(int(budget * MAP_SHARE), excluded)
            remaining -= estimate_tokens(ctx.project_map)

        words = set(_tokens(query))
        for path, score, reason in candidates:
            if remaining < MIN_USEFUL_TOKENS:
                ctx.omitted.append((path, "budget exhausted"))
                continue
            chosen = self._fit(path, words, score, reason, remaining, forced=path in focus, budget=budget)
            if chosen is None:
                ctx.omitted.append((path, "does not fit in the remaining budget"))
                continue
            ctx.files.append(chosen)
            remaining -= chosen.tokens

        logger.debug("Selected context for %r: %s", query[:60], ctx.summary())
        return ctx

    # ---- candidates -------------------------------------------------------- #

    def _candidates(self, query: str, focus: list, include_tests: bool, depth: int,
                    limit: int, excluded: set) -> list:
        """[(path, score, reason)] best first. Focus files outrank everything."""
        scores: dict = {}
        reasons: dict = {}

        def add(path: str, score: float, reason: str) -> None:
            if path in excluded or path not in self.index:
                return
            if not include_tests and self.index.get(path).is_test and path not in focus:
                return
            if score > scores.get(path, 0.0):
                scores[path], reasons[path] = score, reason

        hits = self.index.search(query, limit=limit, include_tests=include_tests) if query.strip() else []
        top = max((s for _, s in hits), default=1.0)
        big = (top + 1.0) * 10                                  # focus files always beat search hits
        for i, path in enumerate(focus):
            add(path, big - i * 0.01, "focus file")
        for path, score in hits:
            add(path, score, f"matches query (score {score})")

        seeds = list(focus) + [p for p, _ in hits[:3]]
        frontier = {p: scores[p] for p in seeds if p in scores}
        for hop in range(max(0, depth)):
            nxt: dict = {}
            for path, base in frontier.items():
                for dep in self.index.dependencies(path):
                    nxt[dep] = max(nxt.get(dep, 0.0), base * NEIGHBOUR_DECAY)
                for dep in self.index.dependents(path):
                    nxt[dep] = max(nxt.get(dep, 0.0), base * NEIGHBOUR_DECAY * 0.8)   # importers slightly less
            for dep, score in sorted(nxt.items()):
                if dep not in scores:
                    add(dep, score, "imported by / imports a selected file")
            frontier = {p: s for p, s in nxt.items() if p in scores}

        if include_tests:
            for path in seeds:
                for test in self.index.tests_for(path):
                    add(test, scores.get(path, 1.0) * 0.4, f"test for {path}")

        ranked = sorted(scores, key=lambda p: (-scores[p], p))
        keep = [p for p in ranked if p in focus] + [p for p in ranked if p not in focus][: max(0, limit)]
        keep.sort(key=lambda p: (-scores[p], p))
        return [(p, scores[p], reasons[p]) for p in keep]

    # ---- fitting one file ---------------------------------------------------- #

    def _read(self, path: str) -> Optional[str]:
        try:
            return self.repo.read_file(path, errors="replace")
        except Exception as e:                       # vanished / unreadable / binary: skip, never crash
            logger.debug("Cannot read %s for context: %s", path, e)
            return None

    def _fit(self, path: str, words: set, score: float, reason: str, remaining: int,
             forced: bool, budget: int) -> Optional[SelectedFile]:
        text = self._read(path)
        if text is None:
            return None
        fi: FileIndex = self.index.get(path)
        full_cap = remaining if forced else min(remaining, max(MIN_USEFUL_TOKENS, int(budget * MAX_FILE_SHARE)))

        full = estimate_tokens(text)
        if full <= full_cap:
            return SelectedFile(path, FULL, text, full, score, reason)

        snippet = self._snippets(fi, text, words, remaining)
        if snippet:
            return SelectedFile(path, SNIPPETS, snippet, estimate_tokens(snippet), score, reason)

        outline = self._outline(fi, remaining)
        if outline:
            return SelectedFile(path, OUTLINE, outline, estimate_tokens(outline), score, reason)
        return None

    @staticmethod
    def _outline(fi: FileIndex, max_tokens: int) -> str:
        """Signatures only: what the file offers, without bodies."""
        lines = [f"# {fi.summary}"] if fi.summary else []
        for s in fi.symbols:
            indent = "    " if s.parent else ""
            if s.kind in ("function", "method"):
                lines.append(f"{indent}def {s.name}{s.signature}  # L{s.line}")
            elif s.kind == "class":
                bases = f"({', '.join(s.bases)})" if s.bases else ""
                lines.append(f"{indent}class {s.name}{bases}  # L{s.line}")
            else:
                lines.append(f"{indent}{s.kind} {s.name}  # L{s.line}")
        out, used = [], 0
        for line in lines:
            cost = estimate_tokens(line) + 1
            if used + cost > max_tokens:
                out.append("# ... (more symbols omitted)")
                break
            out.append(line)
            used += cost
        text = "\n".join(out)
        return text if text and estimate_tokens(text) <= max_tokens else ""

    def _snippets(self, fi: FileIndex, text: str, words: set, max_tokens: int) -> str:
        """
        The parts of a big file that matter: the import header plus every symbol whose name or
        docstring matches the query (methods bring their class header). Falls back to '' when nothing
        matches, so the caller can use the outline instead.
        """
        if not words:
            return ""
        lines = text.splitlines()
        symbols = fi.symbols
        matched = []
        for i, s in enumerate(symbols):
            hay = set(_tokens(s.qualified_name)) | set(_tokens(s.doc))
            if hay & words and s.kind != "variable":
                matched.append(i)
        if not matched:
            return ""

        ranges = []                                             # (start, end) 1-based inclusive
        for i in matched:
            s = symbols[i]
            ranges.append((s.line, self._end_line(fi, i, len(lines))))
            if s.parent:                                        # method: include the class line for orientation
                for c in symbols:
                    if c.kind == "class" and c.name == s.parent and c.line < s.line:
                        ranges.append((c.line, c.line))
                        break
        first_symbol = min((s.line for s in symbols), default=len(lines) + 1)
        header_end = min(first_symbol - 1, 30)
        if header_end >= 1:
            ranges.append((1, header_end))

        merged: list = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        out, used = [], 0
        for start, end in merged:
            end = min(end, len(lines))
            block = "\n".join(lines[start - 1:end])
            cost = estimate_tokens(block) + 8
            gap = f"# ... lines {start}-{end} of {fi.lines} ..."
            if used + cost > max_tokens:
                # truncate the last block rather than dropping it entirely, if it's the first one
                if not out:
                    keep_chars = max(0, (max_tokens - 12) * 4)
                    block = block[:keep_chars].rsplit("\n", 1)[0]
                    if block.strip():
                        out.append(f"{gap}\n{block}\n# ... (truncated)")
                break
            out.append(f"{gap}\n{block}")
            used += cost
        text_out = "\n".join(out)
        return text_out if text_out and estimate_tokens(text_out) <= max_tokens else ""

    @staticmethod
    def _end_line(fi: FileIndex, i: int, total: int) -> int:
        s = fi.symbols[i]
        if s.end_line and s.end_line > s.line:
            return s.end_line
        # regex-indexed: run until the next symbol starts (bounded)
        nxt = fi.symbols[i + 1].line - 1 if i + 1 < len(fi.symbols) else total
        return max(s.line, min(nxt, s.line + NON_PY_SNIPPET_LINES, total))

    # ---- project map ----------------------------------------------------------- #

    def _project_map(self, max_tokens: int, excluded: set) -> str:
        """Compact outline of the repo, most important files first, trimmed to `max_tokens`."""
        paths = [p for p in self.index.ranked_files() if p not in excluded]
        text = self.index.outline(paths=paths, max_chars=max(0, max_tokens * 4), include_methods=False)
        return text if estimate_tokens(text) <= max_tokens else text[: max_tokens * 4]


# --------------------------------------------------------------------------- #
# Self-test:  python -m context.selector     (no network, no API keys)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from context.repository import Repository
    from core.logger import setup_logging

    setup_logging("ERROR")

    BIG_BODY = "".join(f"    x{i} = {i}  # filler line to make this file large\n" for i in range(220))

    FILES = {
        "auth/jwt.py": '"""JWT token creation and refresh."""\nimport time\n\n'
                       'def create_token(user):\n    """Create a signed token."""\n    return f"tok-{user}"\n\n'
                       'def refresh_token(token):\n    """Refresh an expired token."""\n    return token + "!"\n',
        "auth/routes.py": '"""HTTP routes for login."""\nfrom auth.jwt import create_token, refresh_token\n\n'
                          'def login(user):\n    return create_token(user)\n',
        "auth/middleware.py": '"""Checks tokens on every request."""\nfrom auth.jwt import refresh_token\n\n'
                              'def check(request):\n    return refresh_token(request)\n',
        "billing/invoice.py": '"""Invoices."""\ndef make_invoice(order):\n    return order\n',
        "billing/tax.py": '"""Tax rules."""\ndef tax(amount):\n    return amount * 0.2\n',
        "utils/big.py": '"""Large helper module."""\nimport os\n\n'
                        'def parse_config(path):\n    """Parse the config file."""\n    return {}\n\n'
                        'def unrelated_function():\n' + BIG_BODY + '\n'
                        'def render_chart(data):\n    """Draw a chart."""\n    return data\n',
        "tests/test_jwt.py": "from auth.jwt import create_token\ndef test_create(): assert create_token('a')\n",
        "main.py": '"""App entry."""\nfrom auth.routes import login\nif __name__ == "__main__":\n    login("a")\n',
    }

    tmp = tempfile.TemporaryDirectory()
    repo = Repository(Path(tmp.name) / "p")
    repo.write_files(FILES)
    index = RepoIndex()
    index.update(repo)
    sel = ContextSelector(repo, index)

    # 1. Query-driven selection: relevant files in, unrelated files out; tests excluded by default
    ctx = sel.select("fix the jwt token refresh", budget_tokens=3000)
    assert ctx.paths[0] == "auth/jwt.py", ctx.paths
    assert "billing/invoice.py" not in ctx.paths and "tests/test_jwt.py" not in ctx.paths
    assert ctx.files[0].mode == FULL and "def refresh_token" in ctx.files[0].content
    print("PASS 1: relevant files first ->", ctx.paths[:4])

    # 2. Neighbours in the import graph come along (importers of the best match)
    assert {"auth/routes.py", "auth/middleware.py"} <= set(ctx.paths), ctx.paths
    assert any(f.path == "auth/routes.py" and "imports" in f.reason for f in ctx.files)
    print("PASS 2: import-graph neighbours included")

    # 3. The budget is a hard limit at every size, and small budgets degrade rather than overflow
    for b in (0, 50, 120, 300, 600, 1500, 3000, 8000):
        c = sel.select("fix the jwt token refresh", budget_tokens=b)
        assert c.tokens <= b, (b, c.tokens, c.summary())
        assert estimate_tokens(c.render()) <= b + 40, (b, estimate_tokens(c.render()))     # headers are tiny
    tight = sel.select("jwt token refresh", budget_tokens=250, include_map=False)
    assert tight.tokens <= 250 and tight.files, tight.summary()
    print("PASS 3: budget never exceeded (0..8000 tokens); tight ->", tight.summary())

    # 4. A big file is cut down to the relevant snippets (header + matching functions), not sent whole
    c = sel.select("parse the config file", budget_tokens=700, include_map=False)
    big = next(f for f in c.files if f.path == "utils/big.py")
    assert big.mode == SNIPPETS and "def parse_config" in big.content
    assert "unrelated_function" not in big.content and "x100 = 100" not in big.content
    assert "import os" in big.content and "lines" in big.content
    full_cost = estimate_tokens(repo.read_file("utils/big.py"))
    assert big.tokens < full_cost / 3
    print(f"PASS 4: big file -> snippets ({big.tokens} tokens instead of {full_cost})")

    # 5. No keyword match inside a big file -> signatures only (outline)
    c2 = sel.select("zzz", budget_tokens=400, focus_files=["utils/big.py"], include_map=False)
    f2 = c2.files[0]
    assert f2.mode == OUTLINE and "def render_chart" in f2.content and "x100" not in f2.content, f2.mode
    print("PASS 5: file without keyword matches -> outline of signatures")

    # 6. Focus files are forced to the top; their tests and dependents follow when asked
    c = sel.select("review the changes", focus_files=["billing/tax.py"], include_tests=True, budget_tokens=3000)
    assert c.paths[0] == "billing/tax.py" and c.files[0].reason == "focus file"
    c = sel.select("review", focus_files=["auth/jwt.py"], include_tests=True, budget_tokens=3000)
    assert "tests/test_jwt.py" in c.paths and any("test for" in f.reason or "imports" in f.reason
                                                  for f in c.files if f.path == "tests/test_jwt.py")
    print("PASS 6: focus files first; tests included on request ->", c.paths)

    # 7. Project map: present, small, lists files, and honours the budget share
    c = sel.select("billing tax", budget_tokens=2000)
    assert c.project_map and "auth/jwt.py" in c.project_map
    assert estimate_tokens(c.project_map) <= 2000 * MAP_SHARE + 2
    assert not sel.select("billing tax", budget_tokens=2000, include_map=False).project_map
    assert "=== PROJECT MAP ===" in c.render() and "=== billing/tax.py [full] ===" in c.render()
    print("PASS 7: project map included within its share ->", estimate_tokens(c.project_map), "tokens")

    # 8. Exclusions, omitted reporting, JSON-serialisable summary
    c = sel.select("jwt token", budget_tokens=3000, exclude=["auth/jwt.py"])
    assert "auth/jwt.py" not in c.paths
    c = sel.select("jwt token refresh login billing", budget_tokens=170, include_map=False, max_candidates=8)
    assert c.omitted and all(isinstance(o, tuple) for o in c.omitted)
    json.dumps(c.to_dict())
    print("PASS 8: exclude works; omitted files reported ->", len(c.omitted), "omitted")

    # 9. Deterministic
    a = sel.select("jwt token refresh", budget_tokens=1200).render()
    b = sel.select("jwt token refresh", budget_tokens=1200).render()
    assert a == b
    print("PASS 9: same inputs -> identical context")

    # 10. Robust: empty query, empty repo, vanished file, stale index, refresh
    assert sel.select("", budget_tokens=1000).files == []                       # nothing to rank by
    assert sel.select("", budget_tokens=1000, focus_files=["main.py"]).paths[0] == "main.py"
    empty = ContextSelector(Repository(Path(tmp.name) / "empty"), RepoIndex())
    assert empty.select("anything").files == [] and empty.select("anything").project_map == ""
    (Path(tmp.name) / "p" / "billing" / "tax.py").unlink()                       # index is now stale
    c = sel.select("tax rules", budget_tokens=1000)                              # must not crash
    assert "billing/tax.py" not in c.paths
    repo.write_file("billing/discount.py", '"""Discount rules."""\ndef discount(x):\n    return x\n')
    assert "billing/discount.py" not in sel.select("discount rules", budget_tokens=1000).paths
    c = sel.select("discount rules", budget_tokens=1000, refresh=True)
    assert c.paths[0] == "billing/discount.py" and "billing/tax.py" not in c.paths
    print("PASS 10: empty query/repo, vanished file and stale index tolerated; refresh=True re-syncs")

    # 11. Non-python (regex-indexed) big file still yields bounded snippets
    js = "import fs from 'fs';\n\nexport function loadSettings() {\n  return 1;\n}\n\n" + \
         "".join(f"export function filler{i}() {{\n  return {i};\n}}\n" for i in range(150)) + \
         "export function saveSettings() {\n  return 2;\n}\n"
    repo.write_file("web/settings.ts", js)
    index.update(repo)
    c = sel.select("load settings", budget_tokens=500, include_map=False)
    f = next(x for x in c.files if x.path == "web/settings.ts")
    assert f.mode == SNIPPETS and "loadSettings" in f.content and "filler100" not in f.content and c.tokens <= 500
    print("PASS 11: TypeScript file -> bounded snippets")