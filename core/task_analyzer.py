"""
core/task_analyzer.py

Turns a natural-language requirement into a structured TaskAnalysis:
task type, complexity (0-10), languages, UI/database/network needs, a rough token plan,
and the decision "is multi-model orchestration worth it?".

Two analysis modes that work together:
  * heuristic (always runs, free, deterministic): keyword/pattern based
  * LLM (optional): any callable ModelRequest -> ModelResponse, e.g.
        analyzer.analyze(req, generate=lambda r: registry.generate("local-qwen", r))
    The scheduler decides WHICH model to spend on this; the analyzer never picks one itself.

The LLM answer is validated and blended with the heuristic estimate (70% / 30% on complexity).
If the model fails, returns garbage, or is unavailable, the heuristic result is used instead.
Analysis never raises because of a model failure.

Save the result as task_spec.json with TaskAnalysis.to_dict(); reload with TaskAnalysis.from_dict().
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Optional

from config.settings import OrchestratorSettings
from core.logger import get_logger
from models.base import ModelError, ModelRequest, ModelResponse

logger = get_logger("core.task_analyzer")

GenerateFn = Callable[[ModelRequest], ModelResponse]

# The first four match the keys of TASK_TYPE_WEIGHTS in config/settings.py
TASK_TYPES = (
    "competitive_programming", "backend_api", "ml_project", "desktop_application",
    "web_app", "cli_tool", "data_pipeline", "library", "script", "general",
)

LLM_COMPLEXITY_WEIGHT = 0.7            # blend: 70% model estimate, 30% heuristic estimate
ANALYSIS_MAX_TOKENS = 700
MAX_PROMPT_REQUIREMENT_CHARS = 6000
DEFAULT_LANGUAGE = "python"


# --------------------------------------------------------------------------- #
# Keyword tables (heuristic analysis)
# --------------------------------------------------------------------------- #

def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# task type -> [(pattern, weight)]; the highest total wins, and it needs a total of at least 2
_TYPE_PATTERNS = {
    "competitive_programming": [
        (_rx(r"competitive programming|leetcode|codeforces|hackerrank|atcoder|time limit"), 4),
        (_rx(r"dynamic programming|\bdp\b|shortest path|\bgraph problem|\bknapsack|\bsubarray"), 2),
        (_rx(r"test cases?|constraints?|\bo\(n"), 1),
    ],
    "ml_project": [
        (_rx(r"machine learning|\bml\b|neural net|deep learning"), 3),
        (_rx(r"pytorch|tensorflow|scikit-learn|sklearn|keras|hugging ?face"), 3),
        (_rx(r"classif(y|ier|ication)|regression|clustering|\bnlp\b|computer vision|\bllm\b"), 2),
        (_rx(r"\bdataset\b|training data|model accuracy|train(ing)? (a|the|my) model"), 1),
    ],
    "desktop_application": [
        (_rx(r"desktop (app|application|program|tool)"), 4),
        (_rx(r"\bgui\b|tkinter|pyqt|pyside|electron|wxpython|kivy|customtkinter"), 3),
        (_rx(r"system tray|tray icon|native window"), 2),
    ],
    "web_app": [
        (_rx(r"web ?app(lication)?|web ?site|landing page"), 3),
        (_rx(r"front-?end|\breact\b|\bvue\b|angular|svelte|next\.?js"), 3),
        (_rx(r"\bdashboard\b|\bhtml\b|\bcss\b"), 1),
    ],
    "backend_api": [
        (_rx(r"fastapi|flask|django|express\.?js|spring boot|graphql"), 3),
        (_rx(r"\brest(ful)?\b|\bapi\b|endpoints?|back-?end|microservices?"), 2),
        (_rx(r"\bserver\b|\bjwt\b|oauth|\bcrud\b|websockets?"), 1),
    ],
    "cli_tool": [
        (_rx(r"\bcli\b|command[- ]line"), 3),
        (_rx(r"terminal (tool|app|utility)|argparse|\btyper\b|subcommands?"), 2),
    ],
    "data_pipeline": [
        (_rx(r"data pipeline|\betl\b|data (cleaning|processing|ingestion|transformation)"), 3),
        (_rx(r"scrap(e|er|ing)|crawler"), 2),
        (_rx(r"\bcsv\b|\bparquet\b|\bpandas\b"), 1),
    ],
    "library": [
        (_rx(r"pip install|pypi|npm package|publish(ed)? (a|as a) (package|library)"), 3),
        (_rx(r"\blibrary\b|\bsdk\b|\bpackage\b"), 2),
    ],
    "script": [
        (_rx(r"\bscript\b"), 2),
        (_rx(r"\bautomat(e|ion)\b|one[- ]off|\brename\b|convert (the )?files?"), 1),
    ],
}

# Starting complexity per task type (before signals from the text)
_TYPE_BASE = {
    "competitive_programming": 3.0, "ml_project": 4.0, "desktop_application": 3.5, "web_app": 3.5,
    "backend_api": 3.0, "cli_tool": 2.0, "data_pipeline": 3.0, "library": 2.5, "script": 1.0,
    "general": 2.0,
}

_LANGUAGE_PATTERNS = {
    "python": _rx(r"\bpython3?\b|\bpy\b|\bdjango\b|\bflask\b|\bfastapi\b|\bpandas\b|\bpytorch\b|tkinter|pyqt"),
    "javascript": _rx(r"\bjavascript\b|\bjs\b|\bnode\.?js\b|\bexpress\.?js\b|\breact\b|\bvue\b"),
    "typescript": _rx(r"\btypescript\b|\bts\b"),
    "java": _rx(r"\bjava\b|\bspring boot\b"),
    "csharp": _rx(r"c#|\bcsharp\b|\.net\b|\bdotnet\b"),
    "cpp": _rx(r"c\+\+|\bcpp\b"),
    "go": _rx(r"\bgolang\b|\bin go\b|\bgo (service|app|program|backend|cli|server)\b"),
    "rust": _rx(r"\brust\b"),
    "kotlin": _rx(r"\bkotlin\b"),
    "swift": _rx(r"\bswift\b"),
    "php": _rx(r"\bphp\b|\blaravel\b"),
    "ruby": _rx(r"\bruby\b|\brails\b"),
}

_UI_RX = _rx(r"\bui\b|\bgui\b|user interface|desktop (app|application)|web ?app|web ?site|dashboard|"
             r"front-?end|tkinter|pyqt|pyside|electron|\breact\b|\bvue\b|angular|\bbuttons?\b|"
             r"\bwindow\b|mobile app")
_DB_RX = _rx(r"database|\bsql\b|sqlite|postgres(ql)?|mysql|mongo(db)?|\bredis\b|\borm\b|\bcrud\b|"
             r"persist(ent|ence)?")
_NET_RX = _rx(r"\bapi\b|\brest(ful)?\b|endpoints?|\bhttp\b|\bserver\b|websockets?|\bscrap|webhook|\bsockets?\b")

# Each distinct pattern that matches adds to the complexity estimate
_HEAVY_RX = [_rx(p) for p in (
    r"distributed", r"event[- ]driven", r"microservices?", r"kubernetes|\bk8s\b", r"scalab",
    r"concurren", r"real[- ]?time", r"multi[- ]?tenant", r"load[- ]balanc",
    r"machine learning|neural net|deep learning", r"encrypt", r"payments?|billing|stripe",
    r"oauth|\bsso\b", r"role[- ]based|\brbac\b", r"high[- ]availability|fault[- ]toleran",
    r"\bcompiler\b|\binterpreter\b",
)]
_MEDIUM_RX = [_rx(p) for p in (
    r"authenticat|authoriz|\blogin\b|sign[- ]?up", r"multi[- ]?user", r"\bqueues?\b", r"\bcach(e|ing)\b",
    r"\bdocker", r"plug-?ins?", r"schedul", r"notification", r"monitor",
    r"background (job|task|worker|process)", r"websockets?", r"\bsync", r"\badmin\b", r"upload",
    r"\bsearch\b", r"\breports?\b", r"\basync", r"\bwebhook", r"\bemail", r"\bpdf\b",
)]
_SIMPLE_RX = [_rx(p) for p in (
    r"\bsimple\b", r"\bbasic\b", r"\bsmall\b", r"\btiny\b", r"\bquick(ly)?\b", r"\bminimal\b",
    r"\bhello world\b", r"\btoy\b", r"\bsingle (function|file)\b", r"\bone[- ](function|file|liner)\b",
)]
_ENUMERATION_RX = _rx(r",|;|\band\b|\n\s*(?:[-*\u2022]|\d+[.)])")
_BULLET_RX = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.+)$", re.MULTILINE)

_LANGUAGE_ALIASES = {
    "js": "javascript", "node": "javascript", "nodejs": "javascript", "ts": "typescript",
    "c++": "cpp", "c#": "csharp", ".net": "csharp", "golang": "go", "py": "python",
    "python3": "python", "kt": "kotlin",
}
_TASK_TYPE_ALIASES = {
    "web_application": "web_app", "webapp": "web_app", "website": "web_app", "api": "backend_api",
    "rest_api": "backend_api", "backend": "backend_api", "desktop_app": "desktop_application",
    "desktop": "desktop_application", "gui_application": "desktop_application", "cli": "cli_tool",
    "command_line_tool": "cli_tool", "machine_learning": "ml_project", "ml": "ml_project",
    "etl": "data_pipeline", "scraper": "data_pipeline", "package": "library",
    "algorithm": "competitive_programming", "algorithms": "competitive_programming",
}


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class TaskAnalysis:
    requirement: str
    summary: str
    task_type: str
    complexity: float                    # 0..10 (final, blended)
    languages: list                      # best fit first
    primary_language: str
    needs_ui: bool
    needs_database: bool
    needs_network: bool
    key_features: list
    risks: list
    needs_orchestration: bool
    orchestration_reason: str
    token_plan: dict                     # rough token estimates, see estimate_token_plan()
    source: str                          # "heuristic" | "llm+heuristic"
    confidence: float                    # 0..1
    assumptions: list
    heuristic_complexity: float
    llm_complexity: Optional[float] = None
    complexity_breakdown: dict = field(default_factory=dict)
    analyzer_model: Optional[str] = None
    analysis_tokens: int = 0             # tokens spent on the LLM analysis itself (0 = none)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "TaskAnalysis":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def short(self) -> str:
        return (f"{self.task_type} | complexity {self.complexity:.1f} | {self.primary_language} | "
                f"orchestrate={self.needs_orchestration} | source={self.source}")


# --------------------------------------------------------------------------- #
# Token plan
# --------------------------------------------------------------------------- #

def estimate_token_plan(
    complexity: float,
    settings: OrchestratorSettings,
    n_candidates: Optional[int] = None,
) -> dict:
    """
    Rough planning numbers (not promises): what each pipeline stage is likely to cost,
    and what orchestration costs compared with asking one model directly.
    """
    n = settings.max_benchmark_candidates if n_candidates is None else max(0, n_candidates)
    files = max(1, round(1 + complexity * 1.6))
    hld = int(1500 + 400 * complexity)
    coding = files * 900
    rounds = max(0, min(2, settings.max_review_iterations))      # typical, not worst case
    review = rounds * (1200 + files * 350) + max(0, rounds - 1) * files * 300
    per_candidate = min(settings.lite_task_token_budget, int(2500 + 600 * complexity))
    benchmark = per_candidate * n
    return {
        "estimated_files": files,
        "hld": hld,
        "coding": coding,
        "review": review,
        "benchmark_per_candidate": per_candidate,
        "benchmark_candidates": n,
        "benchmark_total": benchmark,
        "direct_total": coding,
        "orchestrated_total": benchmark + hld + coding + review,
    }


# --------------------------------------------------------------------------- #
# JSON / validation helpers
# --------------------------------------------------------------------------- #

def extract_json_object(text: Optional[str]) -> Optional[dict]:
    """Find the first valid JSON object in model output (handles code fences, prose, <think> text)."""
    if not text:
        return None
    cleaned = text.strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except ValueError:
        pass

    start = cleaned.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(cleaned[start:i + 1])
                        if isinstance(data, dict):
                            return data
                    except ValueError:
                        pass
                    break
        start = cleaned.find("{", start + 1)
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _as_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes"):
            return True
        if lowered in ("false", "no"):
            return False
    return None


def _as_str_list(value, limit: int) -> list:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items = [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()]
    return items[:limit]


def _normalize_language(name: str) -> str:
    lowered = name.strip().lower()
    return _LANGUAGE_ALIASES.get(lowered, lowered)


def _normalize_task_type(name) -> Optional[str]:
    if not isinstance(name, str):
        return None
    key = re.sub(r"[\s-]+", "_", name.strip().lower())
    key = _TASK_TYPE_ALIASES.get(key, key)
    return key if key in TASK_TYPES else None


def _validate_llm_fields(data: dict) -> Optional[dict]:
    """Keep only well-formed values from the model's JSON. Missing/invalid values become None/empty."""
    complexity = _as_float(data.get("complexity"))
    fields_ = {
        "summary": str(data["summary"]).strip()[:300] if isinstance(data.get("summary"), str) else "",
        "task_type": _normalize_task_type(data.get("task_type")),
        "complexity": None if complexity is None else _clamp(complexity, 0.0, 10.0),
        "languages": list(dict.fromkeys(
            _normalize_language(x) for x in _as_str_list(data.get("languages"), 5))),
        "needs_ui": _as_bool(data.get("needs_ui")),
        "needs_database": _as_bool(data.get("needs_database")),
        "needs_network": _as_bool(data.get("needs_network")),
        "key_features": _as_str_list(data.get("key_features"), 8),
        "risks": _as_str_list(data.get("risks"), 5),
    }
    if fields_["task_type"] is None and fields_["complexity"] is None:
        return None        # nothing trustworthy in the answer
    return fields_


# --------------------------------------------------------------------------- #
# LLM prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = (
    "You are a senior software architect who estimates how hard a software request is. "
    "Respond with ONLY a JSON object: no prose, no code fences."
)

_PROMPT_TEMPLATE = """Analyze this software requirement.

REQUIREMENT:
<<<
__REQUIREMENT__
>>>

Return a JSON object with exactly these keys:
{
  "summary": "one sentence describing what must be built",
  "task_type": one of __TYPES__,
  "complexity": a number from 0 to 10 (1 = trivial one-liner, 3 = small script, 5 = typical multi-file app, 8 = large system with concurrency, scaling or security concerns, 10 = research-grade),
  "languages": ["best-fit language first", ...],
  "needs_ui": true or false,
  "needs_database": true or false,
  "needs_network": true or false,
  "key_features": ["up to 8 short items"],
  "risks": ["up to 5 things likely to be hard or ambiguous"]
}"""


def _build_prompt(requirement: str) -> str:
    text = requirement if len(requirement) <= MAX_PROMPT_REQUIREMENT_CHARS \
        else requirement[:MAX_PROMPT_REQUIREMENT_CHARS] + "\n[...truncated...]"
    return (_PROMPT_TEMPLATE
            .replace("__TYPES__", json.dumps(list(TASK_TYPES)))
            .replace("__REQUIREMENT__", text))


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #

class TaskAnalyzer:
    def __init__(self, settings: OrchestratorSettings):
        self.settings = settings

    # ---- public ------------------------------------------------------------ #

    def analyze(
        self,
        requirement: str,
        generate: Optional[GenerateFn] = None,
        force_orchestration: Optional[bool] = None,
        n_candidates: Optional[int] = None,
    ) -> TaskAnalysis:
        """
        Analyze a requirement.

        generate:             optional callable (ModelRequest -> ModelResponse) used for the LLM pass
        force_orchestration:  True/False overrides the complexity-threshold decision
        n_candidates:         how many models would compete in the benchmark (for the token plan)
        """
        requirement = (requirement or "").strip()
        if not requirement:
            raise ValueError("Requirement is empty")

        heuristic = self._heuristic(requirement)

        llm, response = None, None
        if generate is not None:
            try:
                llm, response = self._llm_analysis(requirement, generate)
            except Exception as e:      # a model failure must never break analysis
                logger.warning("LLM task analysis failed (%s: %s); using heuristics only",
                               type(e).__name__, e)

        analysis = self._merge(requirement, heuristic, llm, response, force_orchestration, n_candidates)
        logger.info("Task analysis: %s", analysis.short())
        return analysis

    # ---- LLM pass ---------------------------------------------------------- #

    def _llm_analysis(self, requirement: str, generate: GenerateFn):
        request = ModelRequest(
            prompt=_build_prompt(requirement),
            system=_SYSTEM_PROMPT,
            max_tokens=ANALYSIS_MAX_TOKENS,
            temperature=0.0,
            purpose="task_analysis",
        )
        response = generate(request)
        data = extract_json_object(getattr(response, "text", None))
        if data is None:
            logger.warning("Task analysis reply was not valid JSON; using heuristics only")
            return None, response
        fields_ = _validate_llm_fields(data)
        if fields_ is None:
            logger.warning("Task analysis reply had no usable fields; using heuristics only")
        return fields_, response

    # ---- heuristic pass ---------------------------------------------------- #

    def _heuristic(self, text: str) -> dict:
        # task type
        best_type, best_score = "general", 0
        for task_type, patterns in _TYPE_PATTERNS.items():
            score = sum(weight for rx, weight in patterns if rx.search(text))
            if score > best_score:
                best_type, best_score = task_type, score
        task_type = best_type if best_score >= 2 else "general"

        # languages, ordered by where they first appear in the text
        found = []
        for language, rx in _LANGUAGE_PATTERNS.items():
            match = rx.search(text)
            if match:
                found.append((match.start(), language))
        languages = [lang for _, lang in sorted(found)]
        languages_explicit = bool(languages)
        if not languages:
            languages = [DEFAULT_LANGUAGE]

        needs_ui = bool(_UI_RX.search(text))
        needs_database = bool(_DB_RX.search(text))
        needs_network = bool(_NET_RX.search(text))

        # complexity
        words = len(text.split())
        breakdown = {
            "type_base": _TYPE_BASE[task_type],
            "heavy_signals": min(4.8, 1.2 * sum(1 for rx in _HEAVY_RX if rx.search(text))),
            "medium_signals": min(2.4, 0.6 * sum(1 for rx in _MEDIUM_RX if rx.search(text))),
            "components": 0.8 * (needs_ui + needs_database + needs_network),
            "length": min(2.0, words / 60),
            "enumeration": min(1.5, 0.25 * len(_ENUMERATION_RX.findall(text))),
            "simplicity": -min(2.5, 1.0 * sum(1 for rx in _SIMPLE_RX if rx.search(text))),
        }
        complexity = round(_clamp(sum(breakdown.values()), 0.5, 10.0), 1)

        return {
            "task_type": task_type,
            "complexity": complexity,
            "languages": languages,
            "languages_explicit": languages_explicit,
            "needs_ui": needs_ui,
            "needs_database": needs_database,
            "needs_network": needs_network,
            "key_features": [m.strip() for m in _BULLET_RX.findall(text)][:8],
            "breakdown": {k: round(v, 2) + 0.0 for k, v in breakdown.items()},
            "words": words,
        }

    # ---- merge and decide -------------------------------------------------- #

    def _merge(self, requirement, h, llm, response, force_orchestration, n_candidates) -> TaskAnalysis:
        llm = llm or {}
        assumptions = []

        task_type = llm.get("task_type") or h["task_type"]
        llm_complexity = llm.get("complexity")
        if llm_complexity is not None:
            complexity = LLM_COMPLEXITY_WEIGHT * llm_complexity + (1 - LLM_COMPLEXITY_WEIGHT) * h["complexity"]
        else:
            complexity = h["complexity"]
        complexity = round(_clamp(complexity, 0.0, 10.0), 1)

        languages = llm.get("languages") or h["languages"]
        if not llm.get("languages") and not h["languages_explicit"]:
            assumptions.append(f"no language specified: assuming {DEFAULT_LANGUAGE}")

        def pick(key):
            return llm[key] if llm.get(key) is not None else h[key]

        summary = llm.get("summary") or (re.split(r"(?<=[.!?])\s", requirement, maxsplit=1)[0][:140])
        key_features = llm.get("key_features") or h["key_features"]

        if h["words"] < 6:
            assumptions.append("requirement is very short: details may be missing")
        if response is not None and not llm:
            assumptions.append("model analysis failed or was unusable: keyword heuristics were used")
        if response is None:
            assumptions.append("analysis is keyword-based: no model was used")

        needs_orchestration, reason = self._decide(complexity, force_orchestration)

        # confidence: low for pure keyword analysis, higher when a model agrees with the heuristics
        if llm:
            confidence = 0.8
            if llm_complexity is not None and abs(llm_complexity - h["complexity"]) > 3:
                confidence -= 0.15
            if llm.get("task_type") and llm["task_type"] != h["task_type"] and h["task_type"] != "general":
                confidence -= 0.05
        else:
            confidence = 0.4 + (0.1 if task_type != "general" else 0) + (0.1 if h["languages_explicit"] else 0)

        return TaskAnalysis(
            requirement=requirement,
            summary=summary,
            task_type=task_type,
            complexity=complexity,
            languages=languages,
            primary_language=languages[0],
            needs_ui=pick("needs_ui"),
            needs_database=pick("needs_database"),
            needs_network=pick("needs_network"),
            key_features=key_features,
            risks=llm.get("risks", []),
            needs_orchestration=needs_orchestration,
            orchestration_reason=reason,
            token_plan=estimate_token_plan(complexity, self.settings, n_candidates),
            source="llm+heuristic" if llm else "heuristic",
            confidence=round(_clamp(confidence, 0.0, 1.0), 2),
            assumptions=assumptions,
            heuristic_complexity=h["complexity"],
            llm_complexity=llm_complexity,
            complexity_breakdown=h["breakdown"],
            analyzer_model=getattr(response, "model_id", None) if response is not None else None,
            analysis_tokens=getattr(response, "total_tokens", 0) if response is not None else 0,
        )

    def _decide(self, complexity: float, force: Optional[bool]) -> tuple:
        threshold = self.settings.min_complexity_for_orchestration
        if force is True:
            return True, f"orchestration forced on (complexity {complexity:.1f}, threshold {threshold:.1f})"
        if force is False:
            return False, f"orchestration forced off (complexity {complexity:.1f}, threshold {threshold:.1f})"
        if complexity >= threshold:
            return True, f"complexity {complexity:.1f} >= threshold {threshold:.1f}: competitive benchmarking is worth the tokens"
        return False, f"complexity {complexity:.1f} < threshold {threshold:.1f}: a single model is enough"


# --------------------------------------------------------------------------- #
# Self-test (no network, no API key): python -m core.task_analyzer
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from core.logger import setup_logging

    setup_logging("ERROR")
    settings = OrchestratorSettings()
    analyzer = TaskAnalyzer(settings)

    def fake_generate(text=None, error=None, model_id="fake-model", tokens=(300, 100)):
        calls = []

        def generate(request):
            calls.append(request)
            if error:
                raise error
            return ModelResponse(text=text, model_id=model_id, input_tokens=tokens[0],
                                 output_tokens=tokens[1], cost_usd=0.0, latency_seconds=0.1, attempts=1)
        generate.calls = calls
        return generate

    # 1. Simple script: low complexity, no orchestration, language assumed
    a = analyzer.analyze("Write a simple python script that renames all files in a folder to lowercase.")
    assert a.task_type == "script" and a.complexity < 2 and not a.needs_orchestration
    assert a.primary_language == "python" and a.source == "heuristic" and a.analysis_tokens == 0
    print("PASS 1: simple script ->", a.short())

    # 2. The transcript's example project: desktop app, orchestrated
    a = analyzer.analyze("Build me a desktop app that monitors my downloads and automatically categorizes them.")
    assert a.task_type == "desktop_application" and a.needs_ui and a.needs_orchestration
    assert any("no language specified" in x for x in a.assumptions)
    print("PASS 2: desktop app ->", a.short())

    # 3. Heavy backend: very complex, db + network detected
    a = analyzer.analyze("Build a scalable REST API with JWT authentication, PostgreSQL, Redis caching, a "
                         "background job queue, role-based access control and Docker deployment")
    assert a.task_type == "backend_api" and a.complexity >= 8 and a.needs_database and a.needs_network
    assert a.token_plan["orchestrated_total"] > a.token_plan["direct_total"]
    print("PASS 3: heavy backend ->", a.short())

    # 4. Task types and languages
    assert analyzer.analyze("Solve this LeetCode problem using dynamic programming").task_type == "competitive_programming"
    cli = analyzer.analyze("Write a Rust CLI tool that greps logs")
    assert cli.task_type == "cli_tool" and cli.languages == ["rust"]
    assert analyzer.analyze("Port this C++ code to Python").languages == ["cpp", "python"]
    assert analyzer.analyze("Build a REST API in Go with a SQLite database").primary_language == "go"
    assert analyzer.analyze("make a thing").task_type == "general"
    print("PASS 4: task types and language detection (order = first mention)")

    # 5. LLM pass: fenced JSON with prose around it, blended complexity, token accounting
    llm_text = ('Sure! Here is the analysis:\n```json\n{"summary": "Organise downloads automatically", '
                '"task_type": "Desktop App", "complexity": 8, "languages": ["Python", "py"], '
                '"needs_ui": true, "needs_database": "yes", "needs_network": false, '
                '"key_features": ["watch folder", "categorise by type"], "risks": ["file locks"]}\n```')
    gen = fake_generate(llm_text)
    req = "Build me a desktop app that monitors my downloads and automatically categorizes them."
    a = analyzer.analyze(req, generate=gen)
    h = a.heuristic_complexity
    assert a.source == "llm+heuristic" and a.task_type == "desktop_application"
    assert a.llm_complexity == 8.0 and abs(a.complexity - round(0.7 * 8 + 0.3 * h, 1)) < 1e-9
    assert a.languages == ["python"] and a.needs_database is True and a.needs_network is False
    assert a.risks == ["file locks"] and a.analyzer_model == "fake-model" and a.analysis_tokens == 400
    request = gen.calls[0]
    assert request.purpose == "task_analysis" and request.temperature == 0.0 and req in request.prompt
    print("PASS 5: LLM analysis parsed and blended ->", a.short(), "| tokens", a.analysis_tokens)

    # 6. Bad model output falls back to heuristics (tokens still counted)
    for bad in ("I think it is hard.", "{not json", '{"complexity": "very high", "task_type": "wizardry"}', ""):
        a = analyzer.analyze(req, generate=fake_generate(bad))
        assert a.source == "heuristic" and a.task_type == "desktop_application"
        assert any("unusable" in x for x in a.assumptions)
    assert analyzer.analyze(req, generate=fake_generate("nope")).analysis_tokens == 400
    print("PASS 6: garbage replies fall back to heuristics")

    # 7. Model errors never break analysis
    for exc in (RuntimeError("boom"), ModelError("down"), KeyError("model")):
        a = analyzer.analyze(req, generate=fake_generate(error=exc))
        assert a.source == "heuristic" and a.analysis_tokens == 0
    print("PASS 7: exceptions from the model callable are absorbed")

    # 8. Out-of-range / odd values are sanitised
    odd = '{"task_type": "api", "complexity": 99, "languages": "golang", "needs_ui": "maybe"}'
    a = analyzer.analyze("Build a REST API", generate=fake_generate(odd))
    assert a.task_type == "backend_api" and a.llm_complexity == 10.0 and a.complexity <= 10.0
    assert a.languages == ["go"] and a.needs_ui is False       # "maybe" ignored -> heuristic value
    print("PASS 8: aliases normalised, complexity clamped, invalid booleans ignored")

    # 9. Forced orchestration overrides the threshold
    simple = "Write a simple python script that renames files"
    assert analyzer.analyze(simple, force_orchestration=True).needs_orchestration
    assert "forced on" in analyzer.analyze(simple, force_orchestration=True).orchestration_reason
    assert not analyzer.analyze(req, force_orchestration=False).needs_orchestration
    print("PASS 9: force_orchestration override")

    # 10. Token plan grows with complexity and respects settings
    low, high = estimate_token_plan(2, settings), estimate_token_plan(9, settings)
    assert high["coding"] > low["coding"] and high["hld"] > low["hld"] and high["estimated_files"] > low["estimated_files"]
    assert high["benchmark_per_candidate"] <= settings.lite_task_token_budget
    assert estimate_token_plan(5, settings, n_candidates=0)["benchmark_total"] == 0
    assert estimate_token_plan(5, OrchestratorSettings(max_review_iterations=0))["review"] == 0
    print("PASS 10: token plan", high)

    # 11. JSON extraction edge cases
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('text {"a": {"b": "}"}} more {"c": 2}') == {"a": {"b": "}"}}
    assert extract_json_object('<think>{x}</think> {"ok": true}') == {"ok": True}
    assert extract_json_object("[1, 2]") is None and extract_json_object(None) is None
    print("PASS 11: JSON extraction handles fences, nesting, braces in strings, think blocks")

    # 12. Save/load round trip, empty input
    a = analyzer.analyze(req, generate=fake_generate(llm_text))
    restored = TaskAnalysis.from_dict(json.loads(json.dumps(a.to_dict())))
    assert restored == a
    for empty in ("", "   \n"):
        try:
            analyzer.analyze(empty)
            raise SystemExit("FAIL 12")
        except ValueError:
            pass
    print("PASS 12: to_dict/from_dict round trip, empty requirement rejected")