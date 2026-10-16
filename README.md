# Multi-LLM Coding Orchestrator

Give it a plain-English software requirement. It decides whether one model is enough or whether to
orchestrate several (Claude, GPT, Gemini, a local Ollama model):

1. **Analyze** the task (type, complexity, size) and decide direct vs. orchestrated.
2. **Lite benchmark** (orchestrated only): each candidate builds a reduced version of the project; results are scored
   objectively (compiles, tests, static analysis, run results, runtime/memory). Past results can skip the benchmark.
3. **Select** the best coder (waiting for a quota reset or falling back if needed) and an independent reviewer.
4. **Architecture (HLD)**, then **coding** in batches.
5. **Review loop**: evaluate, review, revise (max 3 rounds), roll back revisions that make things worse,
   escalate once if a blocking issue persists. An optional LLM judge only adjusts soft metrics.

It tracks token quotas and cooldowns, reserves tokens for later stages, saves state after every stage
(resume after a crash or pause) and logs every decision (`data/logs/decisions.jsonl`).

## Setup

```bash
pip install -r requirements.txt        # only the SDKs you need
cp .env.example .env                   # add your API keys
python -m config.settings              # shows which models are usable
```

Local model: install [Ollama](https://ollama.com), then `ollama pull qwen2.5-coder` and `ollama serve`.
Model names, budgets and capability scores are in `config/settings.py`; override them in `config/config.json`.
**The default model names are placeholders: set the ones you really have access to.**

## Usage

```bash
python main.py "Build a desktop app that monitors my downloads and categorizes them"
python main.py --file requirement.txt
python main.py --direct "script that renames files to lowercase"     # one model only
python main.py --orchestrate "..."                                    # always benchmark + review
python main.py --list                                                 # past projects
python main.py --resume <project_id>                                  # continue paused/failed run
python main.py --models                                               # usable models
```

Output goes to `projects/<id>/implementation/`; `final_report.md`, `architecture.md`, `reviews/` and
`benchmark/` sit next to it. Exit codes: 0 completed, 1 needs attention/failed, 2 paused, 3 setup error.

## Tests

Each module has a self-test: `PYTHONPATH=. python -m core.orchestrator` (also `core.scheduler`, `agents.coder`, ...).

## TO-DO:

- UI
- Codebase cleanup

## Notes

- Generated programs are only run if they are plain terminating Python scripts; GUIs, servers and programs needing
  input/arguments are never run automatically. Candidate code runs in subprocesses with a scrubbed environment.
- Status: tested with scripted fake models against the real components; provider adapters have not been run against
  the live APIs yet. Expect to tune prompts and model names on first real use.
