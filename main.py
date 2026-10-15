"""
main.py - command line entry point of the multi-LLM coding orchestrator.

    python main.py "Build a desktop app that monitors my downloads and categorizes them"
    python main.py --file requirement.txt
    python main.py --direct "Write a script that renames files to lowercase"     # force a single model
    python main.py --orchestrate "..."                                            # force benchmark + review
    python main.py --list                                                         # past projects
    python main.py --resume <project_id>                                          # continue a paused/failed run
    python main.py --models                                                       # which models are usable
    python main.py --config my_config.json ...

Exit code: 0 completed, 1 needs attention / failed, 2 paused (resume later), 3 usage or setup error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXIT = {"completed": 0, "needs_attention": 1, "failed": 1, "paused": 2}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Multi-LLM coding orchestrator")
    p.add_argument("requirement", nargs="*", help="what to build (plain English)")
    p.add_argument("-f", "--file", help="read the requirement from a text file")
    p.add_argument("--resume", metavar="PROJECT_ID", help="continue a paused, failed or interrupted project")
    p.add_argument("--list", action="store_true", help="list existing projects")
    p.add_argument("--models", action="store_true", help="show which models are usable and why others are not")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--direct", action="store_true", help="use one model, no benchmark and no second reviewer")
    mode.add_argument("--orchestrate", action="store_true", help="always benchmark and use an independent reviewer")
    p.add_argument("--project-id", help="name for the new project folder")
    p.add_argument("--no-wait", action="store_true", help="pause instead of sleeping until a quota resets")
    p.add_argument("--no-judge", action="store_true", help="skip the optional final LLM judge")
    p.add_argument("--config", type=Path, help="path to a config.json (default: config/config.json)")
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("-q", "--quiet", action="store_true", help="do not print progress lines")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from config.settings import load_settings
    from core.logger import setup_logging
    from core.orchestrator import build_orchestrator

    setup_logging(args.log_level)
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        print(f"Could not load the settings: {exc}", file=sys.stderr)
        return 3

    def progress(project_id: str, message: str) -> None:
        if not args.quiet:
            print(f"[{project_id}] {message}", flush=True)

    try:
        orch = build_orchestrator(settings, wait_for_quota=not args.no_wait, use_judge=not args.no_judge,
                                  on_event=progress)
    except Exception as exc:
        print(f"Could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    if args.models:
        usable = orch.registry.ids()
        print("Usable models:", ", ".join(usable) or "none")
        for model_id, reason in sorted(orch.registry.skipped.items()):
            print(f"  skipped {model_id}: {reason}")
        return 0 if usable else 3

    if args.list:
        projects = orch.state.list_projects()
        for item in projects:
            print(f"{item.get('project_id', '?'):40} {item.get('status', '?'):12} {str(item.get('requirement', ''))[:60]}")
        if not projects:
            print("No projects yet.")
        return 0

    if not orch.registry.ids():
        print("No model is usable. Add API keys to .env (see .env.example) or start a local model.", file=sys.stderr)
        for model_id, reason in orch.registry.skipped.items():
            print(f"  {model_id}: {reason}", file=sys.stderr)
        return 3

    try:
        if args.resume:
            result = orch.resume(args.resume)
        else:
            text = Path(args.file).read_text(encoding="utf-8") if args.file else " ".join(args.requirement)
            if not text.strip():
                print("Give a requirement, --file, --resume, --list or --models. See --help.", file=sys.stderr)
                return 3
            force = True if args.orchestrate else (False if args.direct else None)
            result = orch.run(text, project_id=args.project_id, force_orchestration=force)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; continue with --resume <project_id>.", file=sys.stderr)
        return 2
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3

    print()
    print(result.summary())
    return EXIT.get(result.status, 1)


if __name__ == "__main__":
    sys.exit(main())