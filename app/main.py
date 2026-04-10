"""
main.py
-------
CLI entry point for the Daily Research Paper Agent.

Usage examples
--------------
# Run once with default keywords:
    python -m app.main

# Run once with custom keywords:
    python -m app.main --keywords "finance, AI, federated learning, XAI"

# Start the daily scheduler (runs at DAILY_RUN_TIME then once per day):
    python -m app.main --schedule

# Run once and output JSON to stdout:
    python -m app.main --json
"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import config
from app.pipelines.daily_pipeline import DailyResearchPipeline
from app.pipelines.scheduler import start_scheduler
from app.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Daily Research Paper Agent — fetch, rank, and summarise research papers.",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help=(
            'Comma-separated list of keywords.  '
            'Example: --keywords "finance, AI, federated learning, XAI"'
        ),
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Start the daily scheduler instead of running once.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print final result as JSON to stdout (suppresses pretty-print).",
    )
    return parser.parse_args()


def resolve_keywords(raw: str | None) -> list[str]:
    if raw:
        return [kw.strip() for kw in raw.split(",") if kw.strip()]
    return config.DEFAULT_KEYWORDS


def main() -> None:
    args = parse_args()
    keywords = resolve_keywords(args.keywords)

    logger.info("Starting Daily Research Paper Agent")
    logger.info("Keywords: %s", keywords)

    if args.schedule:
        # Blocking scheduler loop
        start_scheduler(keywords=keywords)
        return

    # Single run
    pipeline = DailyResearchPipeline(keywords=keywords)
    result = pipeline.run()

    if result is None:
        print("[ERROR] No paper could be found for the given keywords.", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        # Machine-readable JSON to stdout
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
