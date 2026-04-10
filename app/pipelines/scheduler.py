"""
pipelines/scheduler.py
-----------------------
Wraps the `schedule` library to run the pipeline once per day at the
configured time.  Also provides a cron-compatible "run-once" mode.
"""

from __future__ import annotations

import time

import schedule

from app.config import config
from app.pipelines.daily_pipeline import DailyResearchPipeline
from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(keywords: list[str] | None = None) -> None:
    """Wrapper executed by the scheduler."""
    pipeline = DailyResearchPipeline(keywords=keywords)
    pipeline.run()


def start_scheduler(keywords: list[str] | None = None) -> None:
    """
    Start the blocking daily scheduler.

    Schedules `run_pipeline` at the time specified by DAILY_RUN_TIME
    (default 08:00 local time) and then enters the event loop.

    Press Ctrl+C to stop.
    """
    run_time = config.DAILY_RUN_TIME
    logger.info("Scheduler started — will run daily at %s.", run_time)
    print(f"[Scheduler] Daily job set for {run_time}.  Press Ctrl+C to stop.")

    schedule.every().day.at(run_time).do(run_pipeline, keywords=keywords)

    # Also run immediately on first start so the user sees output right away
    logger.info("Running pipeline immediately on startup …")
    run_pipeline(keywords=keywords)

    while True:
        schedule.run_pending()
        time.sleep(30)  # check every 30 s
