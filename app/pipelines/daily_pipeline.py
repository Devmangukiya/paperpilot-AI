"""
pipelines/daily_pipeline.py
----------------------------
Orchestrates the full end-to-end pipeline:

  1. Retrieve papers from arXiv.
  2. Filter by date + keyword presence.
  3. Remove already-seen papers (cache check).
  4. Rank with hybrid engine.
  5. Summarise the top paper with the LLM.
  6. Save result to JSON + print to console.
  7. Mark paper as seen in cache.

This module is the single entry point called by the scheduler and the CLI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from app.agents.ranker import HybridRanker
from app.agents.summariser import PaperSummariser
from app.config import config
from app.models.paper import AgentResult
from app.services.filter import PaperFilter
from app.services.groq_client import GroqClient
from app.services.retrieval import OpenAlexRetriever
from app.utils.cache import PaperCache
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DailyResearchPipeline:
    """
    Full research-paper agent pipeline.

    Parameters
    ----------
    keywords : list[str]
        Topics to search and rank by.
    """

    def __init__(self, keywords: list[str] | None = None):
        self.keywords = keywords or config.DEFAULT_KEYWORDS
        logger.info("Pipeline initialised with keywords: %s", self.keywords)

        # Shared services
        self._groq = GroqClient()
        self._cache = PaperCache()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> AgentResult | None:
        """
        Execute the full pipeline.

        Returns
        -------
        AgentResult or None
            Structured result dict, or None if no suitable paper was found.
        """
        logger.info("=" * 60)
        logger.info("Daily Research Paper Agent — starting run")
        logger.info("Keywords: %s", self.keywords)
        logger.info("=" * 60)

        # ── Step 1: Retrieve ──────────────────────────────────────────
        retriever = OpenAlexRetriever()
        papers = retriever.fetch(self.keywords)

        if not papers:
            logger.error("No papers found at all. Aborting.")
            return None

        # ── Step 2: Filter ────────────────────────────────────────────
        paper_filter = PaperFilter(keywords=self.keywords, min_keyword_matches=1)
        filtered_papers = paper_filter.filter(papers)

        if not filtered_papers:
            logger.warning("All papers filtered out. Bypassing filter to guarantee 100% retrieval rate.")
            filtered_papers = papers

        papers = filtered_papers

        # ── Step 3: Cache deduplication ──────────────────────────────
        unseen_ids = self._cache.unseen_ids([p.paper_id for p in papers])
        unseen_papers = [p for p in papers if p.paper_id in unseen_ids]

        if not unseen_papers:
            logger.info(
                "All %d retrieved papers are already in cache. "
                "Falling back to best-ever paper from this batch.",
                len(papers),
            )
            # If everything is cached, still run ranking so the user gets output
            unseen_papers = papers

        logger.info("%d unseen papers after cache check.", len(unseen_papers))

        # ── Step 4: Rank ─────────────────────────────────────────────
        ranker = HybridRanker(keywords=self.keywords, groq_client=self._groq)
        ranked = ranker.rank(unseen_papers)
        top_paper = ranked[0]

        # ── Step 5: Summarise ─────────────────────────────────────────
        summariser = PaperSummariser(groq_client=self._groq, keywords=self.keywords)
        result = summariser.summarise(top_paper)

        # ── Step 6: Save & Display ───────────────────────────────────
        self._save_result(result)
        self._display_result(result)

        # ── Step 7: Mark as seen ──────────────────────────────────────
        self._cache.mark_seen(top_paper.paper_id)

        logger.info("Pipeline run complete.")
        return result

    # ------------------------------------------------------------------ #
    #  Output helpers                                                      #
    # ------------------------------------------------------------------ #

    def _save_result(self, result: AgentResult) -> None:
        """Persist the result to a dated JSON file in data/output/."""
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        fname = os.path.join(config.OUTPUT_DIR, f"paper_{date_str}.json")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info("Result saved to: %s", fname)
        except OSError as exc:
            logger.error("Failed to save result: %s", exc)

    @staticmethod
    def _display_result(result: AgentResult) -> None:
        """Pretty-print the result to stdout."""
        border = "═" * 70
        print(f"\n{border}")
        print("  📄  DAILY RESEARCH PAPER AGENT")
        print(border)
        print(f"\n🏆  TITLE : {result.title}")
        print(f"👥  AUTHORS: {result.authors}")
        print(f"📅  DATE  : {result.published}")
        print(f"🔗  LINK  : {result.link}")
        print(f"⭐  SCORE : {result.score} / 10\n")
        print("─" * 70)
        print("📝  SUMMARY")
        print(f"   {result.summary}")
        print("\n🎯  WHY RELEVANT")
        print(f"   {result.why_relevant}")
        print("\n💡  KEY INSIGHTS")
        for insight in result.key_insights:
            print(f"   • {insight}")
        print(f"\n{border}\n")
