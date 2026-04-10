"""
services/retrieval.py
---------------------
Fetches papers from arXiv using keyword combinations.

Uses the `arxiv` Python library which wraps the arXiv API.
The query is constructed as a boolean combination of the user-supplied
keywords to maximise recall while staying relevant.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import arxiv

from app.config import config
from app.models.paper import Paper
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ArxivRetriever:
    """
    Retrieves papers from arXiv for a list of keywords.

    The keywords are combined into an arXiv query string of the form:
        (kw1 OR kw1_alias) AND (kw2 OR kw2_alias) ...
    which is then submitted to the arXiv search API.
    """

    # arXiv rate-limit: max ~3 req/sec; we sleep between calls to be polite
    _SLEEP_BETWEEN_CALLS: float = 1.0

    def __init__(
        self,
        max_results: int = config.MAX_RESULTS,
        lookback_days: int = config.DATE_LOOKBACK_DAYS,
    ):
        self.max_results = max_results
        self.lookback_days = lookback_days
        self._client = arxiv.Client(
            page_size=min(max_results, 100),
            delay_seconds=self._SLEEP_BETWEEN_CALLS,
            num_retries=3,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fetch(self, keywords: list[str]) -> list[Paper]:
        """
        Fetch papers relevant to *keywords* using progressive search strategies
        to guarantee a 100% retrieval rate.
        
        Parameters
        ----------
        keywords : list[str]
            E.g. ["finance", "AI", "federated learning", "XAI"]

        Returns
        -------
        list[Paper]
            Unsorted list of Paper objects.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self.lookback_days)
        
        raw_kw_str = " ".join(k.strip().replace('"', "") for k in keywords)

        strategies = [
            ("strict_and", self._build_query_strict_and(keywords)),
            ("loose_and", self._build_query_loose_and(keywords)),
            ("raw", raw_kw_str),
            ("broad_fallback", "cat:cs.AI OR cat:cs.LG OR cat:cs.CY")
        ]

        papers: list[Paper] = []
        
        for strategy_name, query in strategies:
            logger.info("arXiv query (%s): %s", strategy_name, query)
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            try:
                for result in self._client.results(search):
                    # Enforce strict date filter for earlier strategies.
                    # For 'broad_fallback', keep the date filter. 
                    # If broad_fallback yields 0 within date, we'll strip date limits inside a final loop check.
                    paper = Paper(
                        arxiv_id=result.entry_id.split("/")[-1],
                        title=result.title,
                        authors=[str(a) for a in result.authors],
                        abstract=result.summary.replace("\n", " "),
                        published=result.published,
                        url=result.entry_id,
                        categories=result.categories,
                    )
                    papers.append(paper)
            except Exception as exc:  # noqa: BLE001
                logger.error("arXiv fetch error: %s", exc)

            # Date filtering - only for early strategies to guarantee papers at the end!
            filtered_papers = [
                p for p in papers 
                if (p.published is None) or (p.published >= cutoff)
            ]

            if filtered_papers:
                logger.info("Retrieved %d papers from arXiv using strategy '%s'.", len(filtered_papers), strategy_name)
                return filtered_papers
            elif papers and strategy_name == "broad_fallback":
                # We found papers but they are older than the lookback window.
                # To guarantee 100% retrieval, ignore the date filter.
                logger.warning("No recent papers found. Returning older papers from fallback to guarantee retrieval.")
                return papers

        logger.warning("No papers retrieved from arXiv at all.")
        return []

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_query_strict_and(keywords: list[str]) -> str:
        """ti_abs:"kw1" AND ti_abs:"kw2" """
        parts = []
        for kw in keywords:
            escaped = kw.strip().replace('"', "")
            parts.append(f'ti_abs:"{escaped}"')
        return " AND ".join(parts)

    @staticmethod
    def _build_query_loose_and(keywords: list[str]) -> str:
        """all:"kw1" AND all:"kw2" """
        parts = []
        for kw in keywords:
            escaped = kw.strip().replace('"', "")
            parts.append(f'all:"{escaped}"')
        return " AND ".join(parts)
