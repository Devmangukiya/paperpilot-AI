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
        Fetch papers relevant to *keywords* published in the last
        ``lookback_days`` days.

        Parameters
        ----------
        keywords : list[str]
            E.g. ["finance", "AI", "federated learning", "XAI"]

        Returns
        -------
        list[Paper]
            Unsorted list of Paper objects.
        """
        query = self._build_query(keywords)
        logger.info("arXiv query: %s", query)

        cutoff = datetime.now(tz=timezone.utc) - timedelta(
            days=self.lookback_days
        )

        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers: list[Paper] = []
        try:
            for result in self._client.results(search):
                # Hard date filter
                if result.published and result.published < cutoff:
                    continue

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

        logger.info("Retrieved %d papers from arXiv.", len(papers))
        return papers

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_query(keywords: list[str]) -> str:
        """
        Build an arXiv query string.

        Strategy
        --------
        Each keyword (phrase) is wrapped in quotes and searched in the
        title+abstract fields.  Multiple keywords are joined with AND so
        that returned papers must touch every topic.

        Example
        -------
        keywords = ["finance", "federated learning", "XAI"]
        → ti_abs:"finance" AND ti_abs:"federated learning" AND ti_abs:"XAI"
        """
        parts = []
        for kw in keywords:
            # arXiv field prefix for title+abstract search
            escaped = kw.strip().replace('"', "")
            parts.append(f'ti_abs:"{escaped}"')

        # Join with AND for strict multi-topic matching
        # Fall back to OR-chain if a strict AND yields 0 results
        # (the pipeline handles 0-result fallback separately)
        return " AND ".join(parts)
