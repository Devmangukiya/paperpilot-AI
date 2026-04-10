"""
services/filter.py
------------------
Filters a raw list of Paper objects by:
  1. Publication date  (within lookback window)
  2. Abstract keyword relevance  (at least K keywords must appear)
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from app.config import config
from app.models.paper import Paper
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PaperFilter:
    """
    Lightweight filter that runs *before* the expensive ranking step.

    Parameters
    ----------
    keywords : list[str]
        User-supplied keywords.  At least ``min_keyword_matches`` of them
        must appear (case-insensitive) in the abstract or title.
    lookback_days : int
        Papers older than this are discarded.
    min_keyword_matches : int
        Minimum number of keywords that must be present in title+abstract.
    """

    def __init__(
        self,
        keywords: list[str],
        lookback_days: int = config.DATE_LOOKBACK_DAYS,
        min_keyword_matches: int = 1,
    ):
        self.keywords = [kw.lower() for kw in keywords]
        self.lookback_days = lookback_days
        self.min_keyword_matches = min_keyword_matches
        self._cutoff = datetime.now(tz=timezone.utc) - timedelta(
            days=lookback_days
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def filter(self, papers: list[Paper]) -> list[Paper]:
        """
        Return papers that pass both date and keyword checks.
        """
        kept: list[Paper] = []
        for paper in papers:
            if not self._date_ok(paper):
                continue
            if not self._keyword_ok(paper):
                continue
            kept.append(paper)

        logger.info(
            "Filter: %d → %d papers after date + keyword checks.",
            len(papers),
            len(kept),
        )
        return kept

    # ------------------------------------------------------------------ #
    #  Internal checks                                                     #
    # ------------------------------------------------------------------ #

    def _date_ok(self, paper: Paper) -> bool:
        """True if paper was published within the lookback window."""
        if paper.published is None:
            return True  # unknown date → keep
        pub = paper.published
        # Normalise to UTC-aware datetime
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return pub >= self._cutoff

    def _keyword_ok(self, paper: Paper) -> bool:
        """True if at least min_keyword_matches keywords appear in title+abstract."""
        haystack = (paper.title + " " + paper.abstract).lower()
        matches = sum(
            1 for kw in self.keywords if self._kw_in_text(kw, haystack)
        )
        return matches >= self.min_keyword_matches

    @staticmethod
    def _kw_in_text(keyword: str, text: str) -> bool:
        """
        Flexible keyword match: checks for:
          - exact phrase  ("federated learning")
          - common abbreviations  ("fl" for federated learning, "xai", etc.)
        """
        # Direct substring
        if keyword in text:
            return True

        # Common abbreviation map
        abbreviations: dict[str, list[str]] = {
            "federated learning": ["fl", "federated"],
            "explainable ai": ["xai", "explainability", "interpretab"],
            "artificial intelligence": ["ai", "machine learning", "deep learning", "ml"],
            "finance": ["financial", "fintech", "stock", "portfolio", "trading"],
        }
        aliases = abbreviations.get(keyword, [])
        return any(alias in text for alias in aliases)
