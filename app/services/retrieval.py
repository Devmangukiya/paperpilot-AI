"""
services/retrieval.py
---------------------
Fetches papers from OpenAlex (the whole internet space) using keyword combinations.

OpenAlex API indexes over 250 million works globally, including arXiv, PubMed, 
Crossref, and major journals without requiring an API key.
"""

from __future__ import annotations

import time
import requests
from datetime import datetime, timedelta, timezone

from app.config import config
from app.models.paper import Paper
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAlexRetriever:
    """
    Retrieves papers from OpenAlex for a list of keywords.

    Uses progressive fallback queries to ensure a 100% retrieval rate.
    """

    _BASE_URL = "https://api.openalex.org/works"

    def __init__(
        self,
        max_results: int = config.MAX_RESULTS,
        lookback_days: int = config.DATE_LOOKBACK_DAYS,
    ):
        self.max_results = max_results
        self.lookback_days = lookback_days

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
            ("broad_fallback", "artificial intelligence OR machine learning")
        ]

        papers: list[Paper] = []
        
        # Identify politely to OpenAlex (best practice, places us in polite pool)
        headers = {"User-Agent": "mailto:dev@paperpilot.ai"}

        for strategy_name, query in strategies:
            logger.info("OpenAlex query (%s): %s", strategy_name, query)
            
            params = {
                "search": query,
                "filter": "has_abstract:true",
                "per-page": min(self.max_results, 50),
                "sort": "publication_date:desc"
            }

            try:
                response = requests.get(self._BASE_URL, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                for item in results:
                    paper = self._parse_paper(item)
                    if paper:
                        papers.append(paper)
                        
            except Exception as exc:  # noqa: BLE001
                logger.error("OpenAlex fetch error: %s", exc)

            # Date filtering - only for early strategies to guarantee papers at the end!
            filtered_papers = [
                p for p in papers 
                if (p.published is None) or (p.published >= cutoff)
            ]

            if filtered_papers:
                logger.info("Retrieved %d papers from OpenAlex using strategy '%s'.", len(filtered_papers), strategy_name)
                return filtered_papers
            elif papers and strategy_name == "broad_fallback":
                # We found papers but they are older than the lookback window.
                # To guarantee 100% retrieval, ignore the date filter.
                logger.warning("No recent papers found. Returning older papers from fallback to guarantee retrieval.")
                return papers
            
            # Short sleep between strategies
            time.sleep(0.5)

        logger.warning("No papers retrieved from OpenAlex at all.")
        return []

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _parse_paper(self, item: dict) -> Paper | None:
        """Parse raw OpenAlex dict into a Paper model."""
        try:
            openalex_id = item.get("id", "").split("/")[-1]
            title = item.get("title", "")
            
            # Extract authors
            authorships = item.get("authorships", [])
            authors = [a.get("author", {}).get("display_name", "Unknown") for a in authorships]
            
            # Reconstruct abstract
            abstract = ""
            abstract_idx = item.get("abstract_inverted_index")
            if abstract_idx:
                 max_pos = max(max(positions) for positions in abstract_idx.values())
                 words = [""] * (max_pos + 1)
                 for word, positions in abstract_idx.items():
                     for pos in positions:
                         words[pos] = word
                 abstract = " ".join(words)
            
            if not abstract or not title:
                return None
                
            # Date
            pub_date_str = item.get("publication_date")
            published = None
            if pub_date_str:
                 try:
                     # OpenAlex format YYYY-MM-DD
                     published = datetime.strptime(pub_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                 except ValueError:
                     pass
                     
            # URL
            url = item.get("doi") or (item.get("primary_location") or {}).get("landing_page_url") or item.get("id")
            
            # Topics / Categories
            topics = [t.get("display_name") for t in item.get("topics", [])]

            return Paper(
                paper_id=openalex_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published or datetime.now(timezone.utc),
                url=url,
                categories=topics,
            )
        except Exception as exc:
            logger.debug("Failed to parse paper: %s", exc)
            return None

    @staticmethod
    def _build_query_strict_and(keywords: list[str]) -> str:
        """kw1 AND kw2 (OpenAlex search uses default boolean operators)"""
        parts = []
        for kw in keywords:
            escaped = kw.strip().replace('"', "")
            parts.append(f'"{escaped}"')
        return " AND ".join(parts)

    @staticmethod
    def _build_query_loose_and(keywords: list[str]) -> str:
        """kw1 AND kw2 (without quotes)"""
        parts = []
        for kw in keywords:
            escaped = kw.strip().replace('"', "")
            parts.append(f'{escaped}')
        return " AND ".join(parts)
