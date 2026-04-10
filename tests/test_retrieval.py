"""
tests/test_retrieval.py
-----------------------
Unit tests for the arXiv retrieval service.
Uses pytest; does NOT make live API calls (mocked with unittest.mock).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.services.retrieval import ArxivRetriever


# ── Helpers ────────────────────────────────────────────────────────────

def _make_mock_result(title: str, days_old: int = 30) -> MagicMock:
    """Build a minimal mock arxiv.Result object."""
    r = MagicMock()
    r.entry_id = f"https://arxiv.org/abs/2401.{hash(title) % 99999:05d}"
    r.title = title
    r.authors = [MagicMock(__str__=lambda s: "Author A")]
    r.summary = "This is a test abstract about " + title.lower()
    r.published = datetime.now(tz=timezone.utc).replace(
        day=max(1, datetime.now().day - days_old % 28)
    )
    r.categories = ["cs.AI"]
    return r


# ── Tests ──────────────────────────────────────────────────────────────

class TestArxivRetriever:
    def test_build_query_single_keyword(self):
        retriever = ArxivRetriever()
        query = retriever._build_query(["finance"])
        assert 'ti_abs:"finance"' in query

    def test_build_query_multiple_keywords(self):
        retriever = ArxivRetriever()
        query = retriever._build_query(["finance", "federated learning"])
        assert "AND" in query
        assert 'ti_abs:"finance"' in query
        assert 'ti_abs:"federated learning"' in query

    @patch("app.services.retrieval.arxiv.Client")
    def test_fetch_returns_papers(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.results.return_value = [
            _make_mock_result("Finance AI Paper"),
            _make_mock_result("Federated XAI Study"),
        ]

        retriever = ArxivRetriever(max_results=5)
        papers = retriever.fetch(["finance", "AI"])

        assert len(papers) == 2
        assert papers[0].title == "Finance AI Paper"

    @patch("app.services.retrieval.arxiv.Client")
    def test_fetch_handles_api_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.results.side_effect = Exception("Network error")

        retriever = ArxivRetriever()
        papers = retriever.fetch(["finance"])
        # Should return empty list, not raise
        assert papers == []
