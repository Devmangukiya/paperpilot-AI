"""
tests/test_filter.py
---------------------
Unit tests for the PaperFilter service.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models.paper import Paper
from app.services.filter import PaperFilter


def _paper(title: str, abstract: str, days_old: int = 10) -> Paper:
    return Paper(
        arxiv_id=f"test-{hash(title) % 9999}",
        title=title,
        authors=["Test Author"],
        abstract=abstract,
        published=datetime.now(tz=timezone.utc) - timedelta(days=days_old),
        url="https://arxiv.org/abs/0000.00000",
    )


class TestPaperFilter:
    def setup_method(self):
        self.keywords = ["finance", "federated learning", "explainable AI"]
        self.f = PaperFilter(keywords=self.keywords, lookback_days=365, min_keyword_matches=1)

    def test_keeps_recent_relevant_paper(self):
        p = _paper("Finance and FL", "federated learning for financial forecasting")
        result = self.f.filter([p])
        assert len(result) == 1

    def test_removes_old_paper(self):
        p = _paper("Finance Paper", "finance and AI", days_old=400)
        result = self.f.filter([p])
        assert len(result) == 0

    def test_removes_irrelevant_paper(self):
        p = _paper("Quantum Optics", "laser diffraction patterns in vacuum")
        result = self.f.filter([p])
        assert len(result) == 0

    def test_abbreviation_matching(self):
        # "XAI" should match "explainable AI"
        p = _paper("XAI in Banking", "we use XAI to explain model outputs in banking")
        result = self.f.filter([p])
        assert len(result) == 1

    def test_multiple_papers(self):
        papers = [
            _paper("Finance AI", "deep learning for financial forecasting"),
            _paper("Quantum Optics", "laser experiments with no AI"),
            _paper("FL Healthcare", "federated learning in hospitals"),
        ]
        result = self.f.filter(papers)
        assert len(result) == 2
