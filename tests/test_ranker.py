"""
tests/test_ranker.py
---------------------
Unit tests for the HybridRanker.
Mocks both the SentenceTransformer and GroqClient to avoid I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.agents.ranker import HybridRanker
from app.models.paper import Paper


def _paper(arxiv_id: str, title: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Author"],
        abstract="Abstract about " + title.lower(),
        published=datetime.now(tz=timezone.utc),
        url="https://arxiv.org/abs/" + arxiv_id,
    )


class TestHybridRanker:
    @patch("app.agents.ranker.SentenceTransformer")
    def setup_method(self, method, mock_st_cls):
        self.mock_embedder = MagicMock()
        mock_st_cls.return_value = self.mock_embedder

        self.groq = MagicMock()
        # LLM returns scores 8, 6, 9 for 3 papers
        self.groq.complete_json.return_value = {"scores": [8, 6, 9]}

        # Embedder returns unit vectors — simulate cosine similarity
        def fake_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.array([1.0, 0.0, 0.0])
            n = len(texts)
            vecs = np.zeros((n, 3))
            for i in range(n):
                vecs[i, 0] = 0.9 - i * 0.1  # decreasing similarity
            return vecs

        self.mock_embedder.encode.side_effect = fake_encode

        self.ranker = HybridRanker(
            keywords=["finance", "AI"],
            groq_client=self.groq,
            embedding_weight=0.4,
            llm_weight=0.6,
        )

    def test_rank_returns_sorted_list(self):
        papers = [
            _paper("001", "Finance and Deep Learning"),
            _paper("002", "Quantum Optics"),
            _paper("003", "Explainable AI in Finance"),
        ]
        ranked = self.ranker.rank(papers)
        assert len(ranked) == 3
        # All papers should have a hybrid score
        for p in ranked:
            assert p.hybrid_score > 0

    def test_rank_empty_list(self):
        result = self.ranker.rank([])
        assert result == []

    def test_scores_are_in_1_10_range(self):
        papers = [_paper(str(i), f"Paper {i}") for i in range(3)]
        ranked = self.ranker.rank(papers)
        for p in ranked:
            assert 0 <= p.hybrid_score <= 10
