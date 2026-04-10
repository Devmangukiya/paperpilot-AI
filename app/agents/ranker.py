"""
agents/ranker.py
----------------
Hybrid ranking engine that scores papers using:

  1. Embedding similarity  – cosine similarity between the query vector and
     each paper's title+abstract, computed with sentence-transformers
     (runs entirely locally, no API call needed).

  2. LLM relevance scoring – Groq model rates each paper on a 1-10 scale
     given the user's keywords.

Final hybrid score = EMBEDDING_WEIGHT × embedding_sim
                   + LLM_WEIGHT      × llm_score_normalised
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import config
from app.models.paper import Paper
from app.services.groq_client import GroqClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRanker:
    """
    Rank a list of Paper objects and return them sorted best-first.

    Parameters
    ----------
    keywords : list[str]
        User topic keywords.  Used to build the query text for embeddings
        and the LLM scoring prompt.
    groq_client : GroqClient
        Shared Groq API client instance.
    embedding_weight : float
        Weight for embedding similarity component (0-1).
    llm_weight : float
        Weight for LLM score component (0-1).
    """

    def __init__(
        self,
        keywords: list[str],
        groq_client: GroqClient,
        embedding_weight: float = config.EMBEDDING_WEIGHT,
        llm_weight: float = config.LLM_WEIGHT,
        embedding_model_name: str = config.EMBEDDING_MODEL,
    ):
        self.keywords = keywords
        self.query_text = " ".join(keywords)
        self.groq = groq_client
        self.emb_w = embedding_weight
        self.llm_w = llm_weight

        logger.info("Loading embedding model: %s", embedding_model_name)
        self._embedder = SentenceTransformer(embedding_model_name)
        logger.info("Embedding model loaded.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def rank(self, papers: list[Paper]) -> list[Paper]:
        """
        Score and rank papers.  Returns the same list sorted by
        `hybrid_score` descending.
        """
        if not papers:
            return []

        logger.info("Ranking %d papers …", len(papers))

        self._compute_embedding_scores(papers)
        self._compute_llm_scores(papers)
        self._compute_hybrid_scores(papers)

        ranked = sorted(papers, key=lambda p: p.hybrid_score, reverse=True)
        logger.info(
            "Top paper after ranking: '%s' (hybrid_score=%.3f)",
            ranked[0].title[:60],
            ranked[0].hybrid_score,
        )
        return ranked

    # ------------------------------------------------------------------ #
    #  Step 1: Embedding similarity                                        #
    # ------------------------------------------------------------------ #

    def _compute_embedding_scores(self, papers: list[Paper]) -> None:
        """
        Compute cosine similarity between the query embedding and each
        paper's title+abstract embedding.

        Scores are stored in paper.embedding_score (0-1 range).
        """
        texts = [f"{p.title}. {p.abstract[:512]}" for p in papers]
        query_vec = self._embedder.encode(
            self.query_text, convert_to_numpy=True, normalize_embeddings=True
        )
        paper_vecs = self._embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=16
        )

        # Cosine similarity (embeddings are L2-normalised → dot product = cosine)
        sims: np.ndarray = paper_vecs @ query_vec  # shape: (N,)

        # Clip to [0, 1] – cosine can be slightly negative for unrelated texts
        sims = np.clip(sims, 0.0, 1.0)

        for paper, sim in zip(papers, sims):
            paper.embedding_score = float(sim)

        logger.debug(
            "Embedding scores — min: %.3f  max: %.3f  mean: %.3f",
            sims.min(), sims.max(), sims.mean(),
        )

    # ------------------------------------------------------------------ #
    #  Step 2: LLM relevance scoring                                       #
    # ------------------------------------------------------------------ #

    def _compute_llm_scores(self, papers: list[Paper]) -> None:
        """
        Ask Groq to rate each paper on a 1-10 scale.

        To reduce API calls we score all papers in a single prompt
        (batch of up to 10 papers).  For larger lists we split into batches.
        """
        batch_size = 8
        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            self._score_batch(batch)

    def _score_batch(self, papers: list[Paper]) -> None:
        """Score a small batch of papers with one Groq API call."""
        topics = ", ".join(self.keywords)

        # Build a numbered list of papers
        paper_list = "\n\n".join(
            f"[{idx + 1}] Title: {p.title}\nAbstract: {p.abstract[:400]}"
            for idx, p in enumerate(papers)
        )

        prompt = (
            f"You are a research relevance judge.  "
            f"The user cares about: {topics}.\n\n"
            f"Rate each paper below on a scale of 1 to 10 based on how relevant "
            f"it is to those topics (10 = extremely relevant, 1 = not at all).\n\n"
            f"{paper_list}\n\n"
            f"Respond ONLY with a JSON object like:\n"
            f'{{ "scores": [7, 9, 4, ...] }}\n'
            f"No explanation, no extra text."
        )

        try:
            result = self.groq.complete_json(user_prompt=prompt, max_tokens=256)
            raw_scores = result.get("scores", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM batch scoring failed: %s. Defaulting to 5.", exc)
            raw_scores = [5] * len(papers)

        # Pad / truncate to match batch size
        raw_scores = list(raw_scores) + [5] * len(papers)
        raw_scores = raw_scores[: len(papers)]

        for paper, score in zip(papers, raw_scores):
            try:
                paper.llm_score = max(1.0, min(10.0, float(score))) / 10.0
            except (ValueError, TypeError):
                paper.llm_score = 0.5  # safe default

    # ------------------------------------------------------------------ #
    #  Step 3: Hybrid combination                                          #
    # ------------------------------------------------------------------ #

    def _compute_hybrid_scores(self, papers: list[Paper]) -> None:
        """
        Compute the weighted hybrid score and scale to 1-10 for readability.
        """
        for paper in papers:
            raw = (
                self.emb_w * paper.embedding_score
                + self.llm_w * paper.llm_score
            )
            # Scale to 1-10
            paper.hybrid_score = raw * 10.0
