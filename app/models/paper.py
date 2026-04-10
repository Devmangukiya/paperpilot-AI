"""
models/paper.py
---------------
Pydantic data-models used throughout the pipeline.
Using dataclasses here to keep the dependency footprint minimal;
swap to pydantic BaseModel if you add FastAPI later.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Paper:
    """Raw paper as returned by the retrieval module."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    url: str
    categories: list[str] = field(default_factory=list)

    # Populated during ranking
    embedding_score: float = 0.0   # cosine similarity vs. query embedding
    llm_score: float = 0.0         # LLM relevance score (0-10, normalised to 0-1)
    hybrid_score: float = 0.0      # weighted combination of the two

    def author_string(self) -> str:
        """Return a comma-joined author list (truncated at 5 names)."""
        if len(self.authors) <= 5:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:5]) + f" … (+{len(self.authors) - 5} more)"


@dataclass
class AgentResult:
    """Final structured output delivered to the user."""

    title: str
    authors: str
    published: str          # ISO-8601 date string
    summary: str            # 3-5 line LLM summary
    why_relevant: str       # 1-2 line relevance explanation
    key_insights: list[str] # bullet-point insights
    link: str
    score: float            # final hybrid score (1-10 scale)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "summary": self.summary,
            "why_relevant": self.why_relevant,
            "key_insights": self.key_insights,
            "link": self.link,
            "score": round(self.score, 2),
        }
