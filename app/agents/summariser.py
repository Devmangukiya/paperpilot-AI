"""
agents/summariser.py
--------------------
Generates a rich, structured summary for the top-ranked paper using the
Groq LLM.  Produces:
  - A concise 3-5 line plain-English summary.
  - A short explanation of why it's relevant to the user's keywords.
  - 3 bullet-point key insights.
"""

from __future__ import annotations

import json

from app.models.paper import AgentResult, Paper
from app.services.groq_client import GroqClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PaperSummariser:
    """
    Uses the Groq LLM to produce a structured summary of a single paper.

    Parameters
    ----------
    groq_client : GroqClient
        Shared Groq API client.
    keywords : list[str]
        User keywords — used to personalise the 'why relevant' section.
    """

    def __init__(self, groq_client: GroqClient, keywords: list[str]):
        self.groq = groq_client
        self.keywords = keywords
        self.topics = ", ".join(keywords)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def summarise(self, paper: Paper) -> AgentResult:
        """
        Generate a full AgentResult for *paper*.

        Returns
        -------
        AgentResult
            Structured output ready for display or JSON serialisation.
        """
        logger.info("Summarising: '%s'", paper.title[:70])
        llm_data = self._call_llm(paper)

        return AgentResult(
            title=paper.title,
            authors=paper.author_string(),
            published=paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
            summary=llm_data.get("summary", "No summary available."),
            why_relevant=llm_data.get("why_relevant", "No relevance explanation."),
            key_insights=llm_data.get("key_insights", []),
            link=paper.url,
            score=round(paper.hybrid_score, 2),
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _call_llm(self, paper: Paper) -> dict:
        """
        Ask Groq to produce a JSON summary blob for the paper.
        """
        prompt = f"""You are a senior AI research analyst. A user is tracking these topics: {self.topics}.

Here is a research paper to analyse:

TITLE: {paper.title}

AUTHORS: {paper.author_string()}

ABSTRACT:
{paper.abstract[:1200]}

Your task: respond ONLY with a valid JSON object (no markdown fences) with exactly these keys:
{{
  "summary": "<3-5 sentence plain-English summary of the paper>",
  "why_relevant": "<1-2 sentences explaining why this paper matters for {self.topics}>",
  "key_insights": ["<insight 1>", "<insight 2>", "<insight 3>"]
}}

Rules:
- summary must be 3-5 sentences, accessible to a non-expert.
- why_relevant must directly mention at least one user topic.
- key_insights must be exactly 3 short bullet strings (no bullet symbols).
- Do NOT include any text outside the JSON object.
"""

        try:
            return self.groq.complete_json(
                user_prompt=prompt,
                system_prompt="You are a precise research summariser. Output only valid JSON.",
                max_tokens=600,
                temperature=0.3,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Summarisation LLM call failed: %s", exc)
            # Return a graceful fallback so the pipeline doesn't crash
            return {
                "summary": paper.abstract[:400] + " …",
                "why_relevant": f"This paper relates to: {self.topics}.",
                "key_insights": [
                    "See abstract for details.",
                    "Full text available at the linked URL.",
                    "Manual review recommended.",
                ],
            }
