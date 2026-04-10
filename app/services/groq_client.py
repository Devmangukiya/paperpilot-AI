"""
services/groq_client.py
-----------------------
Thin wrapper around the Groq REST API (OpenAI-compatible endpoint).
Does NOT import the openai package; uses plain `requests` as required.
"""

from __future__ import annotations

import json
import time

import requests

from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum retries on transient errors (rate limits / 5xx)
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds


class GroqClient:
    """
    Minimal client for the Groq chat completions API.

    Usage
    -----
    client = GroqClient()
    text = client.complete("Summarise this abstract: ...")
    """

    def __init__(
        self,
        api_key: str = config.GROQ_API_KEY,
        model: str = config.GROQ_MODEL,
        base_url: str = config.GROQ_BASE_URL,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set.  Add it to your .env file."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful research assistant.",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """
        Send a chat completion request and return the assistant's reply.

        Parameters
        ----------
        user_prompt : str
            The main prompt / question.
        system_prompt : str
            Optional system-level instruction.
        max_tokens : int
            Max tokens in the response.
        temperature : float
            Sampling temperature (lower = more deterministic).

        Returns
        -------
        str
            The model's text response, stripped of leading/trailing whitespace.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status == 429 or (status and status >= 500):
                    # Rate-limited or server error → retry
                    wait = _RETRY_DELAY * attempt
                    logger.warning(
                        "Groq API HTTP %s on attempt %d/%d. Retrying in %.1fs …",
                        status, attempt, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Groq API error: %s", exc)
                    raise

            except requests.exceptions.RequestException as exc:
                logger.error("Groq request failed: %s", exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * attempt)
                else:
                    raise

        raise RuntimeError("Groq API: max retries exceeded.")

    def complete_json(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful research assistant. Always respond with valid JSON only.",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict:
        """
        Same as `complete` but parses and returns the response as a dict.
        Robust to markdown code-fences (```json ... ```) in the response.
        """
        raw = self.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Strip common markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first and last fence lines
            cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Groq JSON response: %s\nRaw: %s", exc, raw)
            raise
