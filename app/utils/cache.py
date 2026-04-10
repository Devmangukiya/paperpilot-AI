"""
utils/cache.py
--------------
Simple JSON-based cache to track which arXiv papers have already been
processed.  Prevents the agent from delivering the same paper twice.
"""

import json
import os
from datetime import datetime, timedelta, timezone

from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PaperCache:
    """
    Manages a local JSON file that maps arxiv_id → ISO timestamp of when
    it was first seen.

    The cache is automatically pruned every load so it never grows unboundedly
    (entries older than 2× DATE_LOOKBACK_DAYS are removed).
    """

    def __init__(self, cache_file: str = config.CACHE_FILE):
        self.cache_file = cache_file
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        self._data: dict[str, str] = self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def is_seen(self, arxiv_id: str) -> bool:
        """Return True if this paper was already processed."""
        return arxiv_id in self._data

    def mark_seen(self, arxiv_id: str) -> None:
        """Record a paper as processed and persist to disk."""
        self._data[arxiv_id] = datetime.now(timezone.utc).isoformat()
        self._save()
        logger.debug("Cache: marked %s as seen.", arxiv_id)

    def unseen_ids(self, arxiv_ids: list[str]) -> list[str]:
        """Filter a list of IDs, returning only those not yet seen."""
        return [aid for aid in arxiv_ids if not self.is_seen(aid)]

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load(self) -> dict[str, str]:
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            pruned = self._prune(data)
            return pruned
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache load failed (%s); starting fresh.", exc)
            return {}

    def _save(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError as exc:
            logger.error("Cache save failed: %s", exc)

    @staticmethod
    def _prune(data: dict[str, str]) -> dict[str, str]:
        """Remove entries older than 2× DATE_LOOKBACK_DAYS."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=config.DATE_LOOKBACK_DAYS * 2
        )
        cleaned = {
            k: v
            for k, v in data.items()
            if datetime.fromisoformat(v) > cutoff
        }
        return cleaned
