"""
utils/logger.py
---------------
Centralised logger used across all modules.
Writes to both console and a rotating log file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from app.config import config


def get_logger(name: str = "research_agent") -> logging.Logger:
    """
    Return (and lazily configure) a named logger.

    Parameters
    ----------
    name : str
        Logger name (use __name__ in calling module for clarity).
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # --- File handler (rotating, max 5 MB × 3 backups) ---
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    fh = RotatingFileHandler(
        config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
