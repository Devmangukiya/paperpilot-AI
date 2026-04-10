"""
config.py
---------
Centralised configuration for the Daily Research Paper Agent.
All secrets are read from environment variables (never hard-coded).
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    # Model requested in the brief (Groq-hosted)
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", "30"))
    # Days back to look when filtering by date
    DATE_LOOKBACK_DAYS: int = int(os.getenv("DATE_LOOKBACK_DAYS", "365"))

    # ------------------------------------------------------------------ #
    #  Ranking                                                             #
    # ------------------------------------------------------------------ #
    # Weight split between embedding similarity and LLM score
    EMBEDDING_WEIGHT: float = float(os.getenv("EMBEDDING_WEIGHT", "0.4"))
    LLM_WEIGHT: float = float(os.getenv("LLM_WEIGHT", "0.6"))

    # Sentence-Transformers model (free, runs locally)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    CACHE_FILE: str = os.path.join(DATA_DIR, "cache", "seen_papers.json")
    LOG_FILE: str = os.path.join(DATA_DIR, "logs", "agent.log")
    OUTPUT_DIR: str = os.path.join(DATA_DIR, "output")

    # ------------------------------------------------------------------ #
    #  Scheduler                                                           #
    # ------------------------------------------------------------------ #
    # Time-of-day to run the daily job (24-h HH:MM, local time)
    DAILY_RUN_TIME: str = os.getenv("DAILY_RUN_TIME", "08:00")

    # ------------------------------------------------------------------ #
    #  Default keywords (overridden by CLI --keywords flag)               #
    # ------------------------------------------------------------------ #
    DEFAULT_KEYWORDS: list[str] = [
        "finance",
        "artificial intelligence",
        "federated learning",
        "explainable AI",
    ]


# Singleton – import this everywhere
config = Config()
