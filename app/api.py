"""
app/api.py
----------
FastAPI server that the frontend (frontend/index.html) talks to.

Endpoints
---------
GET  /health       Liveness check — confirms server + Groq model name.
POST /run          Run the full pipeline with a list of keywords.
GET  /latest       Return the most recently saved result JSON from disk.
GET  /history      Return last N saved results.

CORS is open to all origins so the HTML file can be opened directly
from the filesystem (file:// protocol) or any dev server.

Run with:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Daily Research Paper Agent API",
    description=(
        "Fetch, rank, and summarise research papers from arXiv.\n\n"
        "Open `frontend/index.html` in your browser to use the UI."
    ),
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────
# Allow all origins so the standalone HTML file works even when opened
# directly from disk (file:// protocol) or from a different dev port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict to specific origins in production
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────────────
# If the frontend/ directory exists, serve it at /ui so you can visit
# http://localhost:8000/ui/  without opening the file manually.
_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend",
)

if os.path.isdir(_FRONTEND_DIR):
    app.mount(
        "/ui",
        StaticFiles(directory=_FRONTEND_DIR, html=True),
        name="frontend",
    )
    logger.info("Frontend served at /ui  ->  %s", _FRONTEND_DIR)


# ── Request / Response models ─────────────────────────────────────────

class RunRequest(BaseModel):
    keywords: Optional[list[str]] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


from fastapi.responses import RedirectResponse

# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect to the frontend UI."""
    return RedirectResponse(url="/ui/")

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    """
    Liveness check.
    Returns server status, the configured Groq model name, and API version.
    The frontend 'Test connection' button calls this.
    """
    return HealthResponse(
        status="ok",
        model=config.GROQ_MODEL,
        version="1.0.0",
    )

@app.get("/debug-env", tags=["Utility"])
def debug_env():
    import os
    has_key = "GROQ_API_KEY" in os.environ
    val_len = len(os.environ.get("GROQ_API_KEY", ""))
    return {"has_key_in_os": has_key, "key_length": val_len, "config_len": len(config.GROQ_API_KEY)}


@app.post("/run", tags=["Pipeline"])
def run_pipeline(body: RunRequest = RunRequest()) -> JSONResponse:
    """
    Run the full research pipeline.

    Body (JSON)
    -----------
    { "keywords": ["finance", "federated learning", "XAI"] }

    keywords is optional — omit it to use DEFAULT_KEYWORDS from config.
    Returns the top-ranked paper as a structured JSON object.
    """
    keywords = body.keywords or config.DEFAULT_KEYWORDS
    logger.info("POST /run — keywords: %s", keywords)

    try:
        from app.pipelines.daily_pipeline import DailyResearchPipeline
        pipeline = DailyResearchPipeline(keywords=keywords)
        result = pipeline.run()
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No paper found for the given keywords. Try broader terms.",
        )

    return JSONResponse(content=result.to_dict())


@app.get("/latest", tags=["Pipeline"])
def get_latest() -> JSONResponse:
    """
    Return the most recently saved result from data/output/.
    Useful for checking yesterday's result without re-running the pipeline.
    """
    pattern = os.path.join(config.OUTPUT_DIR, "paper_*.json")
    files   = sorted(glob.glob(pattern), reverse=True)

    if not files:
        raise HTTPException(
            status_code=404,
            detail="No saved results yet. Run the pipeline first.",
        )

    try:
        with open(files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read saved result: {exc}",
        ) from exc

    return JSONResponse(content=data)


@app.get("/history", tags=["Pipeline"])
def get_history(limit: int = 10) -> JSONResponse:
    """
    Return the last `limit` saved results, newest first.
    """
    pattern = os.path.join(config.OUTPUT_DIR, "paper_*.json")
    files   = sorted(glob.glob(pattern), reverse=True)[:limit]

    results = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass

    return JSONResponse(content={"count": len(results), "results": results})
