"""
main.py
-------
FastAPI backend — Role 2 of the Git ingestion pipeline.

Serves the data produced by pipeline.py to the Next.js frontend (Role 3).
Exposes 3 endpoints:

  GET  /api/timeline               — lightweight chronological commit list
  GET  /api/commit/{commit_hash}   — full commit record including graph state
  POST /api/explain-anomaly        — LLM anomaly analysis via litellm

Environment variables
---------------------
USE_REAL_LLM   : Set to "true" to call the real LLM (requires API key env var).
                 Any other value (or absent) returns a mock string.
OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY
               : Passed through to litellm automatically.
LLM_MODEL      : litellm model string, e.g. "gpt-4o", "claude-3-5-sonnet-20241022",
                 "gemini/gemini-1.5-pro". Defaults to "gpt-4o".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE:    Path = Path(__file__).parent / "pipeline_output.json"
USE_REAL_LLM: bool = os.getenv("USE_REAL_LLM", "false").strip().lower() == "true"
LLM_MODEL:    str  = os.getenv("LLM_MODEL", "gpt-4o")

_SYSTEM_PROMPT = (
    "You are an expert software architect. "
    "Review this code diff. "
    "In ONE sentence, explain why this structural change degrades the "
    "repository's maintainability or architecture. "
    "Be concise and technical."
)

_MOCK_EXPLANATION = (
    "MOCK: This PR bypasses the database abstraction layer, directly coupling "
    "the frontend component to the database model."
)


# ---------------------------------------------------------------------------
# In-memory data store — populated once at startup
# ---------------------------------------------------------------------------

class DataStore:
    """
    Holds the ingested pipeline data in two structures for O(1) access:

    - ``chronological``  : ordered list of all commit records (for timeline)
    - ``by_hash``        : dict[commit_hash → full record]  (for detail view)
    """

    def __init__(self) -> None:
        self.chronological: list[dict[str, Any]] = []
        self.by_hash:       dict[str, dict[str, Any]] = {}

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "Run  python pipeline.py  first to generate it."
            )

        with open(path, encoding="utf-8") as fh:
            records: list[dict[str, Any]] = json.load(fh)

        self.chronological = records
        self.by_hash       = {r["commit_hash"]: r for r in records}

        logger.info(
            "Loaded %d commit record(s) from %s", len(records), path
        )


store = DataStore()

# ---------------------------------------------------------------------------
# Analysis job state — tracks the background clone+pipeline job
# ---------------------------------------------------------------------------

class AnalysisJob:
    """Singleton tracking the current/last analysis job."""
    def __init__(self) -> None:
        self.status:  str = "idle"   # idle | cloning | analyzing | done | error
        self.message: str = "No analysis running."
        self.repo_url: str = ""

    def reset(self, repo_url: str) -> None:
        self.repo_url = repo_url
        self.status  = "cloning"
        self.message = f"Cloning {repo_url}..."


job = AnalysisJob()


# ---------------------------------------------------------------------------
# Lifespan — load data into memory once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    """Load pipeline_output.json into the in-memory DataStore on startup."""
    store.load(DATA_FILE)
    logger.info(
        "LLM mode: %s  |  Model: %s",
        "REAL" if USE_REAL_LLM else "MOCK",
        LLM_MODEL if USE_REAL_LLM else "N/A",
    )
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Git Ingestion Engine API",
    description=(
        "Serves architectural health metrics, dependency graphs, "
        "and LLM anomaly explanations for a Git repository."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local hackathon development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnomalyExplainRequest(BaseModel):
    """Request body for POST /api/explain-anomaly."""

    git_diff_snippet: str = Field(
        ...,
        description="The raw git diff patch text to be analysed by the LLM.",
        min_length=1,
    )
    commit_hash: str | None = Field(
        default=None,
        description="Optional commit hash for logging/context.",
    )
    trigger_reason: str | None = Field(
        default=None,
        description="Human-readable reason the anomaly was flagged.",
    )
    topological_delta: str | None = Field(
        default=None,
        description="Structural graph changes (new edges, new cycles) detected by AST.",
    )


class AnomalyExplainResponse(BaseModel):
    """Response body for POST /api/explain-anomaly."""

    explanation: str
    model_used:  str
    commit_hash: str | None = None
    is_mock:     bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/timeline",
    response_model=list[dict[str, Any]],
    summary="Chronological commit timeline",
    description=(
        "Lightweight list of all commits — only commit_hash, timestamp, "
        "author, and health_metrics. The heavy graph_state is stripped "
        "so the timeline renders fast."
    ),
)
async def get_timeline() -> list[dict[str, Any]]:
    return [
        {
            "commit_hash":    r["commit_hash"],
            "timestamp":      r["timestamp"],
            "author":         r["author"],
            "health_metrics": r["health_metrics"],
        }
        for r in store.chronological
    ]


@app.get(
    "/api/commit/{commit_hash}",
    response_model=dict[str, Any],
    summary="Full commit detail",
    description=(
        "Full record for one commit including the React Flow-compatible "
        "graph_state (nodes/edges) and the llm_trigger_payload."
    ),
)
async def get_commit(commit_hash: str) -> dict[str, Any]:
    """
    Look up a commit by its full or abbreviated SHA-1.

    - Full hash → direct dict lookup (O(1))
    - Short hash → prefix scan; raises 400 on ambiguity
    - Not found  → 404
    """
    record = store.by_hash.get(commit_hash)

    if record is None:
        matches = [r for h, r in store.by_hash.items() if h.startswith(commit_hash)]
        if len(matches) == 1:
            record = matches[0]
        elif len(matches) > 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Ambiguous abbreviated hash '{commit_hash}' — "
                    f"matches {len(matches)} commits. Use a longer prefix."
                ),
            )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Commit '{commit_hash}' not found in pipeline data.",
        )

    return record


@app.post(
    "/api/explain-anomaly",
    response_model=AnomalyExplainResponse,
    summary="LLM anomaly explanation",
    description=(
        "Accepts a git diff snippet and returns a one-sentence architectural "
        "risk explanation. Uses litellm so the model is swappable via env var. "
        "Returns a mock string when USE_REAL_LLM != 'true'."
    ),
)
async def explain_anomaly(request: AnomalyExplainRequest) -> AnomalyExplainResponse:
    """
    **Defended LLM Trigger.**

    When ``USE_REAL_LLM=true`` the diff is forwarded to the configured model
    via litellm.  Otherwise a static mock string is returned — no tokens spent.
    """
    if not request.git_diff_snippet.strip():
        raise HTTPException(status_code=422, detail="git_diff_snippet must not be empty.")

    logger.info(
        "explain-anomaly  commit=%s  diff_len=%d  real_llm=%s",
        request.commit_hash or "unknown",
        len(request.git_diff_snippet),
        USE_REAL_LLM,
    )

    if not USE_REAL_LLM:
        return AnomalyExplainResponse(
            explanation = _MOCK_EXPLANATION,
            model_used  = "mock",
            commit_hash = request.commit_hash,
            is_mock     = True,
        )

    # ── Real LLM call via litellm ────────────────────────────────────────
    try:
        import litellm  # imported here so mock mode has zero litellm overhead

        user_message = (
            f"Trigger reason: {request.trigger_reason or 'N/A'}\n\n"
        )
        if request.topological_delta:
            user_message += f"Structural graph changes detected by AST analysis:\n{request.topological_delta}\n\n"
        user_message += f"Git diff:\n{request.git_diff_snippet}"

        response = await litellm.acompletion(
            model    = LLM_MODEL,
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = 120,   # one sentence — keep it short
            temperature = 0.3,   # low temp for deterministic architectural critique
        )

        explanation: str = response.choices[0].message.content.strip()

    except Exception as exc:  # noqa: BLE001
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {exc}",
        ) from exc

    return AnomalyExplainResponse(
        explanation = explanation,
        model_used  = LLM_MODEL,
        commit_hash = request.commit_hash,
        is_mock     = False,
    )


# ---------------------------------------------------------------------------
# Repository Analysis (clone + pipeline)
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub/Git repository URL to analyze.")
    max_commits: int = Field(default=600, ge=50, le=2000, description="Max commits to process.")


def _run_analysis_sync(repo_url: str, max_commits: int) -> None:
    """
    Blocking function run in a background thread:
      1. git clone --depth N <repo_url> → temp dir
      2. python pipeline.py <dir> --max-commits N
      3. Move pipeline_output.json next to main.py
      4. Reload the in-memory store
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="rh-analyze-"))
    try:
        # Step 1: clone
        job.status  = "cloning"
        job.message = f"Cloning {repo_url} (depth={max_commits})..."
        logger.info(job.message)
        subprocess.run(
            ["git", "clone", "--depth", str(max_commits), repo_url, str(tmpdir)],
            check=True, capture_output=True, timeout=300,
        )

        # Step 2: run pipeline
        job.status  = "analyzing"
        job.message = f"Ingesting up to {max_commits} commits — this may take a few minutes..."
        logger.info(job.message)
        pipeline_script = Path(__file__).parent / "pipeline.py"
        output_tmp = tmpdir / "pipeline_output.json"
        subprocess.run(
            ["python", str(pipeline_script), str(tmpdir),
             "--output", str(output_tmp),
             "--max-commits", str(max_commits)],
            check=True, capture_output=False, timeout=600,
            cwd=Path(__file__).parent,
        )

        # Step 3: move output next to main.py
        shutil.move(str(output_tmp), str(DATA_FILE))

        # Step 4: reload store
        store.load(DATA_FILE)

        job.status  = "done"
        job.message = f"Analysis complete. {len(store.chronological)} commits loaded."
        logger.info(job.message)

    except subprocess.CalledProcessError as exc:
        job.status  = "error"
        job.message = f"Git/pipeline error: {exc}"
        logger.error(job.message)
    except Exception as exc:  # noqa: BLE001
        job.status  = "error"
        job.message = f"Unexpected error: {exc}"
        logger.error(job.message)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post(
    "/api/analyze",
    summary="Analyze a Git repository",
    description=(
        "Accepts a public Git repository URL. Clones it, runs the ingestion pipeline, "
        "and reloads the data store. Poll /api/analyze/status for progress."
    ),
)
async def analyze_repo(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    if job.status in ("cloning", "analyzing"):
        raise HTTPException(
            status_code=409,
            detail=f"An analysis is already running: {job.message}",
        )
    job.reset(request.repo_url)
    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,  # default ThreadPoolExecutor
        _run_analysis_sync,
        request.repo_url,
        request.max_commits,
    )
    return {"status": "started", "message": job.message}


@app.get(
    "/api/analyze/status",
    summary="Poll analysis progress",
)
async def analyze_status() -> dict[str, str]:
    return {
        "status":   job.status,
        "message":  job.message,
        "repo_url": job.repo_url,
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
async def health() -> dict[str, Any]:
    return {
        "status":    "ok",
        "commits":   len(store.chronological),
        "llm_mode":  "real" if USE_REAL_LLM else "mock",
        "llm_model": LLM_MODEL if USE_REAL_LLM else "N/A",
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = True,
        log_level = "info",
    )
