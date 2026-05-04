"""
FastAPI service for the One-Vision Audience Console backend.

Endpoints:
  GET  /health                       Liveness check
  GET  /taxonomy                     Canonical segments (frontend shape)
  POST /normalize                    Single string -> per-row result
  POST /normalize/batch              List of strings -> list of results
  POST /normalize/csv                CSV upload -> normalized CSV download

  GET  /analyze/sample               Run analysis on bundled sample CSVs.
                                     Returns the full mockData.json shape.
  POST /analyze                      Upload three regional CSVs (or one
                                     combined CSV) and get the full
                                     mockData.json-shaped document back.

  POST /analyze/from-postgres        Read from Airbyte landing tables in
                                     Postgres, run analysis, return shape.

CORS: open by default so Lovable's preview and published URLs can hit it.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from analyzer import REGION_KEYS, analyze
from normalizer import AudienceNormalizer
from taxonomy import to_frontend_taxonomy


app = FastAPI(
    title="Lenovo One-Vision Audience Console API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Bundled sample data path — used by /analyze/sample for demos.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_FILES = {
    "EMEA": DATA_DIR / "emea_germany_plan.csv",
    "NA":   DATA_DIR / "na_plan.csv",
    "LATAM": DATA_DIR / "latam_plan.csv",
}


_normalizer: Optional[AudienceNormalizer] = None


def get_normalizer() -> AudienceNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = AudienceNormalizer()
    return _normalizer


@app.on_event("startup")
def _warm_model() -> None:
    get_normalizer()


# --------- Pydantic schemas ---------

class NormalizeRequest(BaseModel):
    text: str

class NormalizeBatchRequest(BaseModel):
    texts: List[str]

class FromPostgresRequest(BaseModel):
    sources: dict = {
        "EMEA": "public.emea_germany_plan",
        "NA":   "public.na_plan",
        "LATAM": "public.latam_plan",
    }
    audience_column: str = "target_audience"
    channel_column: str = "channel"


# --------- Liveness + reference ---------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _normalizer is not None}


@app.get("/taxonomy")
def taxonomy() -> dict:
    return {"canonical_taxonomy": to_frontend_taxonomy()}


# --------- Per-row normalize endpoints ---------

@app.post("/normalize")
def normalize_one(req: NormalizeRequest) -> dict:
    return get_normalizer().normalize(req.text).to_dict()


@app.post("/normalize/batch")
def normalize_batch(req: NormalizeBatchRequest) -> dict:
    results = get_normalizer().normalize_batch(req.texts)
    return {"results": [r.to_dict() for r in results]}


@app.post("/normalize/csv")
async def normalize_csv(
    file: UploadFile = File(...),
    audience_column: str = "target_audience",
) -> StreamingResponse:
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if audience_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"CSV missing column '{audience_column}'. "
                f"Found: {list(df.columns)}"
            ),
        )

    raws = df[audience_column].fillna("").astype(str).tolist()
    results = get_normalizer().normalize_batch(raws)
    df["normalized_label"] = [r.predicted_label for r in results]
    df["confidence"] = [round(r.confidence, 4) for r in results]
    df["needs_review"] = [r.needs_review for r in results]

    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="normalized_{file.filename}"'
            )
        },
    )


# --------- Frontend-shaped analyze endpoints ---------

def _stack_regional_dfs(
    region_to_df: dict,
    audience_col: str,
    channel_col: str,
) -> pd.DataFrame:
    """Concatenate the three regional dataframes with a region column."""
    frames = []
    for region, df in region_to_df.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df["region"] = region
        # Make sure required columns exist; default channel to empty if absent.
        if audience_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{region} CSV missing audience column "
                    f"'{audience_col}'. Found: {list(df.columns)}"
                ),
            )
        if channel_col not in df.columns:
            df[channel_col] = ""
        frames.append(df[["region", audience_col, channel_col]])
    if not frames:
        raise HTTPException(status_code=400, detail="No regional data provided.")
    return pd.concat(frames, ignore_index=True)


@app.get("/analyze/sample")
def analyze_sample(
    audience_column: str = "target_audience",
    channel_column: str = "channel",
) -> dict:
    """Run analysis against the bundled sample CSVs. No upload required.
    Demo-friendly: hit this endpoint and you get a fully populated
    document the frontend can render."""
    region_dfs = {}
    for region, path in SAMPLE_FILES.items():
        if not path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Sample file missing: {path}",
            )
        region_dfs[region] = pd.read_csv(path)
    df = _stack_regional_dfs(region_dfs, audience_column, channel_column)
    return analyze(
        df,
        get_normalizer(),
        region_col="region",
        audience_col=audience_column,
        channel_col=channel_column,
    )


@app.post("/analyze")
async def analyze_uploads(
    emea: Optional[UploadFile] = File(None),
    na: Optional[UploadFile] = File(None),
    latam: Optional[UploadFile] = File(None),
    audience_column: str = "target_audience",
    channel_column: str = "channel",
) -> dict:
    """Accepts up to three regional CSVs as multipart form fields named
    `emea`, `na`, `latam`. Any subset is fine; missing regions are
    skipped. Returns the full mockData.json-shaped document."""
    region_dfs: dict = {}
    for region, upload in (("EMEA", emea), ("NA", na), ("LATAM", latam)):
        if upload is None:
            continue
        try:
            content = await upload.read()
            region_dfs[region] = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse {region} CSV: {e}",
            )
    if not region_dfs:
        raise HTTPException(
            status_code=400,
            detail="At least one of emea, na, latam must be provided.",
        )
    df = _stack_regional_dfs(region_dfs, audience_column, channel_column)
    return analyze(
        df,
        get_normalizer(),
        region_col="region",
        audience_col=audience_column,
        channel_col=channel_column,
    )


@app.post("/analyze/from-postgres")
def analyze_from_postgres(req: FromPostgresRequest) -> dict:
    """Read the three Airbyte landing tables from Postgres, run analysis,
    return the frontend-shaped document. This is the production path."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise HTTPException(
            status_code=500, detail="DATABASE_URL not configured"
        )
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(db_url)
    region_dfs: dict = {}
    try:
        for region, table in req.sources.items():
            with conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(
                    f"SELECT * FROM {table} "
                    f"WHERE {req.audience_column} IS NOT NULL"
                )
                rows = cur.fetchall()
            if rows:
                region_dfs[region] = pd.DataFrame(rows)
    finally:
        conn.close()

    if not region_dfs:
        raise HTTPException(
            status_code=400,
            detail="No data found in the configured Postgres tables.",
        )
    df = _stack_regional_dfs(
        region_dfs, req.audience_column, req.channel_column
    )
    return analyze(
        df,
        get_normalizer(),
        region_col="region",
        audience_col=req.audience_column,
        channel_col=req.channel_column,
    )
