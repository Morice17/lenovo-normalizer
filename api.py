"""
Lightweight One-Vision API — no ML model loaded.

The ML runs in Google Colab and writes results to Neon Postgres.
This API reads those results and serves them to the Lovable frontend
in the exact shape mockData.json expects.

Memory footprint: ~50MB (FastAPI + pandas + psycopg2 only).
Fits comfortably on any free tier.

Endpoints:
  GET  /health              Liveness check
  GET  /analyze/results     Full mockData-shaped response from Neon
  GET  /analyze/sample      Same as /analyze/results (alias)
  POST /analyze/trigger     Returns instructions to run Colab notebook
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text

app = FastAPI(
    title="One-Vision Audience Console API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Canonical taxonomy — matches Lovable's mockData.json exactly
TAXONOMY = [
    {"id": "immersed_gamer",      "label": "Immersed Gamer",      "color": "#E2231A",
     "definition": "High-intensity PC gamers seeking competitive performance and premium Legion hardware."},
    {"id": "performance_creator", "label": "Performance Creator", "color": "#F97316",
     "definition": "Creator-gamers who use Legion devices for streaming, editing, and GPU-heavy workflows."},
    {"id": "console_gamer",       "label": "Console Gamer",       "color": "#8B5CF6",
     "definition": "Console-first players evaluating PC gaming through ecosystem and cross-platform compatibility."},
    {"id": "esports_aspirant",    "label": "Esports Aspirant",    "color": "#00C896",
     "definition": "Competitive multiplayer audiences motivated by frame rates, refresh rates, and tournament culture."},
    {"id": "student_gamer",       "label": "Student Gamer",       "color": "#3B82F6",
     "definition": "Students balancing study, social play, portability, and price-to-performance needs."},
    {"id": "cloud_casual",        "label": "Cloud Casual",        "color": "#F7B538",
     "definition": "Light and cloud gaming audiences responding to accessibility and subscription-led play."},
]

LABEL_TO_ID = {seg["label"]: seg["id"] for seg in TAXONOMY}
SEGMENT_IDS = [seg["id"] for seg in TAXONOMY]
REGIONS = ["EMEA", "NA", "LATAM"]


def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    return create_engine(db_url)


def build_response(normalized: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """Build the full mockData.json-shaped response from Postgres tables."""

    # ── raw_inputs ──────────────────────────────────────────────
    raw_inputs = []
    for i, row in normalized.iterrows():
        raw_inputs.append({
            "row_id": f"r{i+1:03d}",
            "region": str(row.get("region", "")),
            "channel": str(row.get("sheet", row.get("channel", ""))),
            "raw_targeting_string": str(row.get("raw_targeting", "")),
            "char_count": int(row.get("char_count", 0)),
        })

    # ── normalized_outputs ───────────────────────────────────────
    normalized_outputs = []
    for i, row in normalized.iterrows():
        label = str(row.get("canonical_label", ""))
        seg_id = LABEL_TO_ID.get(label, label.lower().replace(" ", "_"))
        normalized_outputs.append({
            "row_id": f"r{i+1:03d}",
            "canonical_id": seg_id,
            "canonical_label": label,
            "confidence_score": float(row.get("confidence_score", 0)),
            "status": str(row.get("status", "needs_review")),
        })

    # ── regional_breakdown ───────────────────────────────────────
    regional_breakdown = []
    for region in REGIONS:
        region_rows = normalized[normalized["region"] == region]
        total = len(region_rows)
        entry: dict = {"region": region, "row_count": total}
        for seg in TAXONOMY:
            if total > 0:
                count = (region_rows["canonical_label"] == seg["label"]).sum()
                entry[seg["id"]] = round(int(count) / total * 100, 1)
            else:
                entry[seg["id"]] = 0
        regional_breakdown.append(entry)

    # ── segment_table ────────────────────────────────────────────
    segment_table = []
    if not summary.empty and "canonical_label" in summary.columns:
        for seg in TAXONOMY:
            row = summary[summary["canonical_label"] == seg["label"]]
            if not row.empty:
                r = row.iloc[0]
                segment_table.append({
                    "id": seg["id"],
                    "segment_label": seg["label"],
                    "emea_pct":  float(r.get("emea_pct", 0)),
                    "na_pct":    float(r.get("na_pct", 0)),
                    "latam_pct": float(r.get("latam_pct", 0)),
                    "consistency": str(r.get("consistency", "aligned")),
                })
            else:
                segment_table.append({
                    "id": seg["id"],
                    "segment_label": seg["label"],
                    "emea_pct": 0, "na_pct": 0, "latam_pct": 0,
                    "consistency": "aligned",
                })
    else:
        # Build from normalized table directly
        for seg in TAXONOMY:
            pcts = {}
            for region in REGIONS:
                region_rows = normalized[normalized["region"] == region]
                total = len(region_rows)
                count = (region_rows["canonical_label"] == seg["label"]).sum()
                pcts[region] = round(int(count) / total * 100, 1) if total > 0 else 0
            spread = max(pcts.values()) - min(pcts.values())
            consistency = "aligned" if spread < 12 else "diverging"
            segment_table.append({
                "id": seg["id"],
                "segment_label": seg["label"],
                "emea_pct":  pcts["EMEA"],
                "na_pct":    pcts["NA"],
                "latam_pct": pcts["LATAM"],
                "consistency": consistency,
            })

    # ── kpis ─────────────────────────────────────────────────────
    total = len(normalized)
    auto = (normalized["status"] == "auto_approved").sum() if "status" in normalized.columns else 0
    needs = (normalized["status"] == "needs_review").sum() if "status" in normalized.columns else 0

    # Consistency score
    spreads = [
        abs(r["emea_pct"] - r["na_pct"]) +
        abs(r["na_pct"] - r["latam_pct"]) +
        abs(r["emea_pct"] - r["latam_pct"])
        for r in segment_table
    ]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0
    consistency_score = max(0, min(100, round(100 - avg_spread)))

    # Drift alert
    worst_seg = None
    worst_spread = 0
    worst_region = None
    for r in segment_table:
        pcts = {"EMEA": r["emea_pct"], "NA": r["na_pct"], "LATAM": r["latam_pct"]}
        max_r = max(pcts, key=pcts.get)
        others = [v for k, v in pcts.items() if k != max_r]
        spread = pcts[max_r] - (sum(others) / len(others)) if others else 0
        if spread > worst_spread:
            worst_spread = spread
            worst_seg = r["segment_label"]
            worst_region = max_r

    drift_msg = (
        f"{worst_region} is over-indexing '{worst_seg}' by "
        f"{round(worst_spread)} pts vs the global mix."
        if worst_spread >= 12
        else "No significant regional drift detected."
    )

    # AI Q&A
    diverging = [r["segment_label"] for r in segment_table if r["consistency"] == "diverging"]
    aligned = [r["segment_label"] for r in segment_table if r["consistency"] == "aligned"]

    ai_qa = [
        {"question": "Where is the largest audience drift?",
         "answer": drift_msg},
        {"question": "Which regions are most aligned?",
         "answer": f"The segments holding closest to a shared mix are: {', '.join(aligned) if aligned else 'none'}. "
                   f"Diverging segments: {', '.join(diverging) if diverging else 'none'}."},
        {"question": "What should the CMO do next?",
         "answer": "Review the regions flagged in the drift alert before launch. Confirm whether the "
                   "divergence reflects intentional localization or campaign drift."},
        {"question": "How many rows need review?",
         "answer": f"{int(needs)} normalized rows fell below the auto-approval threshold "
                   f"and are queued for regional planner review."},
    ]

    return {
        "canonical_taxonomy": TAXONOMY,
        "raw_inputs": raw_inputs,
        "normalized_outputs": normalized_outputs,
        "kpis": {
            "rows_normalized": int(total),
            "high_confidence_pct": round(int(auto) / total * 100) if total else 0,
            "needs_review_count": int(needs),
            "audience_consistency_score": consistency_score,
            "total_reach": f"{max(1.0, total * 0.015):.1f}M",
            "segment_overlap": round(consistency_score * 0.6),
            "drift_alert": {"message": drift_msg},
        },
        "regional_breakdown": regional_breakdown,
        "segment_table": segment_table,
        "ai_qa_responses": ai_qa,
    }


@app.get("/health")
def health():
    return {"status": "ok", "mode": "lightweight — ML runs in Colab"}


@app.get("/analyze/results")
@app.get("/analyze/sample")
def get_results():
    """
    Read pre-computed ML results from Neon Postgres and return
    the full mockData.json-shaped document for the Lovable frontend.

    Run the Colab notebook to update these results with new data.
    """
    try:
        engine = get_engine()
        normalized = pd.read_sql("SELECT * FROM audience_normalized", engine)
        try:
            summary = pd.read_sql(
                "SELECT * FROM audience_consistency_summary", engine
            )
        except Exception:
            summary = pd.DataFrame()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    if normalized.empty:
        raise HTTPException(
            status_code=404,
            detail="No results found. Run the Colab notebook first to populate the database.",
        )

    return build_response(normalized, summary)


@app.post("/analyze/trigger")
def trigger_analysis():
    """
    Tells the frontend how to trigger a fresh ML run.
    The actual ML runs in Google Colab.
    """
    return {
        "status": "manual_trigger_required",
        "message": "Open the Colab notebook and run all cells to process new data.",
        "notebook_url": "https://colab.research.google.com",
        "steps": [
            "1. Open Lenovo_Audience_Normalizer.ipynb in Google Colab",
            "2. Upload new regional Excel files (optional)",
            "3. Run Runtime → Run All",
            "4. Refresh the Lovable dashboard",
        ],
    }
