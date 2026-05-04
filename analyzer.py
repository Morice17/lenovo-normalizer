"""
Frontend-shaped analyzer for the One-Vision Audience Console.

normalizer.py produces per-row labels. This module wraps that and
produces the full document the frontend consumes:

    {
      canonical_taxonomy: [...],
      raw_inputs:         [...],
      normalized_outputs: [...],
      kpis:               { rows_normalized, high_confidence_pct,
                            needs_review_count, audience_consistency_score,
                            total_reach, segment_overlap, drift_alert },
      regional_breakdown: [{region, row_count, <segment_id>: pct, ...}],
      segment_table:      [{id, segment_label, emea_pct, na_pct,
                            latam_pct, consistency}],
      ai_qa_responses:    [{question, answer}]
    }

This is a drop-in replacement for src/data/mockData.json.

Design choices:
  - status="auto_approved" if confidence >= AUTO_APPROVE_THRESHOLD,
    else "needs_review". Below CONFIDENCE_THRESHOLD the row is still
    given its top label so the frontend can show *something*, but the
    status stays "needs_review".
  - audience_consistency_score: 100 minus the mean per-segment spread
    across regions. A perfectly aligned mix scores 100; a heavily
    diverging one (LATAM vs EMEA on Console Gamer at 39 vs 8) scores
    much lower. Bounded to [0, 100].
  - consistency tag per segment:
      "regional_only"  if one region's pct >= 25 and the others' < 15
      "diverging"      if max-min spread >= 12 pts
      "aligned"        otherwise
  - drift_alert: surfaces the largest single regional outlier in plain
    English. Empty message when nothing exceeds the threshold.
  - ai_qa_responses: deterministic templated answers derived from the
    computed stats. No LLM call required for the demo.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from normalizer import AudienceNormalizer
from taxonomy import (
    AUTO_APPROVE_THRESHOLD,
    TAXONOMY,
    label_to_id,
    to_frontend_taxonomy,
)


REGION_KEYS = ["EMEA", "NA", "LATAM"]


def _row_id(idx: int) -> str:
    return f"r{idx + 1:03d}"


def _classify_status(confidence: float) -> str:
    return "auto_approved" if confidence >= AUTO_APPROVE_THRESHOLD else "needs_review"


def _consistency_tag(pcts: Dict[str, float]) -> str:
    values = list(pcts.values())
    if not values:
        return "aligned"
    spread = max(values) - min(values)
    high_regions = [r for r, v in pcts.items() if v >= 25]
    low_regions = [r for r, v in pcts.items() if v < 15]
    if len(high_regions) == 1 and len(low_regions) == len(REGION_KEYS) - 1:
        return "regional_only"
    if spread >= 12:
        return "diverging"
    return "aligned"


def _format_pct(n: int, total: int) -> int:
    return round(100 * n / total) if total else 0


def _build_regional_breakdown(
    rows_by_region: Dict[str, List[dict]],
) -> List[dict]:
    """One entry per region, with pct of each segment."""
    breakdown = []
    seg_ids = [seg.id for seg in TAXONOMY]
    for region in REGION_KEYS:
        rows = rows_by_region.get(region, [])
        total = len(rows)
        counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["canonical_id"]] += 1
        entry = {"region": region, "row_count": total}
        for sid in seg_ids:
            entry[sid] = _format_pct(counts[sid], total)
        breakdown.append(entry)
    return breakdown


def _build_segment_table(breakdown: List[dict]) -> List[dict]:
    by_region = {b["region"]: b for b in breakdown}
    table = []
    for seg in TAXONOMY:
        pcts = {r: by_region[r].get(seg.id, 0) for r in REGION_KEYS}
        table.append({
            "id": seg.id,
            "segment_label": seg.label,
            "emea_pct": pcts["EMEA"],
            "na_pct": pcts["NA"],
            "latam_pct": pcts["LATAM"],
            "consistency": _consistency_tag(pcts),
        })
    return table


def _build_drift_alert(segment_table: List[dict]) -> dict:
    """Find the largest single-region divergence and describe it."""
    worst = None
    worst_spread = 0
    for row in segment_table:
        pcts = {
            "EMEA": row["emea_pct"],
            "NA": row["na_pct"],
            "LATAM": row["latam_pct"],
        }
        max_region = max(pcts, key=pcts.get)
        max_val = pcts[max_region]
        others = [v for k, v in pcts.items() if k != max_region]
        if not others:
            continue
        avg_others = sum(others) / len(others)
        spread = max_val - avg_others
        if spread > worst_spread:
            worst_spread = spread
            worst = {
                "segment": row["segment_label"],
                "region": max_region,
                "spread_pts": round(spread),
                "max_pct": max_val,
                "others_avg": round(avg_others),
            }
    if not worst or worst_spread < 12:
        return {"message": "No regional drift exceeds the alert threshold."}
    return {
        "message": (
            f"{worst['region']} is over-indexing {worst['segment']} by "
            f"{worst['spread_pts']} pts versus the global campaign mix."
        )
    }


def _audience_consistency_score(segment_table: List[dict]) -> int:
    """100 = perfectly aligned. Lower with larger per-segment spreads."""
    if not segment_table:
        return 0
    spreads = []
    for row in segment_table:
        pcts = [row["emea_pct"], row["na_pct"], row["latam_pct"]]
        spreads.append(max(pcts) - min(pcts))
    mean_spread = sum(spreads) / len(spreads)
    # 0pt spread -> 100, 40pt spread -> 0, linear in between.
    score = 100 - (mean_spread * 2.5)
    return max(0, min(100, round(score)))


def _segment_overlap(breakdown: List[dict]) -> int:
    """Rough overlap proxy: average across segments of the smallest
    regional share. High when every region carries every segment;
    low when one region monopolizes a segment."""
    seg_ids = [seg.id for seg in TAXONOMY]
    mins = []
    for sid in seg_ids:
        vals = [b.get(sid, 0) for b in breakdown]
        mins.append(min(vals) if vals else 0)
    return round(sum(mins) / len(mins) * 6) if mins else 0


def _build_kpis(
    normalized: List[dict],
    segment_table: List[dict],
    breakdown: List[dict],
) -> dict:
    total = len(normalized)
    auto = sum(1 for r in normalized if r["status"] == "auto_approved")
    needs = total - auto
    return {
        "rows_normalized": total,
        "high_confidence_pct": _format_pct(auto, total),
        "needs_review_count": needs,
        "audience_consistency_score": _audience_consistency_score(segment_table),
        "total_reach": _estimate_reach(total),
        "segment_overlap": _segment_overlap(breakdown),
        "drift_alert": _build_drift_alert(segment_table),
    }


def _estimate_reach(rows: int) -> str:
    """Rough placeholder reach scaling with row count.
    Real implementation would join to per-row impression estimates."""
    millions = max(1.0, rows * 0.015)
    return f"{millions:.1f}M"


def _build_ai_qa(
    segment_table: List[dict],
    kpis: dict,
) -> List[dict]:
    drift = kpis["drift_alert"]["message"]
    needs = kpis["needs_review_count"]
    aligned = [s["segment_label"] for s in segment_table
               if s["consistency"] == "aligned"]
    diverging = [s["segment_label"] for s in segment_table
                 if s["consistency"] in ("diverging", "regional_only")]

    return [
        {
            "question": "Where is the largest audience drift?",
            "answer": drift,
        },
        {
            "question": "Which regions are most aligned?",
            "answer": (
                f"Across regions, the segments holding closest to a shared "
                f"mix are: {', '.join(aligned) if aligned else 'none'}. "
                f"The segments showing the most divergence are: "
                f"{', '.join(diverging) if diverging else 'none'}."
            ),
        },
        {
            "question": "What should the CMO do next?",
            "answer": (
                "Review the regions flagged in the drift alert before "
                "launch. Confirm whether the divergence reflects "
                "intentional localization or campaign drift, and align "
                "regional planners on the canonical taxonomy."
            ),
        },
        {
            "question": "How many rows need review?",
            "answer": (
                f"{needs} normalized rows fell below the auto-approval "
                f"threshold and are queued for regional planner review."
            ),
        },
    ]


def analyze(
    df: pd.DataFrame,
    normalizer: AudienceNormalizer,
    region_col: str = "region",
    audience_col: str = "target_audience",
    channel_col: Optional[str] = "channel",
) -> dict:
    """Run the normalizer over the input dataframe and return the full
    document the frontend consumes.

    Required columns: region_col, audience_col.
    Optional: channel_col (used for display in raw_inputs).
    """
    raws = df[audience_col].fillna("").astype(str).tolist()
    results = normalizer.normalize_batch(raws)

    raw_inputs: List[dict] = []
    normalized_outputs: List[dict] = []
    rows_by_region: Dict[str, List[dict]] = defaultdict(list)

    for i, ((_, src), res) in enumerate(zip(df.iterrows(), results)):
        rid = _row_id(i)
        region = str(src.get(region_col, "")).upper() or "UNKNOWN"
        channel = str(src.get(channel_col, "")) if channel_col else ""
        raw_text = str(src.get(audience_col, ""))

        raw_inputs.append({
            "row_id": rid,
            "region": region,
            "channel": channel,
            "raw_targeting_string": raw_text,
            "char_count": len(raw_text),
        })

        # Use top-1 even if below threshold; status reflects confidence band.
        if res.top_3:
            top_label = res.top_3[0][0]
            top_score = float(res.top_3[0][1])
        else:
            top_label = TAXONOMY[0].label
            top_score = 0.0
        out = {
            "row_id": rid,
            "canonical_id": label_to_id(top_label),
            "canonical_label": top_label,
            "confidence_score": round(top_score, 4),
            "status": _classify_status(top_score),
        }
        normalized_outputs.append(out)
        rows_by_region[region].append(out)

    breakdown = _build_regional_breakdown(rows_by_region)
    segment_table = _build_segment_table(breakdown)
    kpis = _build_kpis(normalized_outputs, segment_table, breakdown)
    ai_qa = _build_ai_qa(segment_table, kpis)

    return {
        "canonical_taxonomy": to_frontend_taxonomy(),
        "raw_inputs": raw_inputs,
        "normalized_outputs": normalized_outputs,
        "kpis": kpis,
        "regional_breakdown": breakdown,
        "segment_table": segment_table,
        "ai_qa_responses": ai_qa,
    }
