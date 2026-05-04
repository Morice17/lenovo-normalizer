"""
Evaluate the AudienceNormalizer against a labeled set.

Per the proposal:
  - Multi-class classification, weighted F1
  - Per-class F1 tracked separately (rare segments must not collapse)
  - Production gate: minimum per-class F1 of 0.88

Usage:
    python evaluate.py --labeled data/labeled_eval.csv \
                       --raw-col target_audience \
                       --label-col gold_label

The labeled CSV needs two columns: the raw input string and the
gold-standard canonical label. Build this by sampling rows from each
regional CSV and having a human assign the correct label.
"""

import argparse
import sys

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from normalizer import AudienceNormalizer
from taxonomy import get_labels


PRODUCTION_GATE = 0.88


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labeled", required=True,
                   help="CSV with raw inputs and gold labels")
    p.add_argument("--raw-col", default="target_audience")
    p.add_argument("--label-col", default="gold_label")
    args = p.parse_args()

    df = pd.read_csv(args.labeled)
    if args.raw_col not in df.columns or args.label_col not in df.columns:
        print(f"ERROR: CSV must contain {args.raw_col} and {args.label_col}",
              file=sys.stderr)
        return 2

    valid_labels = set(get_labels())
    bad = set(df[args.label_col]) - valid_labels
    if bad:
        print(f"ERROR: gold labels not in taxonomy: {bad}", file=sys.stderr)
        return 2

    norm = AudienceNormalizer()
    results = norm.normalize_batch(df[args.raw_col].astype(str).tolist())
    # For evaluation we use the model's top pick even when below threshold
    # — needs_review is operational, not an evaluation concept.
    preds = [r.top_3[0][0] if r.top_3 else "UNKNOWN" for r in results]
    gold = df[args.label_col].tolist()

    print("\n=== Classification report ===")
    print(classification_report(gold, preds, digits=4, zero_division=0))

    weighted = f1_score(gold, preds, average="weighted", zero_division=0)
    print(f"Weighted F1: {weighted:.4f}")

    per_class = f1_score(
        gold, preds,
        labels=sorted(valid_labels),
        average=None,
        zero_division=0,
    )
    print("\n=== Per-class F1 (production gate = 0.88) ===")
    failed = []
    for label, score in zip(sorted(valid_labels), per_class):
        flag = "FAIL" if score < PRODUCTION_GATE else "pass"
        print(f"  [{flag}]  {label:<25}  F1 = {score:.4f}")
        if score < PRODUCTION_GATE:
            failed.append((label, score))

    if failed:
        print(f"\n{len(failed)} class(es) below the 0.88 gate. "
              f"Add training data and rerun, or expand anchors in taxonomy.py.")
        return 1
    print("\nAll classes pass the production gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
