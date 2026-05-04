"""
Fine-tune the bi-encoder on labeled (raw_string, canonical_label) pairs.

Start zero-shot: the default model + anchor descriptions in taxonomy.py
will get you reasonable results immediately. Once regional planners have
confirmed enough flagged-for-review rows, run this script to fine-tune.

The proposal explicitly notes: "confirmed mappings are added to the
training pool for continuous improvement."

Loss: MultipleNegativesRankingLoss
  - Each (raw_string, anchor_for_correct_label) is a positive pair.
  - All other anchors in the batch act as in-batch negatives.
  - This is the standard recipe for fine-tuning bi-encoders for
    similarity / retrieval tasks.

Usage:
    python train.py --labeled data/labeled_train.csv \
                    --output models/audience-normalizer-v1 \
                    --epochs 3
"""

import argparse
import random
from pathlib import Path

import pandas as pd
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader

from taxonomy import TAXONOMY


def build_examples(df: pd.DataFrame,
                   raw_col: str,
                   label_col: str) -> list:
    """Pair each raw string with one of its canonical segment's anchors,
    sampled per-row. The bi-encoder learns to pull these embeddings
    together; in-batch negatives push other segments apart."""
    anchors_by_label = {seg.label: list(seg.anchors) for seg in TAXONOMY}
    examples = []
    for _, row in df.iterrows():
        raw = str(row[raw_col])
        label = row[label_col]
        if label not in anchors_by_label:
            continue
        anchor = random.choice(anchors_by_label[label])
        examples.append(InputExample(texts=[raw, anchor]))
    return examples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labeled", required=True)
    p.add_argument("--raw-col", default="target_audience")
    p.add_argument("--label-col", default="gold_label")
    p.add_argument(
        "--base-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    p.add_argument("--output", default="models/audience-normalizer-v1")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    df = pd.read_csv(args.labeled)
    examples = build_examples(df, args.raw_col, args.label_col)
    if not examples:
        raise SystemExit("No usable training examples after filtering.")
    print(f"Built {len(examples)} training pairs")

    model = SentenceTransformer(args.base_model)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(loader) * args.epochs),
        output_path=args.output,
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned model to {args.output}")
    print("Update normalizer.py DEFAULT_MODEL or pass model_name=... "
          "to AudienceNormalizer to load it.")


if __name__ == "__main__":
    main()
