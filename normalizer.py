"""
AudienceNormalizer: bi-encoder sentence transformer that maps raw targeting
strings (any of the three regional formats) to canonical segment labels.

Approach:
  1. At init, embed every anchor for every canonical segment.
  2. For each incoming raw string, embed it once.
  3. Compute cosine similarity against every anchor.
  4. For each segment, take the max similarity across its anchors.
  5. Argmax across segments = predicted label.
  6. If max similarity < CONFIDENCE_THRESHOLD, flag for human review.

Why a multilingual base model: the LATAM file contains Spanish-language
interest strings ("juegos de acción", "videojuegos"), and the proposal
explicitly calls out tolerating those without separate per-region models.

Default model: paraphrase-multilingual-MiniLM-L12-v2
  - 50+ languages, including Spanish and English
  - 384-dim embeddings, fast on CPU
  - Strong on short-text and semantic-similarity tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from taxonomy import (
    TAXONOMY,
    CONFIDENCE_THRESHOLD,
    get_labels,
)


@dataclass
class NormalizationResult:
    raw_input: str
    predicted_label: Optional[str]   # None when below threshold
    confidence: float                # cosine similarity, 0..1 (typically)
    needs_review: bool
    top_3: List[tuple]               # [(label, score), ...] for transparency

    def to_dict(self) -> dict:
        return {
            "raw_input": self.raw_input,
            "predicted_label": self.predicted_label,
            "confidence": round(float(self.confidence), 4),
            "needs_review": self.needs_review,
            "top_3": [(lbl, round(float(s), 4)) for lbl, s in self.top_3],
        }


class AudienceNormalizer:
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.labels: List[str] = get_labels()

        # Build flat list of (label, anchor) so we can vectorize the
        # similarity computation, then groupby label for max.
        self._flat_labels: List[str] = []
        self._flat_anchors: List[str] = []
        for seg in TAXONOMY:
            for anchor in seg.anchors:
                self._flat_labels.append(seg.label)
                self._flat_anchors.append(anchor)

        self._anchor_embeddings: np.ndarray = self.model.encode(
            self._flat_anchors,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # label_index[i] = index of self._flat_labels[i] in self.labels
        self._label_index = np.array(
            [self.labels.index(lbl) for lbl in self._flat_labels]
        )

    def normalize(self, raw: str) -> NormalizationResult:
        if not isinstance(raw, str) or not raw.strip():
            return NormalizationResult(
                raw_input=raw,
                predicted_label=None,
                confidence=0.0,
                needs_review=True,
                top_3=[],
            )

        emb = self.model.encode(
            [raw], normalize_embeddings=True, show_progress_bar=False
        )[0]
        # Normalized vectors -> dot product is cosine similarity.
        sims = self._anchor_embeddings @ emb  # shape: (n_anchors,)

        # Max similarity per segment.
        per_segment_max = np.full(len(self.labels), -np.inf)
        for i, lbl_idx in enumerate(self._label_index):
            if sims[i] > per_segment_max[lbl_idx]:
                per_segment_max[lbl_idx] = sims[i]

        order = np.argsort(per_segment_max)[::-1]
        top_3 = [(self.labels[i], per_segment_max[i]) for i in order[:3]]

        best_idx = int(order[0])
        best_score = float(per_segment_max[best_idx])
        best_label = self.labels[best_idx]

        below_threshold = best_score < self.threshold
        return NormalizationResult(
            raw_input=raw,
            predicted_label=None if below_threshold else best_label,
            confidence=best_score,
            needs_review=below_threshold,
            top_3=top_3,
        )

    def normalize_batch(self, raws: List[str]) -> List[NormalizationResult]:
        # Fast path: single encoding call for the whole batch.
        cleaned = [r if isinstance(r, str) and r.strip() else "" for r in raws]
        if not any(cleaned):
            return [self.normalize(r) for r in raws]

        embs = self.model.encode(
            cleaned, normalize_embeddings=True, show_progress_bar=False
        )
        sims_matrix = embs @ self._anchor_embeddings.T  # (batch, n_anchors)

        results: List[NormalizationResult] = []
        for raw, sims in zip(raws, sims_matrix):
            if not isinstance(raw, str) or not raw.strip():
                results.append(
                    NormalizationResult(raw, None, 0.0, True, [])
                )
                continue

            per_segment_max = np.full(len(self.labels), -np.inf)
            for i, lbl_idx in enumerate(self._label_index):
                if sims[i] > per_segment_max[lbl_idx]:
                    per_segment_max[lbl_idx] = sims[i]

            order = np.argsort(per_segment_max)[::-1]
            top_3 = [
                (self.labels[i], per_segment_max[i]) for i in order[:3]
            ]
            best_idx = int(order[0])
            best_score = float(per_segment_max[best_idx])
            below = best_score < self.threshold
            results.append(
                NormalizationResult(
                    raw_input=raw,
                    predicted_label=None if below else self.labels[best_idx],
                    confidence=best_score,
                    needs_review=below,
                    top_3=top_3,
                )
            )
        return results
