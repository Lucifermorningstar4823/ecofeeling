"""
Phase 4 — Multimodal Fusion
----------------------------
Concatenates normalized feature vectors from:
  - Text  : BoW + TF-IDF + BERT embeddings
  - Emoji : sentiment score(s)
  - Sticker: encoded label weights

All modalities are L2-normalized before concatenation to prevent any
single modality from dominating the unified representation.
"""

import numpy as np
from sklearn.preprocessing import normalize


class MultimodalFusion:
    """
    Fuses heterogeneous feature arrays into one consolidated vector per sample.

    Usage:
        fuser = MultimodalFusion()
        X = fuser.fuse(bow=bow_feats, tfidf=tfidf_feats,
                       bert=bert_feats, emoji=emoji_feats,
                       sticker=sticker_feats)
    """

    def fuse(
        self,
        bow: np.ndarray = None,
        tfidf: np.ndarray = None,
        bert: np.ndarray = None,
        emoji: np.ndarray = None,
        sticker: np.ndarray = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        bow     : (N, bow_dim)     Bag-of-Words features
        tfidf   : (N, tfidf_dim)   TF-IDF features
        bert    : (N, 768)         BERT [CLS] embeddings
        emoji   : (N, 1)           Aggregated emoji sentiment score
        sticker : (N, sticker_dim) Sticker image/label features

        Returns
        -------
        fused : (N, total_dim) Normalized concatenated feature matrix
        """
        parts = []

        if bow is not None:
            parts.append(self._norm(bow))
        if tfidf is not None:
            parts.append(self._norm(tfidf))
        if bert is not None:
            parts.append(self._norm(bert))
        if emoji is not None:
            # Emoji scores already in [-1, 1]; just pass through
            parts.append(np.array(emoji, dtype=np.float32).reshape(len(emoji), -1))
        if sticker is not None:
            parts.append(self._norm(sticker))

        if not parts:
            raise ValueError("At least one modality must be provided for fusion.")

        return np.hstack(parts)

    @staticmethod
    def _norm(X: np.ndarray) -> np.ndarray:
        """L2-normalize rows; handle zero vectors safely."""
        X = np.array(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def describe(self, fused: np.ndarray) -> dict:
        return {
            "n_samples": fused.shape[0],
            "n_features": fused.shape[1],
            "mean": float(fused.mean()),
            "std": float(fused.std()),
        }
