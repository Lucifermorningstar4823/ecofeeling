"""
Echo Feeling Engine
====================
Orchestrates the full six-phase multimodal sentiment analysis pipeline:

  Phase 1 : E-commerce website as data source (review ingestion)
  Phase 2 : Data collection and preprocessing
  Phase 3 : Feature extraction (BoW, TF-IDF, BERT, emoji, sticker)
  Phase 4 : Multimodal fusion
  Phase 5 : Model training and evaluation
  Phase 6 : Deployment as Node.js-compatible module (REST API)

Usage (inference on a single review):
    engine = EchoFeelingEngine()
    engine.load_model("models/echo_feeling_nn.joblib")
    result = engine.predict_review("Great product! 😊👍")
    # → {"sentiment": "positive", "confidence": 0.91, "emoji_score": 0.875}

Usage (training a new model):
    engine = EchoFeelingEngine()
    engine.train(
        reviews=["...", ...],
        labels=[2, 0, 1, ...],
        emoji_lists=[["😊"], [], ["😠"], ...],
        sticker_paths=[...],
        sticker_labels=[1, -1, 0, ...],
        model_type="neural_network",
    )
    engine.save_model("models/echo_feeling_nn.joblib")
"""

import os
import numpy as np
import joblib
from pathlib import Path
from typing import List, Optional

from .preprocessor import TextPreprocessor, EmojiStandardizer
from .features import (
    BagOfWordsExtractor,
    TFIDFExtractor,
    BERTExtractor,
    EmojiFeatureExtractor,
    StickerFeatureExtractor,
)
from .fusion import MultimodalFusion
from .model import SentimentClassifier, LABEL_NAMES

# Sentiment label → integer index
SENTIMENT_INDEX = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "suspicious": 3,
}


class EchoFeelingEngine:
    """
    End-to-end multimodal sentiment analysis engine.

    Parameters
    ----------
    bow_features    : int   Max BoW vocabulary size (default 3000)
    tfidf_features  : int   Max TF-IDF vocabulary size (default 3000)
    use_bert        : bool  Whether to include BERT embeddings (slower but better)
    model_type      : str   'neural_network' | 'svm' | 'random_forest'
    """

    def __init__(
        self,
        bow_features: int = 3000,
        tfidf_features: int = 3000,
        use_bert: bool = False,
        model_type: str = "neural_network",
    ):
        # Phase 2 — preprocessing components
        self.text_preprocessor = TextPreprocessor()
        self.emoji_standardizer = EmojiStandardizer()

        # Phase 3 — feature extractors
        self.bow_extractor = BagOfWordsExtractor(max_features=bow_features)
        self.tfidf_extractor = TFIDFExtractor(max_features=tfidf_features)
        self.bert_extractor = BERTExtractor() if use_bert else None
        self.emoji_extractor = EmojiFeatureExtractor(self.emoji_standardizer)
        self.sticker_extractor = StickerFeatureExtractor()

        # Phase 4 — fusion
        self.fuser = MultimodalFusion()

        # Phase 5 — classifier
        self.classifier = SentimentClassifier(model_type=model_type)

        self._fitted = False
        self._use_bert = use_bert

    # ------------------------------------------------------------------
    # Phase 2 helpers
    # ------------------------------------------------------------------
    def _preprocess_texts(self, reviews: List[str]):
        """Normalize text and extract emoji lists simultaneously."""
        clean_texts, emoji_lists = [], []
        for review in reviews:
            emojis = self.text_preprocessor.extract_emojis(review)
            cleaned = self.text_preprocessor.normalize(review)
            clean_texts.append(cleaned)
            emoji_lists.append(emojis)
        return clean_texts, emoji_lists

    # ------------------------------------------------------------------
    # Phase 3 helpers
    # ------------------------------------------------------------------
    def _extract_text_features(self, clean_texts: List[str], fit: bool = False):
        if fit:
            bow = self.bow_extractor.fit_transform(clean_texts)
            tfidf = self.tfidf_extractor.fit_transform(clean_texts)
        else:
            bow = self.bow_extractor.transform(clean_texts)
            tfidf = self.tfidf_extractor.transform(clean_texts)

        bert = self.bert_extractor.transform(clean_texts) if self._use_bert else None
        return bow, tfidf, bert

    # ------------------------------------------------------------------
    # Phase 5 — Train
    # ------------------------------------------------------------------
    def train(
        self,
        reviews: List[str],
        labels: List[int],
        emoji_lists: Optional[List[list]] = None,
        sticker_paths: Optional[list] = None,
        sticker_labels: Optional[List[int]] = None,
        model_type: str = "neural_network",
    ) -> dict:
        """
        Full training pipeline.

        Parameters
        ----------
        reviews       : Raw review strings (may contain emojis)
        labels        : Integer sentiment labels (0–3)
        emoji_lists   : Pre-extracted emoji lists (optional; extracted if None)
        sticker_paths : List of Path objects to PNG stickers
        sticker_labels: Integer labels for stickers
        model_type    : Classifier to use

        Returns
        -------
        metrics dict with accuracy and F1 on a held-out 20% split
        """
        self.classifier = SentimentClassifier(model_type=model_type)

        # Phase 2
        clean_texts, auto_emoji_lists = self._preprocess_texts(reviews)
        if emoji_lists is None:
            emoji_lists = auto_emoji_lists

        # Phase 3 — text + emoji features
        bow, tfidf, bert = self._extract_text_features(clean_texts, fit=True)
        emoji_feats = self.emoji_extractor.transform(emoji_lists)

        # Phase 3 — sticker features (optional)
        sticker_feats = None
        if sticker_paths and sticker_labels:
            sticker_feats, _ = self.sticker_extractor.fit_transform(
                sticker_paths, sticker_labels
            )
            # Sticker set may differ in length from reviews; use mean pooling to inject
            sticker_mean = np.tile(sticker_feats.mean(axis=0), (len(reviews), 1))
        else:
            sticker_mean = None

        # Phase 4 — fusion
        X = self.fuser.fuse(
            bow=bow,
            tfidf=tfidf,
            bert=bert,
            emoji=emoji_feats,
            sticker=sticker_mean,
        )
        y = np.array(labels)

        # 80/20 train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Phase 5 — fit
        self.classifier.fit(X_train, y_train)
        metrics = self.classifier.evaluate(X_test, y_test)
        cv_metrics = self.classifier.cross_validate(X, y, k=5)
        self._fitted = True

        print(f"\n[EchoFeelingEngine] Training complete ({model_type})")
        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  F1 Macro : {metrics['f1_macro']}")
        print(f"  CV F1    : {cv_metrics['cv_f1_mean']} ± {cv_metrics['cv_f1_std']}")
        return {**metrics, **cv_metrics}

    # ------------------------------------------------------------------
    # Phase 6 — Inference (single review)
    # ------------------------------------------------------------------
    def predict_review(self, review: str) -> dict:
        """
        Predict sentiment for a single raw review string.

        Returns
        -------
        {
          "sentiment"   : "positive" | "negative" | "neutral" | "suspicious",
          "label_index" : int,
          "confidence"  : float,
          "emoji_score" : float,
          "emoji_found" : [str, ...]
        }
        """
        if not self._fitted:
            raise RuntimeError("Engine not fitted. Call train() or load_model() first.")

        emojis = self.text_preprocessor.extract_emojis(review)
        cleaned = self.text_preprocessor.normalize(review)

        bow = self.bow_extractor.transform([cleaned])
        tfidf = self.tfidf_extractor.transform([cleaned])
        bert = self.bert_extractor.transform([cleaned]) if self._use_bert else None
        emoji_feats = self.emoji_extractor.transform([emojis])

        X = self.fuser.fuse(bow=bow, tfidf=tfidf, bert=bert, emoji=emoji_feats)
        label_idx = int(self.classifier.predict(X)[0])
        proba = self.classifier.predict_proba(X)[0]

        return {
            "sentiment": LABEL_NAMES[label_idx] if label_idx < len(LABEL_NAMES) else "unknown",
            "label_index": label_idx,
            "confidence": round(float(proba.max()), 4),
            "emoji_score": round(float(self.emoji_standardizer.score(emojis)), 4),
            "emoji_found": emojis,
        }

    def predict_batch(self, reviews: List[str]) -> List[dict]:
        """Predict sentiment for a list of reviews."""
        return [self.predict_review(r) for r in reviews]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, path: str):
        """Save the fitted engine to a single joblib file."""
        state = {
            "classifier": self.classifier,
            "bow_vectorizer": self.bow_extractor.vectorizer,
            "tfidf_vectorizer": self.tfidf_extractor.vectorizer,
            "sticker_encoder": self.sticker_extractor.label_encoder,
            "use_bert": self._use_bert,
        }
        os.makedirs(Path(path).parent, exist_ok=True)
        joblib.dump(state, path)
        print(f"[EchoFeelingEngine] Saved to {path}")

    def load_model(self, path: str):
        """Restore a previously saved engine."""
        state = joblib.load(path)
        self.classifier = state["classifier"]
        self.bow_extractor.vectorizer = state["bow_vectorizer"]
        self.tfidf_extractor.vectorizer = state["tfidf_vectorizer"]
        self.sticker_extractor.label_encoder = state["sticker_encoder"]
        self._use_bert = state.get("use_bert", False)
        self._fitted = True
        print(f"[EchoFeelingEngine] Loaded from {path}")
        return self
