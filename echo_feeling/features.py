"""
Phase 3 — Feature Extraction
------------------------------
Distinct extraction strategies per modality:
  - Text  : Bag-of-Words, TF-IDF, BERT contextual embeddings
  - Emoji : Sentiment score from lexicon dataframe
  - Sticker: Encoded categorical labels (fine-tuned weights placeholder)
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from PIL import Image


# ---------------------------------------------------------------------------
# Text feature extractors
# ---------------------------------------------------------------------------

class BagOfWordsExtractor:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = CountVectorizer(max_features=max_features)

    def fit(self, texts: list):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: list) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).toarray()


class TFIDFExtractor:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)

    def fit(self, texts: list):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: list) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).toarray()


class BERTExtractor:
    """
    Extract 768-d [CLS] embeddings from bert-base-uncased.
    Falls back to a zero vector if BERT unavailable (CPU-friendly).
    """

    MODEL_NAME = "bert-base-uncased"
    EMBEDDING_DIM = 768

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = BertModel.from_pretrained(self.MODEL_NAME)
            self.model.eval()
            self.model.to(self.device)
            self.available = True
        except Exception as e:
            print(f"[BERTExtractor] BERT unavailable: {e}. Using zero vectors.")
            self.available = False

    def transform(self, texts: list, batch_size: int = 16) -> np.ndarray:
        if not self.available:
            return np.zeros((len(texts), self.EMBEDDING_DIM))

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
            # Use [CLS] token representation
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
        return np.vstack(embeddings)


# ---------------------------------------------------------------------------
# Emoji feature extractor
# ---------------------------------------------------------------------------

class EmojiFeatureExtractor:
    """
    Converts per-review emoji lists into a single numeric sentiment score
    and packs it as a (N, 1) array for fusion.
    """

    def __init__(self, standardizer):
        self.standardizer = standardizer  # EmojiStandardizer instance

    def transform(self, emoji_lists: list) -> np.ndarray:
        scores = [self.standardizer.score(e) for e in emoji_lists]
        return np.array(scores, dtype=np.float32).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Sticker feature extractor
# ---------------------------------------------------------------------------

class StickerFeatureExtractor:
    """
    Loads PNG stickers, resizes to a fixed size, converts to grayscale
    mean intensity, and encodes the categorical label as a weight vector.
    Label encoding: positive=2, neutral=1, negative=0 (ordinal).
    """

    TARGET_SIZE = (64, 64)

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def _image_to_feature(self, path: Path) -> np.ndarray:
        """Flatten a grayscale mean-intensity feature from PNG."""
        try:
            img = Image.open(path).convert("L").resize(self.TARGET_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            return np.array([arr.mean(), arr.std()])
        except Exception:
            return np.zeros(2)

    def fit_transform(self, paths: list, labels: list):
        """
        Returns:
            image_features : (N, 2) pixel stats array
            encoded_labels : (N,)   int array
        """
        self.label_encoder.fit(labels)
        encoded = self.label_encoder.transform(labels)
        features = np.vstack([self._image_to_feature(p) for p in paths])
        return features, encoded

    def transform(self, paths: list) -> np.ndarray:
        return np.vstack([self._image_to_feature(p) for p in paths])
