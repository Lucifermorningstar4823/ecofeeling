"""
Phase 2 — Data Collection and Preprocessing
--------------------------------------------
Handles:
  - Text normalization (tokenization, stop-word removal, lemmatization)
  - Emoji standardization (Unicode → sentiment score)
  - Sticker dataset organization and label loading
"""

import re
import os
import unicodedata
import nltk
import emoji as emoji_lib
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first run
for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# ---------------------------------------------------------------------------
# Emoji sentiment lexicon (Unicode → polarity score in [-1, 1])
# Positive emojis score > 0, negative < 0, neutral ≈ 0
# ---------------------------------------------------------------------------
EMOJI_SENTIMENT_MAP = {
    "😀": 0.90, "😃": 0.88, "😄": 0.87, "😁": 0.85, "😆": 0.82,
    "😊": 0.86, "🥰": 0.92, "😍": 0.91, "🤩": 0.89, "😎": 0.80,
    "👍": 0.85, "❤️": 0.93, "💯": 0.88, "🎉": 0.87, "✅": 0.75,
    "😢": -0.80, "😭": -0.85, "😠": -0.88, "😡": -0.90, "👎": -0.85,
    "💔": -0.88, "😤": -0.75, "🤬": -0.92, "😖": -0.70, "😩": -0.72,
    "😐": 0.00, "😑": 0.00, "🤔": 0.05, "😶": 0.00, "🙂": 0.30,
}

STICKER_SENTIMENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}


class TextPreprocessor:
    """Normalize raw review text for downstream feature extraction."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def normalize(self, text: str) -> str:
        """
        Full normalization pipeline:
          1. Lowercase
          2. Remove URLs and HTML tags
          3. Tokenize
          4. Remove stop words
          5. Lemmatize
        Returns a single cleaned string.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        # Preserve emoji characters — strip only non-emoji punctuation
        text = re.sub(r"[^\w\s\U00010000-\U0010ffff]", " ", text, flags=re.UNICODE)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    def extract_emojis(self, text: str) -> list:
        """Return list of emoji characters found in text."""
        return [ch for ch in text if ch in emoji_lib.EMOJI_DATA]


class EmojiStandardizer:
    """Map emojis to numerical sentiment scores."""

    def __init__(self, lexicon: dict = None):
        self.lexicon = lexicon or EMOJI_SENTIMENT_MAP

    def score(self, emojis: list) -> float:
        """Aggregate sentiment score for a list of emojis. Returns 0.0 if empty."""
        if not emojis:
            return 0.0
        scores = [self.lexicon.get(e, 0.0) for e in emojis]
        return float(np.mean(scores))

    def to_dataframe(self) -> pd.DataFrame:
        """Expose the full lexicon as a DataFrame for inspection/saving."""
        rows = [{"emoji": k, "sentiment_score": v} for k, v in self.lexicon.items()]
        return pd.DataFrame(rows)


class StickerLoader:
    """
    Load PNG sticker dataset organized as:
        stickers/
            positive/  *.png
            negative/  *.png
            neutral/   *.png
    Returns (image_paths, labels) arrays.
    """

    CATEGORIES = ["positive", "negative", "neutral"]

    def __init__(self, sticker_dir: str):
        self.sticker_dir = Path(sticker_dir)

    def load(self):
        """
        Returns:
            paths  : list of Path objects to PNG files
            labels : list of int (1 = positive, -1 = negative, 0 = neutral)
        """
        paths, labels = [], []
        for cat in self.CATEGORIES:
            folder = self.sticker_dir / cat
            if not folder.exists():
                continue
            for png in sorted(folder.glob("*.png")):
                paths.append(png)
                labels.append(STICKER_SENTIMENT_MAP[cat])
        return paths, labels

    def class_distribution(self) -> dict:
        """Return count of stickers per class."""
        dist = {}
        for cat in self.CATEGORIES:
            folder = self.sticker_dir / cat
            dist[cat] = len(list(folder.glob("*.png"))) if folder.exists() else 0
        return dist
