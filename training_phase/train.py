"""
Training Phase — Echo Feeling
==============================
Trains and evaluates all three classifiers described in the paper:
  - SVM           (F1 ≈ 0.85)
  - Random Forest  (F1 ≈ 0.83)
  - Neural Network (F1 ≈ 0.89)  ← saved as default model

Expects one of the following data sources (in order of priority):
  1. CSV at DATA_PATH with columns: review, label
     Labels: 0=negative, 1=neutral, 2=positive, 3=suspicious
  2. A Kaggle sentiment dataset placed at DATA_PATH
  3. Built-in synthetic demo dataset (for smoke-testing)

Stickers:
  Place PNG stickers under stickers/positive|negative|neutral/
  relative to this script, or set STICKER_DIR env variable.

Usage:
  python training_phase/train.py
  python training_phase/train.py --data path/to/data.csv --sticker path/to/stickers
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from echo_feeling.preprocessor import TextPreprocessor, EmojiStandardizer, StickerLoader
from echo_feeling.features import (
    BagOfWordsExtractor, TFIDFExtractor, BERTExtractor,
    EmojiFeatureExtractor, StickerFeatureExtractor,
)
from echo_feeling.fusion import MultimodalFusion
from echo_feeling.model import SentimentClassifier, compare_models, LABEL_NAMES

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent
DATA_PATH    = os.environ.get("ECHO_DATA_PATH", REPO_ROOT / "data" / "reviews.csv")
STICKER_DIR  = os.environ.get("ECHO_STICKER_DIR", REPO_ROOT / "data" / "stickers")
MODELS_DIR   = REPO_ROOT / "models"
REPORTS_DIR  = REPO_ROOT / "reports"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path):
    """Load CSV with columns: review, label (int 0-3)."""
    path = Path(data_path)
    if path.exists():
        df = pd.read_csv(path)
        print(f"[Data] Loaded {len(df)} samples from {path}")
        return df["review"].tolist(), df["label"].tolist()
    else:
        print(f"[Data] {path} not found — using synthetic demo dataset.")
        return _synthetic_dataset()


def _synthetic_dataset():
    """
    Minimal synthetic dataset for smoke-testing the pipeline.
    Replace with a real Kaggle dataset for production use.
    Distribution mirrors paper: ~2700 positive, 250 negative, 150 neutral.
    """
    rng = np.random.default_rng(42)
    positive_reviews = [
        "This product is absolutely amazing! 😊", "Loved it, will buy again 👍",
        "Fantastic quality, highly recommend", "Great value for money ❤️",
        "Super fast delivery, perfect packaging", "Excellent customer service 😍",
        "Works exactly as described, very happy", "Best purchase I made this year 🎉",
    ]
    negative_reviews = [
        "Terrible quality, broke after one use 😠", "Complete waste of money 👎",
        "Arrived damaged, very disappointed 😢", "Does not work as advertised",
        "Horrible experience, avoid this seller", "Worst product ever 😡",
    ]
    neutral_reviews = [
        "It's okay, nothing special", "Average product for the price",
        "Arrived on time, product is decent", "Not bad but not great either",
    ]
    suspicious_reviews = [
        "Buy now best product 100% real fake discount spam",
        "Paid review this is promotional advertisement",
    ]

    reviews, labels = [], []
    # Replicate to reach realistic class sizes (~3100 total)
    for r in positive_reviews * 338: reviews.append(r); labels.append(2)
    for r in negative_reviews * 42:  reviews.append(r); labels.append(0)
    for r in neutral_reviews  * 38:  reviews.append(r); labels.append(1)
    for r in suspicious_reviews * 10: reviews.append(r); labels.append(3)

    # Shuffle
    idx = rng.permutation(len(reviews))
    return [reviews[i] for i in idx], [labels[i] for i in idx]


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(labels, save_path):
    counts = pd.Series(labels).map(
        {0: "negative", 1: "neutral", 2: "positive", 3: "suspicious"}
    ).value_counts()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", color=["#e74c3c", "#95a5a6", "#2ecc71", "#e67e22"])
    plt.title("Sentiment Distribution across Dataset")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Sentiment distribution saved → {save_path}")


def plot_confusion_matrix(cm, model_name, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


def plot_model_comparison(results: dict, save_path):
    models = list(results.keys())
    f1s = [results[m]["f1_macro"] for m in models]
    accs = [results[m]["accuracy"] for m in models]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, f1s, width, label="F1 Macro", color="#3498db")
    ax.bar(x + width / 2, accs, width, label="Accuracy", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(["SVM", "Random Forest", "Neural Network"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison — Multimodal Sentiment Analysis")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Model comparison saved → {save_path}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training(data_path=DATA_PATH, sticker_dir=STICKER_DIR, use_bert=False):
    print("=" * 60)
    print("  Echo Feeling — Training Pipeline (SAMANWAYA'26)")
    print("=" * 60)

    # ── Phase 2: Load & preprocess ──────────────────────────────────
    print("\n[Phase 2] Preprocessing...")
    reviews, labels = load_data(data_path)
    tp = TextPreprocessor()
    es = EmojiStandardizer()

    clean_texts, emoji_lists = [], []
    for rev in reviews:
        emojis = tp.extract_emojis(rev)
        clean = tp.normalize(rev)
        clean_texts.append(clean)
        emoji_lists.append(emojis)

    plot_sentiment_distribution(labels, REPORTS_DIR / "sentiment_distribution.png")

    # ── Phase 3: Feature extraction ─────────────────────────────────
    print("\n[Phase 3] Extracting features...")
    bow_ext   = BagOfWordsExtractor(max_features=3000)
    tfidf_ext = TFIDFExtractor(max_features=3000)
    emoji_ext = EmojiFeatureExtractor(es)

    bow   = bow_ext.fit_transform(clean_texts)
    tfidf = tfidf_ext.fit_transform(clean_texts)
    emoji_feats = emoji_ext.transform(emoji_lists)

    bert = None
    if use_bert:
        print("[Phase 3] Encoding BERT embeddings (this may take a while)...")
        bert_ext = BERTExtractor()
        bert = bert_ext.transform(clean_texts, batch_size=32)

    # Sticker features
    sticker_mean = None
    sticker_loader = StickerLoader(sticker_dir)
    sticker_paths, sticker_labels = sticker_loader.load()
    if sticker_paths:
        dist = sticker_loader.class_distribution()
        print(f"[Phase 3] Sticker dataset — {dist}")
        stk_ext = StickerFeatureExtractor()
        stk_feats, _ = stk_ext.fit_transform(sticker_paths, sticker_labels)
        sticker_mean = np.tile(stk_feats.mean(axis=0), (len(reviews), 1))
    else:
        print("[Phase 3] No sticker dataset found — skipping sticker modality.")

    # ── Phase 4: Fusion ─────────────────────────────────────────────
    print("\n[Phase 4] Fusing modalities...")
    fuser = MultimodalFusion()
    X = fuser.fuse(
        bow=bow, tfidf=tfidf, bert=bert,
        emoji=emoji_feats, sticker=sticker_mean,
    )
    y = np.array(labels)
    desc = fuser.describe(X)
    print(f"  Fused shape: {X.shape}  mean={desc['mean']:.4f}  std={desc['std']:.4f}")

    # ── Phase 5: Train & evaluate all models ────────────────────────
    print("\n[Phase 5] Training and evaluating models...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    comparison = compare_models(X_train, y_train, X_test, y_test)
    plot_model_comparison(comparison, REPORTS_DIR / "model_comparison.png")

    # Save confusion matrices
    for model_name in ["svm", "random_forest", "neural_network"]:
        clf = SentimentClassifier(model_type=model_name)
        clf.fit(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test)
        import numpy as np
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm, model_name,
            REPORTS_DIR / f"confusion_matrix_{model_name}.png",
        )

    # ── Save best model (Neural Network) ────────────────────────────
    print("\n[Phase 6] Saving best model (neural_network)...")
    best_clf = SentimentClassifier(model_type="neural_network")
    best_clf.fit(X_train, y_train)
    best_metrics = best_clf.evaluate(X_test, y_test)

    model_save_path = MODELS_DIR / "echo_feeling_nn.joblib"
    import joblib
    state = {
        "classifier": best_clf,
        "bow_vectorizer": bow_ext.vectorizer,
        "tfidf_vectorizer": tfidf_ext.vectorizer,
        "sticker_encoder": StickerFeatureExtractor().label_encoder,
        "use_bert": use_bert,
    }
    joblib.dump(state, model_save_path)
    print(f"[Saved] {model_save_path}")

    # Save metrics report
    report = {
        "best_model": "neural_network",
        "metrics": {
            "accuracy": best_metrics["accuracy"],
            "f1_macro": best_metrics["f1_macro"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
        },
        "model_comparison": comparison,
        "dataset_size": len(reviews),
        "feature_dim": X.shape[1],
    }
    with open(REPORTS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n[Done] Training complete.")
    print(f"  Best model — Accuracy: {best_metrics['accuracy']}  F1: {best_metrics['f1_macro']}")
    print(f"  Reports saved in: {REPORTS_DIR}")
    return report


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Echo Feeling sentiment model")
    parser.add_argument("--data",    default=DATA_PATH,   help="Path to CSV (review, label)")
    parser.add_argument("--sticker", default=STICKER_DIR, help="Path to sticker folder")
    parser.add_argument("--bert",    action="store_true",  help="Include BERT embeddings")
    args = parser.parse_args()
    run_training(data_path=args.data, sticker_dir=args.sticker, use_bert=args.bert)
