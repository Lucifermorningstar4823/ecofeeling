"""
Phase 6 — Flask REST API (Node.js-compatible module interface)
---------------------------------------------------------------
Endpoints:
  POST /analyze                 — text-only sentiment (backward-compatible)
  POST /analyze/multimodal      — text + emoji + sticker sentiment
  GET  /health                  — liveness probe
  GET  /model/info              — model metadata

Request body (JSON):
  {
    "text"    : "Great product! 😊",       # required
    "emojis"  : ["😊", "👍"],               # optional override
    "sticker" : "positive"                 # optional: "positive"|"negative"|"neutral"
  }

Response body (JSON):
  {
    "sentiment"   : "positive",
    "label_index" : 2,
    "confidence"  : 0.91,
    "emoji_score" : 0.875,
    "emoji_found" : ["😊"],
    "model"       : "neural_network"
  }
"""

import os
from flask import Flask, request, jsonify
from .engine import EchoFeelingEngine

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Engine initialisation — load persisted model if available, otherwise use a
# lightweight rule-based fallback so the API stays responsive at all times.
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("ECHO_MODEL_PATH", "models/echo_feeling_nn.joblib")
_engine: EchoFeelingEngine = None


def _get_engine() -> EchoFeelingEngine:
    global _engine
    if _engine is None:
        _engine = EchoFeelingEngine(use_bert=False, model_type="neural_network")
        if os.path.exists(MODEL_PATH):
            _engine.load_model(MODEL_PATH)
        else:
            # Fallback: rule-based engine using emoji scores only
            print("[API] No trained model found — using rule-based fallback.")
    return _engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)})


@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "name": "Echo Feeling",
        "version": "2.0.0",
        "modalities": ["text", "emoji", "sticker"],
        "classes": ["negative", "neutral", "positive", "suspicious"],
        "paper": "AI-Powered Review Mining for Subjective Text Understanding (SAMANWAYA'26)",
    })


@app.route("/analyze", methods=["POST"])
def analyze_text():
    """
    Text-only sentiment analysis (backward-compatible with v1 API).

    Body: { "text": "..." }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    engine = _get_engine()

    if engine._fitted:
        result = engine.predict_review(text)
    else:
        result = _rule_based_fallback(text)

    result["model"] = engine.classifier.model_type if engine._fitted else "rule_based"
    return jsonify(result)


@app.route("/analyze/multimodal", methods=["POST"])
def analyze_multimodal_route():
    """
    Full multimodal analysis: text + emoji + sticker sentiment.

    Body: {
        "text"   : "...",
        "emojis" : ["😊"],          # optional
        "sticker": "positive"       # optional
    }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    engine = _get_engine()

    if engine._fitted:
        result = engine.predict_review(text)
    else:
        result = _rule_based_fallback(text)

    # Incorporate any explicitly supplied emojis
    extra_emojis = data.get("emojis", [])
    if extra_emojis:
        extra_score = engine.emoji_standardizer.score(extra_emojis)
        result["emoji_score"] = round((result["emoji_score"] + extra_score) / 2, 4)
        result["emoji_found"] = list(set(result.get("emoji_found", []) + extra_emojis))

    # Sticker modality override
    sticker = data.get("sticker")
    if sticker in ("positive", "negative", "neutral"):
        sticker_score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        result["sticker_sentiment"] = sticker
        result["sticker_score"] = sticker_score_map[sticker]

    result["model"] = engine.classifier.model_type if engine._fitted else "rule_based"
    return jsonify(result)


@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    """
    Batch inference for multiple reviews at once.

    Body: { "reviews": ["text1", "text2", ...] }
    """
    data = request.get_json(silent=True) or {}
    reviews = data.get("reviews", [])
    if not reviews or not isinstance(reviews, list):
        return jsonify({"error": "Field 'reviews' must be a non-empty list."}), 400

    engine = _get_engine()
    if engine._fitted:
        results = engine.predict_batch(reviews)
    else:
        results = [_rule_based_fallback(r) for r in reviews]

    return jsonify({"results": results, "count": len(results)})


# ---------------------------------------------------------------------------
# Rule-based fallback (no trained model required)
# ---------------------------------------------------------------------------

POSITIVE_KEYWORDS = {
    "good", "great", "excellent", "amazing", "love", "best", "fantastic",
    "awesome", "perfect", "happy", "satisfied", "recommend", "wonderful",
    "superb", "brilliant", "nice", "pleasant",
}
NEGATIVE_KEYWORDS = {
    "bad", "terrible", "awful", "hate", "worst", "horrible", "poor",
    "disappointed", "broken", "useless", "waste", "defective", "damaged",
    "refund", "return", "scam", "fake", "fraud",
}
SUSPICIOUS_KEYWORDS = {
    "spam", "fake", "bot", "advertis", "paid", "promotional", "not real",
}


def _rule_based_fallback(text: str) -> dict:
    from .preprocessor import EmojiStandardizer, TextPreprocessor
    tp = TextPreprocessor()
    es = EmojiStandardizer()
    emojis = tp.extract_emojis(text)
    emoji_score = es.score(emojis)
    words = set(text.lower().split())

    if words & SUSPICIOUS_KEYWORDS:
        sentiment, idx, conf = "suspicious", 3, 0.70
    elif len(words & NEGATIVE_KEYWORDS) > len(words & POSITIVE_KEYWORDS):
        sentiment, idx, conf = "negative", 0, 0.72
    elif words & POSITIVE_KEYWORDS or emoji_score > 0.3:
        sentiment, idx, conf = "positive", 2, 0.68
    else:
        sentiment, idx, conf = "neutral", 1, 0.55

    return {
        "sentiment": sentiment,
        "label_index": idx,
        "confidence": conf,
        "emoji_score": round(emoji_score, 4),
        "emoji_found": emojis,
    }


# ---------------------------------------------------------------------------
# Public convenience functions (imported by __init__.py)
# ---------------------------------------------------------------------------

def analyze(text: str) -> str:
    """Minimal text-only API — returns sentiment label string."""
    result = _rule_based_fallback(text)
    return result["sentiment"]


def analyze_multimodal(text: str, emojis: list = None, sticker: str = None) -> dict:
    """Full multimodal analysis — returns result dict."""
    engine = _get_engine()
    if engine._fitted:
        result = engine.predict_review(text)
    else:
        result = _rule_based_fallback(text)
    if emojis:
        extra = engine.emoji_standardizer.score(emojis)
        result["emoji_score"] = round((result["emoji_score"] + extra) / 2, 4)
    if sticker in ("positive", "negative", "neutral"):
        result["sticker_sentiment"] = sticker
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
