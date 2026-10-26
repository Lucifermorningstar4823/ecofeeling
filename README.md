# Echo Feeling

![PyPI](https://img.shields.io/pypi/v/echo-feeling)
![Python](https://img.shields.io/pypi/pyversions/echo-feeling)
![License](https://img.shields.io/github/license/Becadyop/final)

**Echo Feeling** is a multimodal sentiment analysis library for e-commerce review mining.  
It integrates **text**, **emoji**, and **sticker** modalities through a six-phase machine learning pipeline to deliver robust customer sentiment insights.

> Paper: *AI-Powered Review Mining for Subjective Text Understanding*  
> SAMANWAYA'26 — College of Engineering Karunagappally  
> Authors: Swathi Lekshmi SS, Romin Varghese, R Arun, Ivin Issac  
> Supervisor: Hridhya J

---

## Features

- **Multimodal fusion** — text (BoW + TF-IDF + BERT) + emoji sentiment scores + sticker visual cues
- **Three classifiers** — SVM (F1 0.85), Random Forest (F1 0.83), Neural Network (F1 0.89)
- **Admin dashboard** — per-product sentiment charts, suspicious review flagging, real-time moderation
- **REST API** — `/analyze`, `/analyze/multimodal`, `/analyze/batch` endpoints
- **Rule-based fallback** — works without a trained model using emoji lexicon + keyword matching
- **Deployable** — Node.js-compatible REST interface, single-file joblib model

---

## Installation

```bash
pip install echo-feeling
```

Or from source:

```bash
git clone https://github.com/Becadyop/final.git
cd final
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### Text-only (v1 compatible)

```python
from echo_feeling import analyze

result = analyze("I feel really happy today!")
print(result)  # → "positive"
```

### Multimodal

```python
from echo_feeling import analyze_multimodal

result = analyze_multimodal(
    text="Loved this product! 😊",
    emojis=["😊", "👍"],
    sticker="positive",
)
print(result)
# {
#   "sentiment": "positive",
#   "confidence": 0.91,
#   "emoji_score": 0.875,
#   "emoji_found": ["😊"],
#   "sticker_sentiment": "positive"
# }
```

### Engine (full pipeline)

```python
from echo_feeling import EchoFeelingEngine

engine = EchoFeelingEngine(use_bert=False, model_type="neural_network")
engine.load_model("models/echo_feeling_nn.joblib")

result = engine.predict_review("Terrible quality, broke in one day 😠")
print(result["sentiment"])    # → "negative"
print(result["confidence"])   # → 0.88
```

---

## Six-Phase Pipeline

| Phase | Description |
|-------|-------------|
| 1 | **E-commerce website** as data source (review + emoji + sticker collection) |
| 2 | **Preprocessing** — tokenization, stop-word removal, lemmatization, emoji standardization |
| 3 | **Feature extraction** — BoW, TF-IDF, BERT embeddings, emoji scores, sticker encodings |
| 4 | **Multimodal fusion** — L2-normalized concatenation into unified feature vector |
| 5 | **Model training** — SVM, Random Forest, MLP with k-fold CV and confusion matrix analysis |
| 6 | **Deployment** — Flask REST API + admin panel with interactive sentiment charts |

---

## Training a Model

Place your dataset at `data/reviews.csv` with columns `review` and `label` (0=negative, 1=neutral, 2=positive, 3=suspicious).  
Place sticker PNGs under `data/stickers/positive/`, `data/stickers/negative/`, `data/stickers/neutral/`.

```bash
python training_phase/train.py
# Optional flags:
#   --data   path/to/reviews.csv
#   --sticker path/to/stickers/
#   --bert   (include BERT embeddings; slower)
```

Training produces:
- `models/echo_feeling_nn.joblib` — best model
- `reports/sentiment_distribution.png`
- `reports/model_comparison.png`
- `reports/confusion_matrix_*.png`
- `reports/training_report.json`

---

## Admin Dashboard

```bash
python deployment_phase/app.py
# → http://localhost:5001/admin
```

The dashboard provides:
- Per-product sentiment distribution (donut chart)
- Review volume over time (line chart)
- Positive / negative / neutral / suspicious counts
- Real-time review submission and sentiment analysis
- Admin delete for suspicious or negative reviews

---

## REST API

Start the API server:

```bash
python -m echo_feeling.api
# → http://localhost:5000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/model/info` | Model metadata |
| POST | `/analyze` | Text-only sentiment |
| POST | `/analyze/multimodal` | Text + emoji + sticker |
| POST | `/analyze/batch` | Batch inference |

### Example

```bash
curl -X POST http://localhost:5000/analyze/multimodal \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing product! 😊👍", "sticker": "positive"}'
```

Response:
```json
{
  "sentiment": "positive",
  "label_index": 2,
  "confidence": 0.91,
  "emoji_score": 0.875,
  "emoji_found": ["😊", "👍"],
  "sticker_sentiment": "positive",
  "model": "neural_network"
}
```

---

## Project Structure

```
echo_feeling/
│
├── echo_feeling/
│   ├── __init__.py        # Public API surface
│   ├── api.py             # Flask REST API + rule-based fallback
│   ├── engine.py          # Six-phase pipeline orchestrator
│   ├── preprocessor.py    # Phase 2: text normalization, emoji & sticker preprocessing
│   ├── features.py        # Phase 3: BoW, TF-IDF, BERT, emoji, sticker feature extractors
│   ├── fusion.py          # Phase 4: multimodal feature fusion
│   └── model.py           # Phase 5: SVM / Random Forest / Neural Network classifiers
│
├── training_phase/
│   └── train.py           # Full training + evaluation + report generation
│
├── deployment_phase/
│   └── app.py             # Admin dashboard (Flask)
│
├── data/
│   ├── reviews.csv        # Training data (review, label)
│   └── stickers/          # PNG sticker dataset
│       ├── positive/
│       ├── negative/
│       └── neutral/
│
├── models/                # Saved model artifacts (.joblib)
├── reports/               # Training charts and JSON report
├── requirements.txt
├── setup.py
└── README.md
```

---

## Sentiment Classes

| Label | Index | Description |
|-------|-------|-------------|
| negative | 0 | Dissatisfied customer feedback |
| neutral | 1 | Neither positive nor negative |
| positive | 2 | Satisfied customer feedback |
| suspicious | 3 | Potentially fake or promotional review |

---

## Model Performance (paper results)

| Model | F1 Macro | Accuracy |
|-------|----------|----------|
| Neural Network | **0.89** | — |
| SVM | 0.85 | — |
| Random Forest | 0.83 | — |

---

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and ensure the pipeline works end-to-end
5. Submit a pull request

---

## License

This project is licensed under the MIT License.

---

## Links

- GitHub: https://github.com/Becadyop/final  
- PyPI: https://pypi.org/project/echo-feeling/

---

## Authors

Created by **Ivin Issac, Romin Varghese, R Arun, Swathi Lekshmi SS**  
Guided by **Hridhya J**, Assistant Professor — Computer Science and Engineering,  
College of Engineering Karunagappally
