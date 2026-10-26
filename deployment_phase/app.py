"""
Deployment Phase — Echo Feeling Admin Dashboard
=================================================
Flask application that:
  • Hosts an e-commerce product review interface (Phase 1 data source)
  • Displays per-product sentiment distribution charts (Phase 6 visualization)
  • Allows admins to review and delete suspicious/negative comments
  • Calls the Echo Feeling engine in real-time as reviews are submitted

Run:
  cd deployment_phase
  python app.py
  → http://localhost:5001/admin

Environment variables:
  ECHO_MODEL_PATH   path to trained joblib model (default models/echo_feeling_nn.joblib)
  SECRET_KEY        Flask session secret (default: echo-feeling-dev)
"""

import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from flask import (
    Flask, request, jsonify, render_template_string,
    redirect, url_for, session,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from echo_feeling.engine import EchoFeelingEngine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "echo-feeling-dev")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "ECHO_MODEL_PATH",
    str(Path(__file__).parent.parent / "models" / "echo_feeling_nn.joblib"),
)
_engine = EchoFeelingEngine(use_bert=False)
if os.path.exists(MODEL_PATH):
    _engine.load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# In-memory review store  (replace with a real DB in production)
# ---------------------------------------------------------------------------
# Structure: { product_id: { "name": str, "reviews": [review_dict, ...] } }
PRODUCTS: dict = {
    "p001": {"name": "Wireless Headphones X200", "reviews": []},
    "p002": {"name": "Running Shoes ProFit",      "reviews": []},
    "p003": {"name": "Stainless Water Bottle",    "reviews": []},
}


def _analyze(text: str) -> dict:
    """Run engine or fallback rule-based analysis."""
    if _engine._fitted:
        return _engine.predict_review(text)
    # Inline fallback
    from echo_feeling.api import _rule_based_fallback
    return _rule_based_fallback(text)


def _sentiment_color(sentiment: str) -> str:
    return {
        "positive": "#2ecc71",
        "neutral": "#95a5a6",
        "negative": "#e74c3c",
        "suspicious": "#e67e22",
    }.get(sentiment, "#bdc3c7")


def _product_stats(product_id: str) -> dict:
    reviews = PRODUCTS[product_id]["reviews"]
    counts = {"positive": 0, "neutral": 0, "negative": 0, "suspicious": 0}
    for r in reviews:
        counts[r.get("sentiment", "neutral")] += 1
    return {"total": len(reviews), **counts}


# ---------------------------------------------------------------------------
# HTML templates (inline for single-file portability)
# ---------------------------------------------------------------------------

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Echo Feeling — {{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f4f6f9; color: #2c3e50; }
    nav  { background: #1a252f; color: #fff; padding: 14px 28px;
           display: flex; align-items: center; gap: 16px; }
    nav h1 { font-size: 1.3rem; }
    nav span { font-size: 0.85rem; opacity: .7; }
    nav a { color: #3498db; text-decoration: none; margin-left: auto; font-size:.9rem; }
    .container { max-width: 1100px; margin: 32px auto; padding: 0 20px; }
    .card { background: #fff; border-radius: 10px; padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 24px; }
    h2 { font-size: 1.1rem; margin-bottom: 16px; color: #1a252f; }
    .badge { display:inline-block; padding:3px 10px; border-radius:12px;
             font-size:.78rem; font-weight:600; color:#fff; }
    .badge-positive   { background:#2ecc71; }
    .badge-neutral    { background:#95a5a6; }
    .badge-negative   { background:#e74c3c; }
    .badge-suspicious { background:#e67e22; }
    table { width:100%; border-collapse:collapse; font-size:.9rem; }
    th, td { padding: 10px 12px; text-align:left; border-bottom:1px solid #ecf0f1; }
    th { background:#f8f9fa; font-weight:600; }
    .btn { padding:6px 14px; border:none; border-radius:6px; cursor:pointer;
           font-size:.85rem; font-weight:600; }
    .btn-delete { background:#e74c3c; color:#fff; }
    .btn-delete:hover { background:#c0392b; }
    .btn-primary { background:#3498db; color:#fff; }
    .btn-primary:hover { background:#2980b9; }
    .stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
    .stat-box { border-radius:8px; padding:14px; text-align:center; color:#fff; }
    .stat-box .num { font-size:1.8rem; font-weight:700; }
    .stat-box .lbl { font-size:.8rem; opacity:.9; }
    input[type=text], textarea {
      width:100%; padding:10px; border:1px solid #ddd; border-radius:6px;
      font-size:.95rem; resize:vertical; }
    label { display:block; font-size:.88rem; font-weight:600;
            margin-bottom:6px; margin-top:14px; }
    select { padding:8px; border:1px solid #ddd; border-radius:6px; }
    .chart-wrap { height:260px; position:relative; }
    .products-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
    .product-card { background:#fff; border-radius:10px; padding:20px;
                    box-shadow:0 2px 8px rgba(0,0,0,.08);
                    cursor:pointer; transition:transform .15s; }
    .product-card:hover { transform:translateY(-3px); }
    .product-card h3 { font-size:.98rem; margin-bottom:8px; }
    .sentiment-bar { height:8px; border-radius:4px; background:#ecf0f1;
                     margin-top:10px; display:flex; overflow:hidden; }
    .bar-seg { height:100%; }
    .flash { background:#d5f5e3; border:1px solid #2ecc71; border-radius:6px;
             padding:10px 16px; margin-bottom:16px; font-size:.9rem; }
    @media(max-width:700px){.stat-grid{grid-template-columns:1fr 1fr;}
      .products-grid{grid-template-columns:1fr;}}
  </style>
</head>
<body>
<nav>
  <h1>🎵 Echo Feeling</h1>
  <span>AI-Powered Review Mining</span>
  <a href="/admin">Dashboard</a>
</nav>
<div class="container">
  {% if flash %}<div class="flash">{{ flash }}</div>{% endif %}
  {% block content %}{% endblock %}
</div>
</body>
</html>
"""

DASHBOARD_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<h2 style="font-size:1.4rem;margin-bottom:20px;">📊 Product Sentiment Dashboard</h2>
<div class="products-grid">
{% for pid, product in products.items() %}
{% set stats = product_stats[pid] %}
<div class="product-card" onclick="location.href='/admin/product/{{ pid }}'">
  <h3>{{ product.name }}</h3>
  <p style="font-size:.82rem;color:#7f8c8d;">{{ stats.total }} reviews</p>
  <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">
    <span class="badge badge-positive">+{{ stats.positive }}</span>
    <span class="badge badge-neutral">~{{ stats.neutral }}</span>
    <span class="badge badge-negative">-{{ stats.negative }}</span>
    <span class="badge badge-suspicious">⚠{{ stats.suspicious }}</span>
  </div>
  <div class="sentiment-bar" style="margin-top:12px;">
    {% if stats.total > 0 %}
    <div class="bar-seg" style="width:{{ (stats.positive/stats.total*100)|int }}%;background:#2ecc71;"></div>
    <div class="bar-seg" style="width:{{ (stats.neutral/stats.total*100)|int }}%;background:#95a5a6;"></div>
    <div class="bar-seg" style="width:{{ (stats.negative/stats.total*100)|int }}%;background:#e74c3c;"></div>
    <div class="bar-seg" style="width:{{ (stats.suspicious/stats.total*100)|int }}%;background:#e67e22;"></div>
    {% endif %}
  </div>
</div>
{% endfor %}
</div>
""")

PRODUCT_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
  <a href="/admin" style="color:#3498db;text-decoration:none;">← Dashboard</a>
  <h2 style="font-size:1.3rem;">{{ product.name }}</h2>
</div>

<div class="stat-grid" style="margin-bottom:24px;">
  <div class="stat-box" style="background:#1a252f;">
    <div class="num">{{ stats.total }}</div><div class="lbl">Total Reviews</div></div>
  <div class="stat-box" style="background:#2ecc71;">
    <div class="num">{{ stats.positive }}</div><div class="lbl">Positive</div></div>
  <div class="stat-box" style="background:#e74c3c;">
    <div class="num">{{ stats.negative }}</div><div class="lbl">Negative</div></div>
  <div class="stat-box" style="background:#e67e22;">
    <div class="num">{{ stats.suspicious }}</div><div class="lbl">Suspicious</div></div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px;">
  <div class="card">
    <h2>Sentiment Distribution</h2>
    <div class="chart-wrap"><canvas id="donutChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Review Volume Over Time</h2>
    <div class="chart-wrap"><canvas id="lineChart"></canvas></div>
  </div>
</div>

<div class="card">
  <h2>Submit a Review</h2>
  <form method="POST" action="/admin/product/{{ product_id }}/review">
    <label>Your Review (text, emojis welcome)</label>
    <textarea name="review_text" rows="3" placeholder="e.g. Great product! 😊👍"></textarea>
    <label>Sticker Sentiment (optional)</label>
    <select name="sticker">
      <option value="">— none —</option>
      <option value="positive">😊 Positive sticker</option>
      <option value="neutral">😐 Neutral sticker</option>
      <option value="negative">😠 Negative sticker</option>
    </select>
    <br><br>
    <button class="btn btn-primary" type="submit">Analyze &amp; Submit</button>
  </form>
</div>

<div class="card">
  <h2>All Reviews</h2>
  {% if reviews %}
  <table>
    <tr><th>Review</th><th>Sentiment</th><th>Confidence</th><th>Emoji Score</th><th>Date</th><th>Action</th></tr>
    {% for r in reviews %}
    <tr>
      <td>{{ r.text[:80] }}{% if r.text|length > 80 %}…{% endif %}</td>
      <td><span class="badge badge-{{ r.sentiment }}">{{ r.sentiment }}</span></td>
      <td>{{ "%.0f"|format(r.confidence*100) }}%</td>
      <td>{{ r.emoji_score }}</td>
      <td style="font-size:.8rem;color:#7f8c8d;">{{ r.date }}</td>
      <td>
        <form method="POST" action="/admin/product/{{ product_id }}/delete/{{ r.id }}" style="display:inline;">
          <button class="btn btn-delete" type="submit" onclick="return confirm('Delete this review?')">🗑</button>
        </form>
      </td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p style="color:#7f8c8d;font-size:.9rem;">No reviews yet. Submit one above.</p>
  {% endif %}
</div>

<script>
const stats = {{ stats_json }};
// Donut chart
new Chart(document.getElementById('donutChart'), {
  type: 'doughnut',
  data: {
    labels: ['Positive', 'Neutral', 'Negative', 'Suspicious'],
    datasets:[{ data:[stats.positive,stats.neutral,stats.negative,stats.suspicious],
      backgroundColor:['#2ecc71','#95a5a6','#e74c3c','#e67e22'],
      borderWidth:2, borderColor:'#fff' }]
  },
  options:{ plugins:{legend:{position:'bottom'}}, maintainAspectRatio:false }
});
// Line chart (mock time-series from reviews)
const reviews = {{ reviews_json }};
const dateCounts = {};
reviews.forEach(r=>{ const d=r.date.split(' ')[0]; dateCounts[d]=(dateCounts[d]||0)+1; });
const dates = Object.keys(dateCounts).sort();
new Chart(document.getElementById('lineChart'),{
  type:'line',
  data:{
    labels: dates.length ? dates : ['No data'],
    datasets:[{ label:'Reviews/day',
      data: dates.map(d=>dateCounts[d]),
      borderColor:'#3498db', backgroundColor:'rgba(52,152,219,.1)',
      tension:.4, fill:true }]
  },
  options:{scales:{y:{beginAtZero:true,ticks:{stepSize:1}}},
    plugins:{legend:{display:false}}, maintainAspectRatio:false}
});
</script>
""")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/admin")
def dashboard():
    product_stats = {pid: _product_stats(pid) for pid in PRODUCTS}
    flash = session.pop("flash", None)
    return render_template_string(
        DASHBOARD_HTML,
        title="Dashboard",
        products=PRODUCTS,
        product_stats=product_stats,
        flash=flash,
    )


@app.route("/admin/product/<product_id>")
def product_page(product_id):
    if product_id not in PRODUCTS:
        return "Product not found", 404
    product = PRODUCTS[product_id]
    stats = _product_stats(product_id)
    reviews = product["reviews"]
    flash = session.pop("flash", None)
    return render_template_string(
        PRODUCT_HTML,
        title=product["name"],
        product=product,
        product_id=product_id,
        reviews=reviews,
        stats=stats,
        stats_json=json.dumps(stats),
        reviews_json=json.dumps(reviews),
        flash=flash,
    )


@app.route("/admin/product/<product_id>/review", methods=["POST"])
def submit_review(product_id):
    if product_id not in PRODUCTS:
        return "Product not found", 404
    text = request.form.get("review_text", "").strip()
    sticker = request.form.get("sticker", "")
    if not text:
        session["flash"] = "Review text cannot be empty."
        return redirect(url_for("product_page", product_id=product_id))

    result = _analyze(text)

    review_record = {
        "id": str(uuid.uuid4())[:8],
        "text": text,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "emoji_score": result.get("emoji_score", 0.0),
        "emoji_found": result.get("emoji_found", []),
        "sticker": sticker or None,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    PRODUCTS[product_id]["reviews"].insert(0, review_record)
    session["flash"] = (
        f"Review analyzed: {result['sentiment'].upper()} "
        f"(confidence {result['confidence']*100:.0f}%)"
    )
    return redirect(url_for("product_page", product_id=product_id))


@app.route("/admin/product/<product_id>/delete/<review_id>", methods=["POST"])
def delete_review(product_id, review_id):
    if product_id not in PRODUCTS:
        return "Product not found", 404
    before = len(PRODUCTS[product_id]["reviews"])
    PRODUCTS[product_id]["reviews"] = [
        r for r in PRODUCTS[product_id]["reviews"] if r["id"] != review_id
    ]
    after = len(PRODUCTS[product_id]["reviews"])
    session["flash"] = "Review deleted." if before != after else "Review not found."
    return redirect(url_for("product_page", product_id=product_id))


# ---------------------------------------------------------------------------
# JSON API (for Node.js / external callers)
# ---------------------------------------------------------------------------

@app.route("/api/products", methods=["GET"])
def api_products():
    return jsonify({
        pid: {
            "name": p["name"],
            "stats": _product_stats(pid),
        }
        for pid, p in PRODUCTS.items()
    })


@app.route("/api/product/<product_id>/reviews", methods=["GET"])
def api_product_reviews(product_id):
    if product_id not in PRODUCTS:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "product": PRODUCTS[product_id]["name"],
        "stats": _product_stats(product_id),
        "reviews": PRODUCTS[product_id]["reviews"],
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "'text' required"}), 400
    result = _analyze(text)
    return jsonify(result)


if __name__ == "__main__":
    print("Echo Feeling Admin Dashboard → http://localhost:5001/admin")
    app.run(host="0.0.0.0", port=5001, debug=True)
