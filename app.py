"""
app.py
======
Production-ready Flask REST API for Malicious URL Detection.

Endpoints:
  POST /predict       — Analyse a URL
  GET  /health        — Health check
  GET  /model/info    — Model metadata

Author: Cybersecurity ML Pipeline
"""

import os
import re
import json
import time
import logging
import traceback
from functools import wraps
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from feature_extractor import (
    extract_features,
    extract_feature_vector,
    get_triggered_reasons,
    FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")

_model = None
_metadata = {}


def load_model():
    """Load model from disk. Called once at startup."""
    global _model, _metadata
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        log.info(f"Model loaded from {MODEL_PATH}")
    else:
        log.warning("No trained model found at models/best_model.pkl — using heuristic fallback.")
        _model = None

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _metadata = json.load(f)


load_model()


# ---------------------------------------------------------------------------
# Risk Scoring
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "low":    (0.0,  0.35),
    "medium": (0.35, 0.65),
    "high":   (0.65, 1.01),
}

RISK_LABELS = {
    "low":    ("Safe",       "Low"),
    "medium": ("Suspicious", "Medium"),
    "high":   ("Malicious",  "High"),
}


def probability_to_risk(probability: float) -> tuple:
    """Map model probability to (prediction_label, risk_level)."""
    for level, (lo, hi) in THRESHOLDS.items():
        if lo <= probability < hi:
            return RISK_LABELS[level]
    return ("Malicious", "High")


# ---------------------------------------------------------------------------
# Heuristic Fallback (when no model is trained yet)
# ---------------------------------------------------------------------------

def heuristic_score(features: dict) -> float:
    """
    Rule-based probability score when no ML model is available.
    Useful for demo / development purposes.
    """
    score = 0.0

    if features["has_ip_address"]:           score += 0.35
    if features["num_at_signs"] > 0:         score += 0.25
    if not features["has_https"]:            score += 0.15
    if features["has_suspicious_keywords"]:  score += 0.20
    if features["is_url_shortener"]:         score += 0.15
    if features["url_length"] > 100:         score += 0.15
    if features["num_subdomains"] >= 3:      score += 0.15
    if features["entropy"] > 4.5:            score += 0.10
    if features["tld_suspicious"]:           score += 0.10
    if features["num_hyphens"] > 4:          score += 0.10

    return min(score, 0.98)


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(
    r"^(https?://)?"
    r"(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})"
    r"(/[^\s]*)?"
    r"(\?[^\s]*)?"
    r"(#[^\s]*)?$"
)

MAX_URL_LENGTH = 2048


def validate_url(url: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message)."""
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string."
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        return False, f"URL exceeds maximum length of {MAX_URL_LENGTH} characters."
    if " " in url:
        return False, "URL must not contain spaces."
    if not URL_PATTERN.match(url):
        return False, "Invalid URL format."
    return True, ""


# ---------------------------------------------------------------------------
# Rate Limiting (simple in-memory, replace with Redis for production)
# ---------------------------------------------------------------------------

_rate_store: dict = {}
RATE_LIMIT = 60      # requests
RATE_WINDOW = 60     # seconds


def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time.time()
        window_start, count = _rate_store.get(ip, (now, 0))
        if now - window_start > RATE_WINDOW:
            _rate_store[ip] = (now, 1)
        else:
            if count >= RATE_LIMIT:
                return jsonify({
                    "error": "Rate limit exceeded. Try again in a moment.",
                    "retry_after": int(RATE_WINDOW - (now - window_start)),
                }), 429
            _rate_store[ip] = (window_start, count + 1)
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    if not _metadata:
        return jsonify({"message": "No model metadata available."}), 404
    return jsonify(_metadata)


@app.route("/predict", methods=["POST"])
@rate_limit
def predict():
    """
    Predict whether a URL is malicious.

    Request body:
      { "url": "http://example.com/path" }

    Response:
      {
        "url": "...",
        "prediction": "Malicious",
        "confidence": 0.87,
        "risk_level": "High",
        "reasons": ["...", "..."],
        "features": { ... },
        "model": "Random Forest",
        "latency_ms": 12.3
      }
    """
    start = time.perf_counter()

    # Parse request
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    raw_url = data.get("url", "")

    # Sanitise + validate
    url = str(raw_url).strip()
    valid, err = validate_url(url)
    if not valid:
        return jsonify({"error": err}), 422

    try:
        # Extract features
        features = extract_features(url)
        feature_vector = extract_feature_vector(url).reshape(1, -1)

        # Predict
        if _model is not None:
            probability = float(_model.predict_proba(feature_vector)[0][1])
            model_name = _metadata.get("best_model", "ML Model")
        else:
            probability = heuristic_score(features)
            model_name = "Heuristic (no model trained)"

        prediction, risk_level = probability_to_risk(probability)
        reasons = get_triggered_reasons(features)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        log.info(
            f"PREDICT | url={url[:80]} | prediction={prediction} "
            f"| conf={probability:.3f} | {elapsed_ms}ms"
        )

        return jsonify({
            "url": url,
            "prediction": prediction,
            "confidence": round(probability, 4),
            "risk_level": risk_level,
            "reasons": reasons,
            "features": {k: float(v) for k, v in features.items()},
            "model": model_name,
            "latency_ms": elapsed_ms,
        })

    except Exception as e:
        log.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal prediction error. Please try again."}), 500


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    log.info(f"Starting API server on port {port} | debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
