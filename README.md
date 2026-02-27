# ThreatScan — AI-Powered Malicious URL Detection

> A production-grade cybersecurity ML application built for final-year engineering placement portfolios.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🎯 What It Does

ThreatScan analyses any URL and predicts whether it is **Safe**, **Suspicious**, or **Malicious** using:

- **26 engineered features** (URL entropy, subdomains, IP detection, keyword matching, etc.)
- **Ensemble ML models** (Random Forest, Logistic Regression, Gradient Boosting)
- A **clean cyberpunk-style dashboard** with risk meter and explanation panel
- A **production-ready REST API** built with Flask

---

## 📁 Project Structure

```
malurl/
│
├── backend/
│   ├── app.py                  ← Flask REST API (main entry point)
│   ├── feature_extractor.py    ← Modular URL feature engineering
│   ├── requirements.txt
│   ├── models/
│   │   ├── best_model.pkl      ← Trained model (after training)
│   │   └── model_metadata.json
│   └── logs/
│       └── api.log
│
├── frontend/
│   └── index.html              ← Full SPA (no framework, vanilla JS)
│
├── ml_pipeline/
│   ├── train_model.py          ← End-to-end training pipeline
│   ├── generate_demo_dataset.py
│   └── dataset.csv             ← Your training data goes here
│
└── docs/
    └── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- pip

### 2. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

---

## 🧠 How to Train the Model

### Option A: Use your own dataset
Your CSV must have two columns:
```
url,label
https://google.com,0
http://phishing-site.xyz/login,1
```
- `label = 0` → Benign / Safe
- `label = 1` → Malicious / Phishing

Place it at `ml_pipeline/dataset.csv`.

### Option B: Generate a synthetic demo dataset
```bash
cd ml_pipeline
python generate_demo_dataset.py --output dataset.csv --n 2500
```
This creates 5,000 balanced URLs (2,500 benign + 2,500 phishing).

### Run the training pipeline
```bash
cd ml_pipeline
python train_model.py --data dataset.csv --output ../backend/models/
```

**Output:**
- `backend/models/best_model.pkl` — Serialized best model
- `backend/models/model_metadata.json` — Performance metrics
- `backend/models/confusion_matrices.png`
- `backend/models/roc_curves.png`
- `backend/models/feature_importance.png`

The pipeline trains **3 models** and prints a full comparison:
```
Logistic Regression   | Accuracy=0.9412 | F1=0.9405 | AUC=0.9803
Random Forest         | Accuracy=0.9731 | F1=0.9728 | AUC=0.9941
Gradient Boosting     | Accuracy=0.9618 | F1=0.9614 | AUC=0.9901
✅ Best model: Random Forest | F1=0.9728
```

---

## 🚀 How to Run Locally

```bash
cd backend
python app.py
```

The API runs at `http://localhost:5000`.

Open `frontend/index.html` in a browser (or serve statically):
```bash
# Option: serve frontend via Python
cd frontend
python -m http.server 8080
# Visit http://localhost:8080
```

> **Note:** The app works in heuristic mode even without a trained model — great for development and demo.

---

## 📡 API Documentation

### `POST /predict`

Analyse a URL for threats.

**Request:**
```json
{
  "url": "http://paypal-verify-account.xyz/login"
}
```

**Response:**
```json
{
  "url": "http://paypal-verify-account.xyz/login",
  "prediction": "Malicious",
  "confidence": 0.94,
  "risk_level": "High",
  "reasons": [
    "Contains phishing-associated keywords (e.g. login, verify, secure)",
    "Uncommon TLD — not a standard trusted extension",
    "No HTTPS — connection is unencrypted"
  ],
  "features": {
    "url_length": 42,
    "has_https": 0,
    "entropy": 3.94,
    "num_subdomains": 0,
    ...
  },
  "model": "Random Forest",
  "latency_ms": 14.2
}
```

**Risk Levels:**
| Confidence | Prediction  | Risk Level |
|------------|-------------|------------|
| 0.00–0.35  | Safe        | Low        |
| 0.35–0.65  | Suspicious  | Medium     |
| 0.65–1.00  | Malicious   | High       |

---

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

### `GET /model/info`
Returns full model metadata including accuracy, F1, and feature names.

---

## 🔧 Features Extracted (26 total)

| Feature | Description |
|---------|-------------|
| `url_length` | Total URL character count |
| `domain_length` | Length of the domain/netloc |
| `num_dots` | Number of dots |
| `num_hyphens` | Number of hyphens |
| `has_ip_address` | 1 if IP used instead of hostname |
| `has_https` | 1 if HTTPS protocol |
| `num_subdomains` | Count of subdomains |
| `has_suspicious_keywords` | 1 if phishing keywords found |
| `is_url_shortener` | 1 if known shortening service |
| `entropy` | Shannon entropy (measures randomness) |
| `digit_ratio` | Proportion of digit characters |
| `tld_suspicious` | 1 if uncommon TLD |
| `num_at_signs` | @ count (redirect trick) |
| `num_percent` | % count (obfuscation) |
| ... | +12 more |

---

## 🚢 Deployment

### Render.com (Recommended — Free)
1. Push code to GitHub
2. Create new **Web Service** on render.com
3. Set:
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `gunicorn backend.app:app --bind 0.0.0.0:$PORT`
4. Add environment variable: `FLASK_ENV=production`

### Railway
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```
Set start command to: `gunicorn backend.app:app`

### AWS EC2 (Manual)
```bash
# SSH into your instance
sudo apt update && sudo apt install python3-pip nginx -y
cd malurl && pip install -r backend/requirements.txt

# Run with gunicorn
gunicorn backend.app:app --workers 4 --bind 0.0.0.0:8000

# Configure nginx as reverse proxy to port 8000
```

---

## 🔒 Security Features

- Input validation with regex (rejects invalid URLs)
- URL length capping (max 2048 chars)
- Rate limiting (60 req/min per IP)
- No SQL — no injection surface
- CORS configured
- Full error handling (400, 422, 429, 500)
- Logging to file for audit trail

---

## 🧪 Testing the API

```bash
# Safe URL
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com"}'

# Phishing URL
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-login-verify@192.168.0.1/account/update"}'
```

---

## 📊 Dataset Sources (Real-World)

For production-quality training, use:
- **PhiUSIIL Phishing URL Dataset** — Kaggle
- **ISCX URL Dataset** — UNB
- **Phishing Site URLs** — Kaggle (Akashsdas)
- **OpenPhish** — https://openphish.com
- **PhishTank** — https://phishtank.org

---

## 👤 Author

Built as a final-year project demonstrating:
- Machine Learning (ensemble methods, feature engineering)
- Cybersecurity (URL threat detection)
- Backend engineering (REST API, rate limiting, logging)
- Frontend development (zero-dependency SPA)
- DevOps (multi-platform deployment)
