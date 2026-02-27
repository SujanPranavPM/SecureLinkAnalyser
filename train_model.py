"""
train_model.py
==============
End-to-end ML pipeline for malicious URL detection.

Steps:
  1. Load + validate dataset
  2. Feature engineering via feature_extractor
  3. Train Logistic Regression & Random Forest
  4. Evaluate both with full metrics
  5. Save best model + scaler + metadata

Usage:
  python train_model.py --data dataset.csv --output models/

Dataset CSV format:
  url,label
  https://google.com,0
  http://phish-login.ru/account,1
  (label: 0 = benign, 1 = malicious/phishing)
"""

import os
import sys
import json
import argparse
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the URL dataset."""
    log.info(f"Loading dataset from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Support multiple common column naming conventions
    url_col_candidates = ["url", "urls", "domain", "link"]
    label_col_candidates = ["label", "labels", "class", "target", "phishing", "type"]

    url_col = next((c for c in url_col_candidates if c in df.columns), None)
    label_col = next((c for c in label_col_candidates if c in df.columns), None)

    if not url_col or not label_col:
        raise ValueError(
            f"Cannot identify URL/label columns. Found: {list(df.columns)}\n"
            "Expected columns like 'url' and 'label'."
        )

    df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})
    df.dropna(inplace=True)
    df["url"] = df["url"].astype(str).str.strip()

    # Normalize labels to 0/1 integers
    unique_labels = df["label"].unique()
    if set(unique_labels).issubset({0, 1, "0", "1"}):
        df["label"] = df["label"].astype(int)
    else:
        # Try mapping common string labels
        label_map = {}
        for lbl in unique_labels:
            l = str(lbl).lower()
            if l in {"benign", "safe", "legitimate", "0", "good"}:
                label_map[lbl] = 0
            elif l in {"phishing", "malicious", "bad", "1", "defacement", "malware"}:
                label_map[lbl] = 1
        if len(label_map) < len(unique_labels):
            raise ValueError(f"Cannot map all labels to 0/1. Found: {unique_labels}")
        df["label"] = df["label"].map(label_map)

    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    log.info(f"Dataset loaded: {len(df):,} rows | "
             f"Benign={sum(df['label']==0):,} | Malicious={sum(df['label']==1):,}")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """Apply feature extractor to every URL and return X, y."""
    log.info("Extracting features from URLs (this may take a moment)...")

    rows = []
    failed = 0
    for url in df["url"]:
        try:
            rows.append(extract_features(url))
        except Exception:
            rows.append({name: 0 for name in FEATURE_NAMES})
            failed += 1

    if failed:
        log.warning(f"Feature extraction failed for {failed} URLs (replaced with zeros).")

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = df["label"].values

    log.info(f"Feature matrix shape: {X.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Train, predict, and return full evaluation metrics."""
    log.info(f"Training: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Cross-validation F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"] = cv_scores.std()

    # Print report
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  CV F1     : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Benign", "Malicious"]))
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    return metrics, model, y_prob


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def save_plots(results: list, y_test: np.ndarray, output_dir: str):
    """Save confusion matrices and ROC curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (metrics, _, _) in zip(axes, results):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Benign", "Malicious"],
                    yticklabels=["Benign", "Malicious"], ax=ax)
        ax.set_title(f"{metrics['name']}\nF1={metrics['f1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close()
    log.info("Saved confusion_matrices.png")

    # ROC curves
    plt.figure(figsize=(8, 6))
    for metrics, model, y_prob in results:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{metrics['name']} (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()
    log.info("Saved roc_curves.png")


def save_feature_importance(model, output_dir: str):
    """Save Random Forest feature importance chart if applicable."""
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(10, 6))
    plt.barh(
        [FEATURE_NAMES[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color="steelblue",
    )
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()
    log.info("Saved feature_importance.png")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def train(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Load data
    df = load_dataset(data_path)

    # 2. Features
    X, y = build_feature_matrix(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 4. Train & evaluate all models
    models = get_models()
    all_results = []
    for name, model in models.items():
        metrics, fitted_model, y_prob = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        all_results.append((metrics, fitted_model, y_prob))

    # 5. Pick best by F1
    best_metrics, best_model, _ = max(all_results, key=lambda x: x[0]["f1"])
    log.info(f"\n✅ Best model: {best_metrics['name']} | F1={best_metrics['f1']:.4f}")

    # 6. Save plots
    save_plots(all_results, y_test, output_dir)
    # Save feature importance for RF specifically
    for metrics, model, _ in all_results:
        if "Random Forest" in metrics["name"]:
            save_feature_importance(model, output_dir)

    # 7. Save best model
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    log.info(f"Model saved → {model_path}")

    # 8. Save metadata
    metadata = {
        "best_model": best_metrics["name"],
        "trained_at": datetime.utcnow().isoformat(),
        "dataset": data_path,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_names": FEATURE_NAMES,
        "metrics": {
            k: v for k, v in best_metrics.items()
            if k not in ("confusion_matrix",)
        },
        "all_models": [
            {k: v for k, v in m.items() if k not in ("confusion_matrix",)}
            for m, _, _ in all_results
        ],
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata saved → {meta_path}")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Model : {best_metrics['name']}")
    print(f"  F1-Score   : {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC    : {best_metrics['roc_auc']:.4f}")
    print(f"  Saved to   : {model_path}")
    print(f"{'='*60}\n")

    return best_model, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train malicious URL detection model")
    parser.add_argument("--data", default="dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--output", default="models/", help="Output directory for model files")
    args = parser.parse_args()
    train(args.data, args.output)
