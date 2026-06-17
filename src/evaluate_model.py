"""
evaluate_model.py
=================
Customer Churn Prediction — Model Evaluation & Visualization

Loads the saved Random Forest model and produces:
  - Classification Report (Precision, Recall, F1)
  - Confusion Matrix (heatmap)
  - ROC Curve with AUC score
  - Detailed metric explanations

Author: Data Science Portfolio Project
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from preprocessing import run_preprocessing_pipeline

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data",   "Telco-Customer-Churn.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
PLOTS_DIR   = os.path.join(BASE_DIR, "models")


# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
print("=" * 65)
print("   CUSTOMER CHURN — MODEL EVALUATION")
print("=" * 65)

X, y, _ = run_preprocessing_pipeline(DATA_PATH)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"\n✅ Model loaded from: {MODEL_PATH}")

# Predictions
y_pred = model.predict(X_test.values)
y_prob = model.predict_proba(X_test.values)[:, 1]


# ─────────────────────────────────────────────
# METRIC EXPLANATIONS + REPORT
# ─────────────────────────────────────────────
"""
METRIC EXPLANATIONS:

1. ACCURACY  = (TP + TN) / Total
   → Overall correct predictions. Misleading with imbalanced classes.

2. PRECISION = TP / (TP + FP)
   → Of all predicted churners, how many actually churned?
   → High precision = fewer false alarms (good for campaign targeting)

3. RECALL    = TP / (TP + FN)
   → Of all actual churners, how many did we catch?
   → High recall = fewer missed churners (critical for retention!)

4. F1 SCORE  = 2 × (Precision × Recall) / (Precision + Recall)
   → Harmonic mean — balances precision and recall.
   → Best metric when classes are imbalanced.

5. ROC-AUC   = Area Under the ROC Curve
   → Measures how well the model distinguishes churners from non-churners.
   → AUC = 1.0 → perfect; AUC = 0.5 → random guessing.

CONFUSION MATRIX:
                  Predicted No    Predicted Yes
  Actual No           TN               FP
  Actual Yes          FN               TP

  FN (False Negative) = We predicted "stay" but customer actually churned.
  → Most costly mistake in churn prediction!
"""

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_prob)

print(f"""
┌─────────────────────────────────────────────┐
│         MODEL PERFORMANCE METRICS           │
├─────────────────────────────────────────────┤
│  Accuracy   : {acc*100:>6.2f}%                      │
│  Precision  : {prec*100:>6.2f}%                      │
│  Recall     : {rec*100:>6.2f}%                      │
│  F1 Score   : {f1*100:>6.2f}%                      │
│  ROC-AUC    : {roc*100:>6.2f}%                      │
└─────────────────────────────────────────────┘
""")

print("📋 Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


# ─────────────────────────────────────────────
# VISUALIZATION — Confusion Matrix + ROC Curve
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(1, 2, figure=fig)

# ── Plot 1: Confusion Matrix ─────────────────
ax1 = fig.add_subplot(gs[0])
cm  = confusion_matrix(y_test, y_pred)
labels = [["TN", "FP"], ["FN", "TP"]]
annot  = np.array([[f"{labels[i][j]}\n{cm[i][j]}"
                    for j in range(2)] for i in range(2)])

sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
            xticklabels=["Predicted: Stay", "Predicted: Churn"],
            yticklabels=["Actual: Stay",    "Actual: Churn"],
            linewidths=2, linecolor="white", ax=ax1,
            cbar_kws={"shrink": 0.8})

ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
ax1.set_xlabel("Predicted Label", fontsize=11)
ax1.set_ylabel("True Label",      fontsize=11)

# Annotate cost of FN
tn, fp, fn, tp = cm.ravel()
ax1.text(0.5, -0.15,
         f"FN={fn}  ← Missed churners (costly!)",
         ha="center", transform=ax1.transAxes,
         fontsize=10, color="red", style="italic")


# ── Plot 2: ROC Curve ────────────────────────
ax2  = fig.add_subplot(gs[1])
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

ax2.plot(fpr, tpr, color="#4F46E5", lw=2.5,
         label=f"Random Forest (AUC = {roc_auc:.3f})")
ax2.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)", alpha=0.5)
ax2.fill_between(fpr, tpr, alpha=0.1, color="#4F46E5")

ax2.set_xlabel("False Positive Rate (FPR)", fontsize=11)
ax2.set_ylabel("True Positive Rate (Recall)", fontsize=11)
ax2.set_title("ROC Curve — Random Forest", fontsize=14, fontweight="bold")
ax2.legend(loc="lower right", fontsize=10)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.02])
ax2.grid(alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.suptitle("Customer Churn — Random Forest Evaluation",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()

eval_plot_path = os.path.join(PLOTS_DIR, "evaluation_plots.png")
plt.savefig(eval_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Evaluation plots saved → {eval_plot_path}")
print("=" * 65)
