"""
train_model.py
==============
Customer Churn Prediction — Model Training Pipeline

Trains 4 ML models, compares performance, and saves the best model.

Models trained:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest  ← Best (saved)
  4. K-Nearest Neighbors

Author: Data Science Portfolio Project
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Add src/ to path ─────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from preprocessing import run_preprocessing_pipeline

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH  = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLS_PATH   = os.path.join(MODELS_DIR, "feature_columns.pkl")


# ─────────────────────────────────────────────
# STEP 1 — LOAD & PREPROCESS
# ─────────────────────────────────────────────
print("=" * 65)
print("   CUSTOMER CHURN PREDICTION — MODEL TRAINING")
print("=" * 65)

X, y, df_encoded = run_preprocessing_pipeline(DATA_PATH)


# ─────────────────────────────────────────────
# STEP 2 — TRAIN-TEST SPLIT (80/20)
# ─────────────────────────────────────────────
"""
Why 80-20 split?
  - 80% gives the model enough examples to learn patterns.
  - 20% ensures a meaningful evaluation on unseen data.
  - Too small a test set → unreliable metrics.
  - Too large a test set → less data to train on.

Why random_state=42?
  - Ensures reproducibility — anyone running the code gets the same split.
  - 42 is a convention (from "The Hitchhiker's Guide to the Galaxy").
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Before SMOTE - Train set: {X_train.shape[0]} samples")
print(f"📊 Before SMOTE - Churn rate: {y_train.mean()*100:.2f}%")

# Apply SMOTE to handle class imbalance (Only on Train Set!)
print("\n🔄 Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"📊 After SMOTE - Train set: {X_train.shape[0]} samples")
print(f"📊 After SMOTE - Churn rate: {y_train.mean()*100:.2f}%")
print(f"📊 Test set remains untouched: {X_test.shape[0]} samples (Churn rate: {y_test.mean()*100:.2f}%)")


# ─────────────────────────────────────────────
# STEP 3 — FEATURE SCALING (for LR & KNN)
# ─────────────────────────────────────────────
"""
StandardScaler: transforms features to zero mean, unit variance.
  - Essential for Logistic Regression & KNN (distance-based)
  - Not needed for tree-based models but doesn't harm them
  - We fit on train only to avoid data leakage
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save scaler for deployment
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"\n✅ Scaler saved → {SCALER_PATH}")

# Save feature column order for deployment
with open(COLS_PATH, "wb") as f:
    pickle.dump(list(X.columns), f)
print(f"✅ Feature columns saved → {COLS_PATH}")


# ─────────────────────────────────────────────
# STEP 4 — DEFINE MODELS
# ─────────────────────────────────────────────
"""
Model explanations:

1. LOGISTIC REGRESSION
   - A linear model that outputs probability of churn using sigmoid function.
   - Advantage: Fast, interpretable, good baseline.
   - Disadvantage: Assumes linear decision boundary; may underfit complex data.

2. DECISION TREE
   - Splits data into branches based on feature thresholds (like a flowchart).
   - Advantage: Interpretable, handles non-linearity.
   - Disadvantage: Prone to overfitting on training data.

3. RANDOM FOREST
   - Builds 100s of Decision Trees on random subsets → averages predictions.
   - Advantage: Reduces overfitting, handles missing values, gives feature importance.
   - Disadvantage: Slower, less interpretable than single tree.

4. K-NEAREST NEIGHBORS
   - Classifies a point based on the majority class of its K nearest neighbors.
   - Advantage: Simple, non-parametric, no training phase.
   - Disadvantage: Slow at prediction, sensitive to feature scaling.
"""

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=0.1
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, n_jobs=-1
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7, metric="minkowski"
    ),
}

# Models that need scaled input
NEEDS_SCALING = ["Logistic Regression", "K-Nearest Neighbors"]


# ─────────────────────────────────────────────
# STEP 5 — TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────

results = []
trained_models = {}

print("\n" + "=" * 65)
print("   TRAINING MODELS...")
print("=" * 65)

for name, model in models.items():
    # Choose scaled or raw features
    if name in NEEDS_SCALING:
        Xtr, Xte = X_train_scaled, X_test_scaled
    else:
        Xtr, Xte = X_train.values, X_test.values

    # Train
    model.fit(Xtr, y_train)
    trained_models[name] = model

    # Predict
    y_pred     = model.predict(Xte)
    y_prob     = model.predict_proba(Xte)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_prob)

    results.append({
        "Model"     : name,
        "Accuracy"  : round(acc  * 100, 2),
        "Precision" : round(prec * 100, 2),
        "Recall"    : round(rec  * 100, 2),
        "F1 Score"  : round(f1   * 100, 2),
        "ROC-AUC"   : round(roc  * 100, 2),
    })

    print(f"\n🔹 {name}")
    print(f"   Accuracy: {acc*100:.2f}%  |  F1: {f1*100:.2f}%  |  ROC-AUC: {roc*100:.2f}%")


# ─────────────────────────────────────────────
# STEP 6 — MODEL COMPARISON TABLE
# ─────────────────────────────────────────────

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("F1 Score", ascending=False).reset_index(drop=True)

print("\n" + "=" * 65)
print("   MODEL COMPARISON TABLE")
print("=" * 65)
print(df_results.to_string(index=False))
print("=" * 65)

# Identify best model by F1 Score
best_model_name = df_results.iloc[0]["Model"]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1 Score:  {df_results.iloc[0]['F1 Score']}%")
print(f"   ROC-AUC:   {df_results.iloc[0]['ROC-AUC']}%")


# ─────────────────────────────────────────────
# STEP 7 — SAVE RANDOM FOREST MODEL
# ─────────────────────────────────────────────
"""
We save Random Forest specifically (not just the best) because:
  - It provides feature importances for explainability
  - It's the most robust and production-ready tree ensemble
  - Interview expectation: Random Forest is the standard baseline for churn

pickle.dump() serializes the Python object to a binary file.
pickle.load() deserializes it back — no need to retrain!
"""

rf_model = trained_models["Random Forest"]

with open(MODEL_PATH, "wb") as f:
    pickle.dump(rf_model, f)

print(f"\n✅ Random Forest model saved → {MODEL_PATH}")


# ─────────────────────────────────────────────
# STEP 8 — FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
"""
Random Forest provides feature importance scores based on how much each
feature reduces impurity (Gini index) across all trees.

Higher score = stronger predictor of churn.
"""

importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature"   : X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

print("\n📊 Top 15 Feature Importances:")
print(feat_imp_df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feat_imp_df)))
bars = ax.barh(feat_imp_df["Feature"][::-1],
               feat_imp_df["Importance"][::-1],
               color=colors[::-1], edgecolor="white")

ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("🌲 Random Forest — Top 15 Feature Importances", fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
feat_imp_path = os.path.join(MODELS_DIR, "feature_importance.png")
plt.savefig(feat_imp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Feature importance plot saved → {feat_imp_path}")

# Save results table
results_path = os.path.join(MODELS_DIR, "model_comparison.csv")
df_results.to_csv(results_path, index=False)
print(f"✅ Model comparison table saved → {results_path}")

print("\n🎉 Training complete! All artifacts saved to models/")
print("=" * 65)
