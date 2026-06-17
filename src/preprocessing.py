"""
preprocessing.py
================
Customer Churn Prediction — Data Cleaning & Feature Engineering Pipeline

Adapted for the IBM Telco Extended Dataset with columns:
  CustomerID, Gender, Senior Citizen, Partner, Dependents, Tenure Months,
  Phone Service, Multiple Lines, Internet Service, Online Security,
  Online Backup, Device Protection, Tech Support, Streaming TV,
  Streaming Movies, Contract, Paperless Billing, Payment Method,
  Monthly Charges, Total Charges, Churn Label (target)

Author: Data Science Portfolio Project
"""

import pandas as pd
import numpy as np
import os
import sys

# ─────────────────────────────────────────────
# COLUMN MAPPING  (dataset col -> standard name)
# ─────────────────────────────────────────────
RENAME_MAP = {
    "Gender"           : "gender",
    "Senior Citizen"   : "SeniorCitizen",
    "Partner"          : "Partner",
    "Dependents"       : "Dependents",
    "Tenure Months"    : "tenure",
    "Phone Service"    : "PhoneService",
    "Multiple Lines"   : "MultipleLines",
    "Internet Service" : "InternetService",
    "Online Security"  : "OnlineSecurity",
    "Online Backup"    : "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support"     : "TechSupport",
    "Streaming TV"     : "StreamingTV",
    "Streaming Movies" : "StreamingMovies",
    "Contract"         : "Contract",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method"   : "PaymentMethod",
    "Monthly Charges"  : "MonthlyCharges",
    "Total Charges"    : "TotalCharges",
    "Churn Label"      : "Churn",
}

# Columns to keep (drop geo, ID, redundant churn cols)
KEEP_COLS = list(RENAME_MAP.values())

# Binary Yes/No columns -> 1/0
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn"
]

# Service columns (Yes / No / No phone service / No internet service) -> 1/0
SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

# Columns for One-Hot Encoding
OHE_COLS = ["InternetService", "Contract", "PaymentMethod"]


# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file: {ext}")
    print(f"[OK] Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -- Step 1: Rename columns to standard names --
    df.rename(columns=RENAME_MAP, inplace=True)

    # -- Step 2: Keep only model-relevant columns --
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    print(f"[OK] Kept {len(available)} columns: {available}")

    # -- Step 3: Convert ALL columns to plain Python str for safety (pandas 3.0) --
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # -- Step 4: Convert TotalCharges to numeric --
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nulls = df["TotalCharges"].isnull().sum()
    print(f"[!] TotalCharges: {n_nulls} blank values -> NaN")

    # -- Step 5: Fill NaN with median --
    median_val = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
    print(f"[OK] Filled TotalCharges NaN with median: {median_val:.2f}")

    # -- Step 6: Check remaining nulls & dupes --
    print(f"[OK] Remaining nulls: {df.isnull().sum().sum()}")
    dupes = df.duplicated().sum()
    if dupes > 0:
        df.drop_duplicates(inplace=True)
        print(f"[!] Removed {dupes} duplicate rows")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure all string-like columns are plain Python str (pandas 3.0 compat)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # -- 1. Encode gender --
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    print("[OK] Encoded gender: Male=1, Female=0")

    # -- 2. SeniorCitizen: handles both 'Yes'/'No' and '0'/'1' strings --
    df["SeniorCitizen"] = df["SeniorCitizen"].map(
        {"Yes": 1, "No": 0, "1": 1, "0": 0}
    )
    print("[OK] Encoded SeniorCitizen")

    # -- 3. Binary Yes/No columns --
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    print(f"[OK] Encoded binary columns: {BINARY_COLS}")

    # -- 4. Service columns (Yes / No / No phone service / No internet service) --
    for col in SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip() == "Yes" else 0)
    print(f"[OK] Encoded service columns: {SERVICE_COLS}")

    # -- 5. One-Hot Encode multi-class columns --
    ohe_present = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_present, drop_first=False)
    print(f"[OK] One-Hot Encoded: {ohe_present}")

    # -- 6. Convert bool columns to int --
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # -- 7. SAFETY: force every column to numeric; drop any that can't convert --
    for col in list(df.columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.isna().sum() == 0:
                df[col] = converted
            else:
                print(f"[WARN] Dropping un-encodable column: '{col}'")
                df.drop(columns=[col], inplace=True)

    # -- 8. Fill any NaN introduced by failed maps (safety net) --
    if df.isnull().sum().sum() > 0:
        print(f"[WARN] Filling {df.isnull().sum().sum()} NaN values from failed encoding with 0")
        df = df.fillna(0)

    print(f"[OK] All columns numeric. Final shape: {df.shape}")
    return df


def get_X_y(df: pd.DataFrame):
    if "Churn" not in df.columns:
        raise ValueError("'Churn' column not found!")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print(f"\n[OK] Feature matrix X: {X.shape}")
    print(f"[OK] Target vector  y: {y.shape}")
    print(f"[OK] Churn distribution:\n{y.value_counts()}")
    print(f"     Churn rate: {y.mean() * 100:.2f}%")
    return X, y


def run_preprocessing_pipeline(filepath: str):
    print("=" * 60)
    print("  CUSTOMER CHURN -- DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    df          = load_data(filepath)
    df_clean    = clean_data(df)
    df_encoded  = encode_features(df_clean)
    X, y        = get_X_y(df_encoded)

    print("\n[OK] Preprocessing complete!")
    print("=" * 60)
    return X, y, df_encoded


# ─────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv"
    )
    X, y, df_encoded = run_preprocessing_pipeline(DATA_PATH)
    print(f"\nFeature columns ({len(X.columns)}):")
    for col in X.columns:
        print(f"  - {col}")
