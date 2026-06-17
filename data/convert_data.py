"""
convert_data.py
===============
Convert the Excel dataset to CSV format and save to data/ folder.
Run this script first before training the model.

Usage:
    python data/convert_data.py
"""

import os
import pandas as pd

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
XLSX_PATH = os.path.join(BASE_DIR, "Telco_customer_churn.xlsx")
CSV_DIR   = os.path.join(BASE_DIR, "data")
CSV_PATH  = os.path.join(CSV_DIR, "Telco-Customer-Churn.csv")

os.makedirs(CSV_DIR, exist_ok=True)

print("📂 Converting Excel → CSV ...")
print(f"   Source : {XLSX_PATH}")
print(f"   Output : {CSV_PATH}")

df = pd.read_excel(XLSX_PATH)

print(f"✅ Loaded Excel: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")

df.to_csv(CSV_PATH, index=False)
print(f"\n✅ Saved CSV → {CSV_PATH}")
print("   You can now run: python src/train_model.py")
