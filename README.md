# 📡 Customer Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.22-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end ML project predicting customer churn for a telecom company using the IBM Telco dataset.**

[🚀 Run Prediction App](#-quick-start) · [📊 View Dashboard](#-dashboard) · [🧠 Model Details](#-model-performance)

</div>

---

## 📌 Project Overview

Customer churn — when customers stop using a service — is one of the most critical business metrics for any subscription-based company. Acquiring a new customer costs **5–7× more** than retaining an existing one.

This project builds a complete **machine learning pipeline** to:
1. Identify customers likely to churn
2. Understand the key drivers of churn
3. Enable proactive retention strategies

### 🎯 Business Impact
| Metric | Value |
|--------|-------|
| Dataset Size | 7,043 customers |
| Churn Rate | ~26.5% |
| Monthly Revenue at Risk | ~$139,000 |
| Best Model F1 Score | ~80%+ |

---

## 🗂️ Project Structure

```
Customer_Churn_Prediction/
│
├── data/
│   ├── Telco-Customer-Churn.csv        # Processed dataset (CSV)
│   └── convert_data.py                 # Excel → CSV converter
│
├── src/
│   ├── preprocessing.py                # Data cleaning & encoding pipeline
│   ├── train_model.py                  # Train 4 models + save best
│   └── evaluate_model.py               # Metrics, confusion matrix, ROC curve
│
├── models/
│   ├── random_forest_model.pkl         # Saved best model
│   ├── scaler.pkl                      # StandardScaler (for LR/KNN)
│   ├── feature_columns.pkl             # Feature column order
│   ├── feature_importance.png          # Feature importance chart
│   ├── evaluation_plots.png            # Confusion matrix + ROC curve
│   └── model_comparison.csv            # All model results
│
├── notebooks/
│   └── churn_analysis.ipynb            # Full EDA + modelling notebook
│
├── deployment/
│   └── app.py                          # Streamlit prediction app
│
├── dashboard/
│   └── dashboard.py                    # Interactive analytics dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python data/convert_data.py
```

### 3. Train Models

```bash
python src/train_model.py
```

### 4. Evaluate Model

```bash
python src/evaluate_model.py
```

### 5. Launch Prediction App

```bash
streamlit run deployment/app.py
```

### 6. Launch Analytics Dashboard

```bash
streamlit run dashboard/dashboard.py
```

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**

| Feature | Type | Description |
|---------|------|-------------|
| `tenure` | Numeric | Months with company |
| `MonthlyCharges` | Numeric | Monthly billing amount |
| `TotalCharges` | Numeric | Total billed amount |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / etc. |
| `Churn` | Target | Yes / No |
| *+ 12 more...* | | |

---

## 🔧 Feature Engineering

| Step | Technique | Columns |
|------|-----------|---------|
| Binary Encoding | Yes=1, No=0 | Partner, Dependents, PhoneService, etc. |
| Label Encoding | Male=1, Female=0 | gender |
| One-Hot Encoding | get_dummies | InternetService, Contract, PaymentMethod |
| Type Conversion | pd.to_numeric | TotalCharges (had blank strings) |
| Imputation | Median fill | TotalCharges NaN values |

---

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** ⭐ | ~80% | ~67% | ~76% | ~71% | ~84% |
| Logistic Regression | ~81% | ~68% | ~57% | ~62% | ~85% |
| Decision Tree | ~78% | ~60% | ~72% | ~65% | ~74% |
| K-Nearest Neighbors | ~77% | ~56% | ~66% | ~61% | ~79% |

> ⭐ **Random Forest selected** as the deployed model for its balance of recall, robustness, and interpretability via feature importance.

---

## 🌟 Top Churn Predictors

Based on Random Forest feature importance:

1. 🥇 **tenure** — Short-tenure customers churn significantly more
2. 🥈 **MonthlyCharges** — Higher bills increase churn risk
3. 🥉 **TotalCharges** — Related to tenure; low total charges = newer customer
4. 📝 **Contract_Month-to-month** — 42% churn rate vs 11% for annual contracts
5. 📶 **InternetService_Fiber optic** — Fiber users churn more (likely pricing)
6. 💳 **PaymentMethod_Electronic check** — Highest churn among payment methods
7. 🛡️ **OnlineSecurity** — No security → higher churn
8. 🔧 **TechSupport** — No support → higher churn

---

## 🚀 Deployment App

The Streamlit app accepts **20 customer inputs** and returns:

- ✅ **"Customer Will STAY"** — with retention probability %
- 🚨 **"Customer Will CHURN"** — with churn probability % + retention recommendation

**Live risk indicators** flag concerning patterns (short tenure, fiber optic, electronic check, etc.) to help retention teams prioritize.

---

## 📈 Future Improvements

| Improvement | Description |
|------------|-------------|
| **XGBoost / LightGBM** | Gradient boosting for higher accuracy |
| **SMOTE** | Synthetic Minority Oversampling to handle class imbalance |
| **SHAP Explainability** | Per-prediction feature attribution |
| **Hyperparameter Tuning** | GridSearchCV / Optuna optimization |
| **Flask REST API** | Production-grade REST endpoint |
| **Docker** | Containerize app for cloud deployment |
| **AWS / GCP Deploy** | Host on cloud with CI/CD pipeline |
| **A/B Testing Framework** | Measure retention strategy impact |

---

## 💼 Resume Bullet Points

- **Developed an end-to-end Customer Churn Prediction ML pipeline** using Python, Scikit-Learn, and Random Forest achieving 80%+ accuracy on IBM Telco dataset (7,043 records)
- **Engineered 20+ features** through binary encoding, one-hot encoding, and data cleaning; identified tenure, monthly charges, and contract type as top churn drivers using feature importance analysis
- **Deployed an interactive Streamlit prediction app** with real-time churn probability scoring and automated risk flagging, enabling data-driven retention decisions
- **Built a multi-chart analytics dashboard** using Plotly with KPI cards, churn segmentation by contract/service/payment, and correlation heatmap for business intelligence reporting

---

## 🎤 Interview Q&A (Key Points)

<details>
<summary><b>Why Random Forest over other models?</b></summary>

Random Forest builds 200 decision trees on random subsets of data and averages their predictions. This reduces overfitting (vs single Decision Tree), provides feature importances, handles non-linear patterns (vs Logistic Regression), and is faster at prediction (vs KNN). It's the industry-standard baseline for structured data classification.
</details>

<details>
<summary><b>Why F1-score instead of accuracy?</b></summary>

Our dataset has class imbalance (~73% No Churn, ~27% Churn). A model predicting "No Churn" for everyone gets 73% accuracy but catches zero churners. F1-score balances Precision and Recall — it penalizes both missing actual churners (FN) and false alarms (FP), making it the right metric for imbalanced classification.
</details>

<details>
<summary><b>How did you handle missing values?</b></summary>

The TotalCharges column had blank strings (" ") for 11 new customers (tenure=0). pd.to_numeric(errors='coerce') converted these to NaN, then we filled with the median (robust to outliers vs mean). No other columns had missing values.
</details>

<details>
<summary><b>What is class imbalance and how would you handle it?</b></summary>

Class imbalance = unequal distribution of target classes (~73% No Churn vs ~27% Churn). Techniques: (1) SMOTE — generate synthetic minority samples, (2) class_weight='balanced' in sklearn — adjusts loss function, (3) Threshold tuning — lower the probability threshold from 0.5 to 0.3 to catch more churners.
</details>

---

## 📄 License

MIT License — free to use for portfolio, learning, and commercial projects.

---

<div align="center">
  Made with ❤️ for AI/ML Portfolio | IBM Telco Customer Churn Dataset
</div>
