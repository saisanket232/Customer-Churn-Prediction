"""
app.py
======
Customer Churn Prediction — Streamlit Deployment App

A production-style web app that lets users input customer details
and get an instant churn prediction with probability score.

Run with:
    streamlit run deployment/app.py

Author: Data Science Portfolio Project
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | Telco AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
  }

  /* Main card */
  .main-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
  }

  /* Header */
  .hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0.3rem;
  }
  .hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
  }

  /* Result cards */
  .churn-card {
    background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
    border: 2px solid #ff4b4b;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    animation: pulse 1.5s infinite;
  }
  .stay-card {
    background: linear-gradient(135deg, #00d4aa22, #00d4aa44);
    border: 2px solid #00d4aa;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
  }
  .result-emoji { font-size: 4rem; }
  .result-title { font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem; }
  .result-sub   { font-size: 1rem;  color: #cbd5e1; margin-top: 0.4rem; }

  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(255,75,75,0.4); }
    70%  { box-shadow: 0 0 0 12px rgba(255,75,75,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,75,75,0); }
  }

  /* Metric tiles */
  .metric-tile {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin: 0.3rem 0;
  }
  .metric-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(90deg, #6d28d9, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
    cursor: pointer;
  }
  .stButton > button:hover {
    background: linear-gradient(90deg, #7c3aed, #6366f1);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
  }

  /* Divider */
  .section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0;
  }
  .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #cbd5e1 !important;
    font-weight: 500;

  }

}
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ───────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
COLS_PATH   = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    with open(COLS_PATH,   "rb") as f: cols   = pickle.load(f)
    return model, scaler, cols

try:
    model, scaler, feature_cols = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📡 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered retention intelligence for telecom companies</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model not found. Please run `python src/train_model.py` first to train and save the model.")
    st.stop()


# ── Sidebar — Input Panel ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Customer Profile")
    st.markdown("Fill in the customer details below:")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Demographics
    st.markdown("### 👤 Demographics")
    gender         = st.selectbox("Gender",         ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner",    ["No", "Yes"])
    dependents     = st.selectbox("Has Dependents", ["No", "Yes"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Account
    st.markdown("### 📋 Account Info")
    tenure            = st.slider("Tenure (months)", 0, 72, 12)
    contract          = st.selectbox("Contract Type",   ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method    = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Services
    st.markdown("### 📶 Services")
    phone_service   = st.selectbox("Phone Service",    ["No", "Yes"])
    multiple_lines  = st.selectbox("Multiple Lines",   ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security",  ["No", "Yes"])
    online_backup   = st.selectbox("Online Backup",    ["No", "Yes"])
    device_protect  = st.selectbox("Device Protection",["No", "Yes"])
    tech_support    = st.selectbox("Tech Support",     ["No", "Yes"])
    streaming_tv    = st.selectbox("Streaming TV",     ["No", "Yes"])
    streaming_movies= st.selectbox("Streaming Movies", ["No", "Yes"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Charges
    st.markdown("### 💰 Billing")
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",    0.0, 9000.0,
                                       float(tenure * monthly_charges), step=10.0)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Churn")


# ── Build Input Row ─────────────────────────────────────────────────────────────
def build_input(feature_cols):
    """Build a single-row DataFrame matching the training feature columns."""
    row = {col: 0 for col in feature_cols}

    # Numeric
    row["tenure"]         = tenure
    row["MonthlyCharges"] = monthly_charges
    row["TotalCharges"]   = total_charges

    # Binary
    row["gender"]          = 1 if gender == "Male" else 0
    row["SeniorCitizen"]   = 1 if senior_citizen == "Yes" else 0
    row["Partner"]         = 1 if partner == "Yes" else 0
    row["Dependents"]      = 1 if dependents == "Yes" else 0
    row["PhoneService"]    = 1 if phone_service == "Yes" else 0
    row["MultipleLines"]   = 1 if multiple_lines == "Yes" else 0
    row["OnlineSecurity"]  = 1 if online_security == "Yes" else 0
    row["OnlineBackup"]    = 1 if online_backup == "Yes" else 0
    row["DeviceProtection"]= 1 if device_protect == "Yes" else 0
    row["TechSupport"]     = 1 if tech_support == "Yes" else 0
    row["StreamingTV"]     = 1 if streaming_tv == "Yes" else 0
    row["StreamingMovies"] = 1 if streaming_movies == "Yes" else 0
    row["PaperlessBilling"]= 1 if paperless_billing == "Yes" else 0

    # One-Hot: InternetService
    for suffix in ["DSL", "Fiber optic", "No"]:
        col = f"InternetService_{suffix}"
        if col in row:
            row[col] = 1 if internet_service == suffix else 0

    # One-Hot: Contract
    for suffix in ["Month-to-month", "One year", "Two year"]:
        col = f"Contract_{suffix}"
        if col in row:
            row[col] = 1 if contract == suffix else 0

    # One-Hot: PaymentMethod
    for suffix in ["Electronic check", "Mailed check",
                   "Bank transfer (automatic)", "Credit card (automatic)"]:
        col = f"PaymentMethod_{suffix}"
        if col in row:
            row[col] = 1 if payment_method == suffix else 0

    return pd.DataFrame([row])[feature_cols]


# ── Main Panel ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1])

with col_left:
    
    st.markdown("### 📊 Customer Summary")

    # Summary table
    summary_data = {
        "Field": [
            "Gender", "Senior Citizen", "Tenure",
            "Contract", "Internet Service",
            "Monthly Charges", "Total Charges", "Payment Method"
        ],
        "Value": [
            gender, senior_citizen, f"{tenure} months",
            contract, internet_service,
            f"${monthly_charges:.2f}", f"${total_charges:.2f}", payment_method
        ]
    }
    st.table(pd.DataFrame(summary_data).set_index("Field"))

    # Churn risk factors
    st.markdown("#### ⚠️ Risk Indicators")
    risks = []
    if tenure < 12:          risks.append("🔴 Short tenure (< 12 months)")
    if contract == "Month-to-month": risks.append("🔴 Month-to-month contract")
    if internet_service == "Fiber optic": risks.append("🟡 Fiber optic (higher churn rate)")
    if monthly_charges > 70: risks.append("🟡 High monthly charges")
    if payment_method == "Electronic check": risks.append("🟡 Electronic check payment")
    if online_security == "No": risks.append("🟡 No online security")
    if tech_support == "No":    risks.append("🟡 No tech support")

    if risks:
        for r in risks:
            st.markdown(f"- {r}")
    else:
        st.markdown("✅ No major risk factors detected.")
    

with col_right:
    
    st.markdown("### 🤖 Prediction Result")

    if predict_btn:
        input_df = build_input(feature_cols)
        input_arr = input_df.values  # RF doesn't need scaling

        prediction  = model.predict(input_arr)[0]
        probability = model.predict_proba(input_arr)[0]
        churn_prob  = probability[1] * 100
        stay_prob   = probability[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="churn-card">
              <div class="result-emoji">🚨</div>
              <div class="result-title" style="color:#ff4b4b;">Customer Will CHURN</div>
              <div class="result-sub">This customer is likely to leave.</div>
              <br>
              <div style="font-size:2rem;font-weight:700;color:#ff4b4b;">{churn_prob:.1f}%</div>
              <div style="color:#94a3b8;">Churn Probability</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 **Retention Action:** Offer a discounted long-term contract or loyalty reward to retain this customer.")
        else:
            st.markdown(f"""
            <div class="stay-card">
              <div class="result-emoji">✅</div>
              <div class="result-title" style="color:#00d4aa;">Customer Will STAY</div>
              <div class="result-sub">This customer is likely to remain.</div>
              <br>
              <div style="font-size:2rem;font-weight:700;color:#00d4aa;">{stay_prob:.1f}%</div>
              <div style="color:#94a3b8;">Retention Probability</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("✅ **Upsell Opportunity:** This loyal customer is a good candidate for premium service upgrades.")

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📈 Probability Breakdown")
        prob_df = pd.DataFrame({
            "Outcome": ["Will Stay", "Will Churn"],
            "Probability (%)": [stay_prob, churn_prob]
        })
        st.bar_chart(prob_df.set_index("Outcome"))

    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem 1rem; color:#64748b;'>
          <div style='font-size:4rem;'>🔮</div>
          <div style='font-size:1.1rem; margin-top:1rem;'>
            Fill in the customer details in the sidebar<br>and click <strong>Predict Churn</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
   


# ── Model Info Footer ────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-tile"><div class="metric-label">Algorithm</div><div class="metric-value" style="font-size:1rem;">Random Forest</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-tile"><div class="metric-label">Trees</div><div class="metric-value">200</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-tile"><div class="metric-label">Dataset</div><div class="metric-value" style="font-size:1rem;">IBM Telco</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-tile"><div class="metric-label">Version</div><div class="metric-value" style="font-size:1rem;">v1.0</div></div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.8rem; margin-top:1.5rem;'>
  Built with ❤️ using Scikit-Learn & Streamlit &nbsp;|&nbsp; IBM Telco Customer Churn Dataset
</div>
""", unsafe_allow_html=True)
