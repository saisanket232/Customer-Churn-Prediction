"""
dashboard.py
============
Customer Churn Prediction — Interactive Analytics Dashboard

A comprehensive data visualization dashboard showing:
  - Churn distribution
  - Contract-wise churn analysis
  - Monthly charges analysis
  - Tenure vs Churn patterns
  - Service adoption impact
  - Correlation heatmap

Run with:
    streamlit run dashboard/dashboard.py

Author: Data Science Portfolio Project
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: linear-gradient(160deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
  }

  .dashboard-header {
    background: linear-gradient(90deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
  }
  .dashboard-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .dashboard-sub {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.5rem;
  }

  /* KPI cards */
  .kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: all 0.3s ease;
  }
  .kpi-card:hover {
    border-color: rgba(99,102,241,0.5);
    background: rgba(99,102,241,0.08);
    transform: translateY(-3px);
  }
  .kpi-icon  { font-size: 2rem; }
  .kpi-value { font-size: 2rem; font-weight: 800; color: #a78bfa; margin: 0.3rem 0; }
  .kpi-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }

  /* Chart cards */
  .chart-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1.5rem;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 0.4rem;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: white !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    base = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(base, "data", "Telco-Customer-Churn.csv")
    xlsx_path = os.path.join(base, "Telco_customer_churn.xlsx")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        st.error("Dataset not found. Run the data conversion script first.")
        st.stop()

    # ADD THESE LINES HERE
    print("Columns loaded:")
    print(df.columns.tolist())

    # Cleaning
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "", regex=False)

    print("Processed columns:")
    print(df.columns.tolist())

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    else:
        st.error(f"TotalCharges column not found.\nColumns are:\n{df.columns.tolist()}")
        st.stop()

    return df
df = load_raw_data()
churned = df[df["ChurnLabel"] == "Yes"]
stayed = df[df["ChurnLabel"] == "No"]

# ── Plotly theme ───────────────────────────────────────────────────────────────
COLORS = {
    "churn"  : "#f87171",
    "stay"   : "#34d399",
    "purple" : "#a78bfa",
    "blue"   : "#60a5fa",
    "orange" : "#fb923c",
    "pink"   : "#f472b6",
}
DARK_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)"


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dashboard-header">
  <div class="dashboard-title">📊 Customer Churn Analytics Dashboard</div>
  <div class="dashboard-sub">IBM Telco Customer Dataset — Interactive Business Intelligence</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
total        = len(df)
n_churn      = len(churned)
churn_rate   = n_churn / total * 100
avg_tenure = df["TenureMonths"].mean()
avg_monthly  = df["MonthlyCharges"].mean()
monthly_loss = churned["MonthlyCharges"].sum()

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">👥</div>
      <div class="kpi-value">{total:,}</div>
      <div class="kpi-label">Total Customers</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">🚪</div>
      <div class="kpi-value" style="color:#f87171;">{n_churn:,}</div>
      <div class="kpi-label">Churned Customers</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">📉</div>
      <div class="kpi-value" style="color:#fb923c;">{churn_rate:.1f}%</div>
      <div class="kpi-label">Churn Rate</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">⏱️</div>
      <div class="kpi-value">{avg_tenure:.0f}mo</div>
      <div class="kpi-label">Avg Tenure</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">💸</div>
      <div class="kpi-value" style="color:#f87171;">${monthly_loss:,.0f}</div>
      <div class="kpi-label">Monthly Revenue Lost</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🍩 Churn Overview",
    "📝 Contract Analysis",
    "💰 Charges Analysis",
    "⏳ Tenure Patterns",
    "🔥 Correlation Heatmap",
])


# ─────────────────────────────────────────────
# TAB 1: CHURN OVERVIEW
# ─────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Donut chart
        churn_counts = df["ChurnLabel"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map(
            {"Yes": "Churned", "No": "Retained"}
        )
        fig = px.pie(
            churn_counts, names="Label", values="Count",
            hole=0.55,
            color="Label",
            color_discrete_map={"Churned": COLORS["churn"], "Retained": COLORS["stay"]},
            title="📊 Customer Churn Distribution",
            template=DARK_TEMPLATE,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont_size=13, pull=[0.03, 0])
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          margin=dict(t=50, b=20, l=20, r=20),
                          showlegend=True,
                          annotations=[dict(text=f"<b>{churn_rate:.1f}%</b><br>Churn",
                                           x=0.5, y=0.5, font_size=16,
                                           showarrow=False, font_color=COLORS["churn"])])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Churn by Internet Service
        internet_churn = df.groupby(["InternetService", "ChurnLabel"]).size().reset_index(name="Count")
        fig2 = px.bar(
            internet_churn, x="InternetService", y="Count", color="ChurnLabel",
            barmode="group",
            color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
            title="📶 Churn by Internet Service",
            template=DARK_TEMPLATE,
            labels={"InternetService": "Internet Service", "ChurnLabel": "Churn"},
        )
        fig2.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                           margin=dict(t=50, b=20, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Churn by senior citizen & gender
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        senior_churn = df.groupby(["SeniorCitizen", "ChurnLabel"]).size().reset_index(name="Count")
        senior_churn["Senior"] = senior_churn["SeniorCitizen"].map(
        {"Yes": "Senior", "No": "Not Senior"}
        )
        fig3 = px.bar(
            senior_churn, x="Senior", y="Count", color="ChurnLabel",
            barmode="stack",
            color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
            title="👴 Churn by Senior Citizen Status",
            template=DARK_TEMPLATE,
        )
        fig3.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                           margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        gender_churn = df.groupby(["Gender", "ChurnLabel"]).size().reset_index(name="Count")
        fig4 = px.bar(
            gender_churn, x="Gender", y="Count", color="ChurnLabel",
            barmode="group",
            color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
            title="⚧ Churn by Gender",
            template=DARK_TEMPLATE,
        )
        fig4.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                           margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2: CONTRACT ANALYSIS
# ─────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        contract_churn = df.groupby(["Contract", "ChurnLabel"]).size().reset_index(name="Count")
        fig = px.bar(
            contract_churn, x="Contract", y="Count", color="ChurnLabel",
            barmode="group",
            color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
            title="📝 Churn by Contract Type",
            template=DARK_TEMPLATE,
            text_auto=True,
        )
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Churn rate by contract
        rate_df = df.groupby("Contract").apply(
            lambda x: (x["ChurnLabel"] == "Yes").mean() * 100
        ).reset_index()
        rate_df.columns = ["Contract", "Churn Rate (%)"]
        fig2 = px.bar(
            rate_df, x="Contract", y="Churn Rate (%)",
            color="Churn Rate (%)",
            color_continuous_scale=["#34d399", "#fb923c", "#f87171"],
            title="📉 Churn Rate % by Contract Type",
            template=DARK_TEMPLATE,
            text_auto=".1f",
        )
        fig2.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                           margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    payment_churn = df.groupby(["PaymentMethod", "ChurnLabel"]).size().reset_index(name="Count")
    fig3 = px.bar(
        payment_churn, x="Count", y="PaymentMethod", color="ChurnLabel",
        barmode="group", orientation="h",
        color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
        title="💳 Churn by Payment Method",
        template=DARK_TEMPLATE,
    )
    fig3.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                       margin=dict(t=50, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 3: CHARGES ANALYSIS
# ─────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="MonthlyCharges", color="ChurnLabel",
            nbins=40, barmode="overlay", opacity=0.75,
            color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
            title="💰 Monthly Charges Distribution",
            template=DARK_TEMPLATE,
            labels={"MonthlyCharges": "Monthly Charges ($)"},
        )
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig2 = px.box(
            df, x="ChurnLabel", y="MonthlyCharges", color="ChurnLabel",
            color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
            title="📦 Monthly Charges Boxplot by Churn",
            template=DARK_TEMPLATE,
            points="outliers",
        )
        fig2.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                           margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig3 = px.scatter(
        df, x="MonthlyCharges", y="TotalCharges",
        color="ChurnLabel", opacity=0.5, size_max=6,
        color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
        title="🔵 Monthly vs Total Charges (Scatter)",
        template=DARK_TEMPLATE,
        hover_data=["TenureMonths"],
    )
    fig3.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                       margin=dict(t=50, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 4: TENURE PATTERNS
# ─────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="TenureMonths", color="ChurnLabel",
            nbins=30, barmode="overlay", opacity=0.75,
            color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["stay"]},
            title="⏳ Tenure Distribution by Churn",
            template=DARK_TEMPLATE,
            labels={"TenureMonths": "Tenure (months)"},
        )
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Tenure bucket analysis
        df["TenureBucket"] = pd.cut(
            df["TenureMonths"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-12 mo", "12-24 mo", "24-48 mo", "48-72 mo"]
        )
        bucket_rate = df.groupby("TenureBucket").apply(
            lambda x: (x["ChurnLabel"] == "Yes").mean() * 100
        ).reset_index()
        bucket_rate.columns = ["Tenure Bucket", "Churn Rate (%)"]
        fig2 = px.bar(
            bucket_rate, x="Tenure Bucket", y="Churn Rate (%)",
            color="Churn Rate (%)",
            color_continuous_scale=["#34d399", "#fb923c", "#f87171"],
            title="📊 Churn Rate by Tenure Group",
            template=DARK_TEMPLATE,
            text_auto=".1f",
        )
        fig2.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                           margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 5: CORRELATION HEATMAP
# ─────────────────────────────────────────────
with tab5:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("#### 🔥 Feature Correlation Heatmap")
    st.caption("Shows Pearson correlation between numeric features. Values close to ±1 indicate strong correlation.")

    # Encode for numeric correlation
    df_num = df.copy()
    binary_map = {"Yes": 1, "No": 0}
    df_num["Churn"]          = df_num["ChurnLabel"].map(binary_map)
    df_num["Gender"]         = df_num["Gender"].map({"Male": 1, "Female": 0})
    df_num["Partner"]        = df_num["Partner"].map(binary_map)
    df_num["Dependents"]     = df_num["Dependents"].map(binary_map)
    df_num["PhoneService"]   = df_num["PhoneService"].map(binary_map)
    df_num["PaperlessBilling"]= df_num["PaperlessBilling"].map(binary_map)
    df_num["SeniorCitizen"] = df_num["SeniorCitizen"].map(binary_map)
    num_cols = ["TenureMonths", "MonthlyCharges", "TotalCharges",
                "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "PaperlessBilling", "Churn"]

    corr = df_num[num_cols].corr()

    fig = px.imshow(
        corr.round(2),
        text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation Heatmap",
        template=DARK_TEMPLATE,
    )
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      margin=dict(t=50, b=10, l=10, r=10),
                      height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Insights:**
    - 📈 `tenure` ↑ → **Churn** ↓ (longer customers churn less)
    - 💰 `MonthlyCharges` ↑ → **Churn** ↑ (expensive plans drive churn)
    - 🔗 `TotalCharges` strongly correlated with `tenure` (multicollinearity)
    - 👴 `SeniorCitizen` slightly positively correlated with churn
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; margin-top:2rem; padding:1rem;
     border-top:1px solid rgba(255,255,255,0.06);'>
  📊 Customer Churn Analytics Dashboard &nbsp;|&nbsp;
  IBM Telco Dataset &nbsp;|&nbsp;
  Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
