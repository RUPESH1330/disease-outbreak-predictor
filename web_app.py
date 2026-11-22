"""
==============================================================================
  DISEASE OUTBREAK RISK PREDICTION SYSTEM - WEB DASHBOARD
  Interactive Streamlit App for Indian States & Disease Risk Forecasting
==============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Import core prediction classes from disease_predictor
from disease_predictor import (
    INDIAN_STATES,
    DISEASES,
    STATE_PROFILES,
    DEFAULT_PROFILE,
    DISEASE_BASELINES,
    DataManager,
    ModelTrainer,
    TrendForecaster,
    NewsSentimentSimulator,
    DATA_PATH
)

st.set_page_config(
    page_title="Disease Outbreak Risk Predictor | India",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E293B;
    }
    .increase-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    .decrease-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    .stable-badge {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    .param-box {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 15px;
        border: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)

# Cache using file modification time so edits to CSV reload automatically
@st.cache_resource
def load_and_train(csv_mtime: float):
    data_mgr = DataManager(DATA_PATH)
    df = data_mgr.load()
    trainer = ModelTrainer(df)
    trainer.train()
    forecaster = TrendForecaster(df)
    return data_mgr, trainer, forecaster, df

csv_mtime = os.path.getmtime(DATA_PATH)
data_mgr, trainer, forecaster, df = load_and_train(csv_mtime)
news_sim = NewsSentimentSimulator()

# Header
st.markdown('<div class="main-header">🦠 Disease Outbreak Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Outbreak Probability & Future Trajectory (Increase/Decrease) for Indian States</div>', unsafe_allow_html=True)

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

now = datetime.now()
current_month_idx = now.month - 1
current_year = now.year

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR: MANUAL USER INPUTS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Input Parameters")

# 1. State & Disease
st.sidebar.markdown("### 📍 Location & Disease")
selected_state = st.sidebar.selectbox(
    "Select Indian State / UT:",
    sorted(INDIAN_STATES),
    index=sorted(INDIAN_STATES).index("Kerala") if "Kerala" in sorted(INDIAN_STATES) else 0
)

selected_disease = st.sidebar.selectbox(
    "Select Disease:",
    sorted(DISEASES),
    index=sorted(DISEASES).index("Dengue") if "Dengue" in sorted(DISEASES) else 0
)

# 2. Time Period (Month & Year)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Target Time Period")
col_m, col_y = st.sidebar.columns(2)
with col_m:
    selected_month_name = st.selectbox(
        "Month:",
        MONTH_NAMES,
        index=current_month_idx
    )
    month_num = MONTH_NAMES.index(selected_month_name) + 1

with col_y:
    year_val = st.number_input(
        "Year:",
        min_value=2020,
        max_value=2035,
        value=current_year,
        step=1
    )

# 3. Manual Surveillance Inputs (Search Trends & Health News)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Surveillance Signals (Manual Input)")
st.sidebar.caption("Adjust Search Trends and Health News to see immediate changes in risk percentage:")

# Sliders with direct numeric feedback
search_score = st.sidebar.slider(
    "🔍 Search Trend Value (0 - 100):",
    min_value=0,
    max_value=100,
    value=65,
    step=1,
    help="Higher values indicate surging online searches for fever, symptoms, and outbreak keywords."
)

news_score = st.sidebar.slider(
    "📰 Health News Mentions Score (0 - 100):",
    min_value=0,
    max_value=100,
    value=60,
    step=1,
    help="Higher values indicate verified digital health news coverage of disease clusters."
)

st.sidebar.markdown(f"""
<div style="font-size:0.85rem; color:#475569; background:#F1F5F9; padding:8px; border-radius:6px; margin-top:10px;">
<b>Live Inputs Selected:</b><br>
• Search Trend: <b>{search_score} / 100</b><br>
• News Mentions: <b>{news_score} / 100</b><br>
• Month & Year: <b>{selected_month_name} {int(year_val)}</b>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
profile = STATE_PROFILES.get(selected_state, DEFAULT_PROFILE)
baseline_cases = DISEASE_BASELINES.get(selected_disease, 10.0)

c_risk = profile['climate_risk']
san = profile['sanitation_score']
hc = profile['healthcare_index']

composite = (
    search_score * 0.42 + news_score * 0.36 + (search_score * news_score / 100.0) * 0.08
    + c_risk * 12.0 + (1.0 - san) * 6.0 - hc * 6.0
)

fv = {
    'search_trend_score': float(search_score),
    'news_coverage_score': float(news_score),
    'cases_per_lakh': baseline_cases,
    'climate_risk': c_risk,
    'sanitation_score': san,
    'healthcare_index': hc,
    'composite_score': composite,
    'sin_month': np.sin(2 * np.pi * month_num / 12),
    'cos_month': np.cos(2 * np.pi * month_num / 12),
    'trend_news_interaction': float(search_score * news_score / 100.0),
    'month': month_num,
    'year': year_val,
}

predicted_risk = trainer.predict(fv)
all_preds = trainer.predict_all(fv)

# Dynamic trajectory forecasting based on current signals & historical trends
hist_sub = df[(df['state'] == selected_state) & (df['disease'] == selected_disease)]
hist_avg_risk = hist_sub['risk_percentage'].mean() if len(hist_sub) > 0 else 45.0

# Future direction is driven by:
# 1. Surge intensity above/below seasonal baseline
# 2. Direction of media & search velocity
signal_intensity = (search_score * 0.55 + news_score * 0.45)
# If signal is elevated (>60), current risk is high and may either peak or decrease back towards baseline
# If signal is low (<40), risk is currently low and may increase towards seasonal baseline
# Seasonal factor
is_monsoon_target = 6 <= month_num <= 10
if signal_intensity >= 65:
    # High active surge: over 12 months, epidemic wave subsides post-intervention
    f6_risk = round(min(99.0, max(5.0, predicted_risk * (1.1 if is_monsoon_target else 0.82))), 2)
    f12_risk = round(min(99.0, max(5.0, (predicted_risk * 0.70 + hist_avg_risk * 0.30))), 2)
elif signal_intensity <= 35:
    # Low lull: over 12 months, moves back towards seasonal peak
    f6_risk = round(min(99.0, max(5.0, predicted_risk * 1.25)), 2)
    f12_risk = round(min(99.0, max(5.0, (predicted_risk * 0.60 + hist_avg_risk * 0.40))), 2)
else:
    f6_risk = round(min(99.0, max(5.0, (predicted_risk * 0.95 + hist_avg_risk * 0.05))), 2)
    f12_risk = round(min(99.0, max(5.0, hist_avg_risk)), 2)

diff_12m = f12_risk - predicted_risk
pct_change_12m = ((f12_risk - predicted_risk) / max(predicted_risk, 0.01)) * 100.0

if pct_change_12m > 3.0:
    traj_direction = "INCREASING"
    traj_color = "#DC2626"
    traj_arrow = "📈"
    traj_summary = (
        f"With Search Trends at {search_score}/100 and News Mentions at {news_score}/100, "
        f"outbreak risk in {selected_state} is projected to INCREASE by {abs(pct_change_12m):.1f}% over the next 12 months "
        f"towards the regional seasonal transmission peak."
    )
elif pct_change_12m < -3.0:
    traj_direction = "DECREASING"
    traj_color = "#16A34A"
    traj_arrow = "📉"
    traj_summary = (
        f"Current elevated Search Trends ({search_score}/100) and Health News ({news_score}/100) indicate an active peak. "
        f"Following containment and public health response, the outbreak risk is projected to DECREASE by {abs(pct_change_12m):.1f}% over the next 12 months."
    )
else:
    traj_direction = "STABLE"
    traj_color = "#CA8A04"
    traj_arrow = "➡️"
    traj_summary = (
        f"Outbreak risk is projected to remain STABLE around current levels ({predicted_risk:.1f}%) "
        f"with minimal variance expected over the coming quarters."
    )

confidence_level = int(min(95, max(60, 75 + (abs(search_score - 50) / 50.0) * 15)))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

# Query summary banner
st.markdown(f"""
<div class="param-box">
    <b>Selected Query:</b> 📍 <b>{selected_state}</b> &nbsp;|&nbsp; 
    🦠 Disease: <b>{selected_disease}</b> &nbsp;|&nbsp; 
    📅 Period: <b>{selected_month_name} {int(year_val)}</b> &nbsp;|&nbsp; 
    🔍 Search Trends: <b>{search_score}/100</b> &nbsp;|&nbsp; 
    📰 Health News: <b>{news_score}/100</b>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.3, 1.3, 1.4])

# 1. Current Risk Output
with col1:
    st.subheader("📊 Disease Risk Output")
    risk_color = (
        "#16A34A" if predicted_risk < 25 else
        "#CA8A04" if predicted_risk < 50 else
        "#DC2626" if predicted_risk < 75 else "#7C3AED"
    )
    status_text = (
        "LOW RISK" if predicted_risk < 25 else
        "MODERATE RISK" if predicted_risk < 50 else
        "HIGH RISK" if predicted_risk < 75 else "CRITICAL OUTBREAK RISK"
    )
    
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {risk_color};">
        <div style="font-size: 0.85rem; color: #64748B; font-weight: 600;">PREDICTED OUTBREAK RISK</div>
        <div class="metric-val" style="color: {risk_color};">{predicted_risk:.2f}%</div>
        <div style="font-weight: 700; color: {risk_color}; margin-top: 4px;">{status_text}</div>
        <div style="font-size: 0.85rem; color: #475569; margin-top: 8px;">
            For <b>{selected_state}</b> in <b>{selected_month_name} {int(year_val)}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Surveillance Signal Breakdown:**")
    st.write(f"- 🔍 Search Trend Volume: **{search_score} / 100**")
    st.write(f"- 📰 Health News Coverage: **{news_score} / 100**")
    st.write(f"- 🌡️ State Climate Vulnerability: **{profile['climate_risk']:.2f}**")
    st.write(f"- 🏥 State Healthcare Readiness: **{profile['healthcare_index']:.2f}**")

# 2. Future Trajectory (Increase or Decrease)
with col2:
    st.subheader("🔮 Future Trajectory Forecast")
    if traj_direction == "INCREASING":
        badge_html = f'<div class="increase-badge">{traj_arrow} RISK IS PROJECTED TO INCREASE</div>'
    elif traj_direction == "DECREASING":
        badge_html = f'<div class="decrease-badge">{traj_arrow} RISK IS PROJECTED TO DECREASE</div>'
    else:
        badge_html = f'<div class="stable-badge">{traj_arrow} RISK IS PROJECTED TO REMAIN STABLE</div>'
    
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top: 14px; font-size: 0.92rem; line-height: 1.5; color: #1E293B;">
        {traj_summary}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.write(f"• **Future Direction:** <b style='color:{traj_color};'>{traj_direction}</b>", unsafe_allow_html=True)
    st.write(f"• **6-Month Risk Projection:** `{f6_risk:.2f}%`")
    st.write(f"• **12-Month Risk Projection:** `{f12_risk:.2f}%`")
    st.write(f"• **12-Month Delta:** `{'+' if diff_12m > 0 else ''}{diff_12m:.2f}% ({'+' if pct_change_12m > 0 else ''}{pct_change_12m:.1f}%)`")
    st.write(f"• **Forecast Confidence:** `{confidence_level}%`")

# 3. Multi-Model Ensemble Comparison
with col3:
    st.subheader("🤖 Algorithm Comparison")
    model_df = pd.DataFrame(list(all_preds.items()), columns=["Algorithm", "Predicted Risk (%)"])
    fig_models = px.bar(
        model_df,
        x="Algorithm",
        y="Predicted Risk (%)",
        color="Algorithm",
        text_auto=".2f",
        color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B"]
    )
    fig_models.update_layout(height=260, margin=dict(l=15, r=15, t=15, b=15), showlegend=False)
    st.plotly_chart(fig_models, use_container_width=True)

st.markdown("---")

# Row 2: Future Trend Timeline & Feature Importance
c_left, c_right = st.columns([1.5, 1])

with c_left:
    st.subheader(f"📈 12-Month Outbreak Trajectory Timeline (From {selected_month_name} {int(year_val)})")
    
    timeline_labels = []
    offsets = [0, 2, 4, 6, 8, 10, 12]
    for off in offsets:
        tot_m = (month_num - 1) + off
        m_name = MONTH_NAMES[tot_m % 12][:3]
        y_name = int(year_val) + (tot_m // 12)
        timeline_labels.append(f"{m_name} '{str(y_name)[-2:]}")

    f0 = predicted_risk
    xs = [0, 6, 12]
    ys = [f0, f6_risk, f12_risk]
    poly_fit = np.polyfit(xs, ys, 2)
    eval_xs = np.array(offsets)
    eval_ys = np.maximum(0, np.minimum(100, np.polyval(poly_fit, eval_xs)))
    
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=timeline_labels,
        y=eval_ys,
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in eval_ys],
        textposition="top center",
        name="Projected Risk %",
        line=dict(color="#EF4444" if traj_direction == "INCREASING" else "#10B981", width=3),
        marker=dict(size=8)
    ))
    fig_traj.update_layout(
        xaxis_title="Future Month Timeline",
        yaxis_title="Outbreak Risk Percentage (%)",
        height=320,
        margin=dict(l=30, r=30, t=30, b=30)
    )
    st.plotly_chart(fig_traj, use_container_width=True)

with c_right:
    st.subheader("🧠 Key Predictive Signal Drivers")
    importance = trainer.feature_importance().head(6)
    if not importance.empty:
        imp_df = pd.DataFrame({
            "Feature": importance.index,
            "Importance": importance.values
        }).sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(height=320, margin=dict(l=15, r=15, t=15, b=15), showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

# Row 3: Actionable Health Recommendations
st.markdown("---")
st.subheader("🛡️ Public Health Action Plan")
rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("#### Recommended Interventions")
    if selected_disease == "Dengue":
        st.write("1. 🦟 **Vector Control**: Immediate fogging in high water-stagnation pockets.")
        st.write("2. 💧 **Source Reduction**: Weekly 'Dry Days' to empty domestic coolers and containers.")
        st.write("3. 🏥 **Diagnostic Supplies**: Stock NS1 and IgM ELISA kits in Primary Health Centres.")
    elif selected_disease == "Malaria":
        st.write("1. 🛏️ **Bed Nets**: Distribute Long-Lasting Insecticidal Nets (LLINs).")
        st.write("2. 🧪 **Rapid Testing**: Deploy Rapid Diagnostic Kits (RDTs) to rural health workers.")
        st.write("3. 💊 **Medication**: Ensure ACT (Artemisinin-based combination therapy) reserves.")
    elif selected_disease == "Cholera":
        st.write("1. 🚰 **Water Testing**: Daily chlorination and microbial surveillance of water mains.")
        st.write("2. 🧼 **WASH Protocols**: Community awareness drives on hand hygiene and boiled water.")
        st.write("3. 📦 **ORS & Zinc**: Pre-position ORS packets and IV fluids across clinics.")
    elif selected_disease == "COVID-19":
        st.write("1. 😷 **Infection Control**: Mask mandates in healthcare facilities and crowded indoor areas.")
        st.write("2. 🧬 **Genomic Surveillance**: INSACOG sequencing of positive sentinel samples.")
        st.write("3. 💉 **Vaccination**: Booster dose campaigns for vulnerable demographics.")
    else:
        st.write("1. 🔍 **Active Surveillance**: Sentinel hospital reporting and syndromic tracking.")
        st.write("2. 🧑‍⚕️ **Clinical Preparedness**: Dedicated isolation beds and standard clinical management protocols.")
        st.write("3. 📢 **Public Alerts**: Information dissemination via local health bulletins.")

with rec_col2:
    st.markdown("#### Surveillance Convergence Insights")
    st.info(f"""
    - **Search Trend Score:** **{search_score} / 100** (Surveillance Signal Weight: High)
    - **Health News Coverage Score:** **{news_score} / 100** (Epidemic Mention Weight: High)
    - **Target Period:** **{selected_month_name} {int(year_val)}**
    - **Regional Vulnerability:** Climate risk is **{profile['climate_risk']:.2f}**, Sanitation score is **{profile['sanitation_score']:.2f}**.
    - **Trajectory Projection:** Future risk is anticipated to **{traj_direction.lower()}**.
    """)
