# 🦠 Disease Outbreak Risk Prediction System (India)

A Machine Learning powered epidemiological surveillance and outbreak risk forecasting system for Indian States and Union Territories. It models disease vulnerability by integrating digital exhaust (Google Search Trends, digital health news media intensity), clinical baseline epidemiology, climate risk factors, sanitation levels, and healthcare infrastructure indices.

---

## 🌟 Key Features

1. **User Input:**
   - Prompts for any **Indian State / Union Territory** (e.g., *Kerala, Maharashtra, Delhi, Bihar, Assam, etc.*).
   - Prompts for any **Disease** (e.g., *Dengue, Malaria, Cholera, COVID-19, Typhoid, Nipah, Chikungunya, etc.*).
   - Case-insensitive search and fuzzy-matching with interactive paginated selection menus.

2. **Core Output:**
   - **Risk Percentage:** Accurately estimates the outbreak probability/vulnerability percentage for the requested state.
   - **Future Trajectory:** Predicts whether the disease risk will **INCREASE** or **DECREASE** over the next 6 to 12 months with confidence scores and seasonal peak identification.

3. **Multi-Model Machine Learning Ensemble:**
   - **Gradient Boosting Regressor** (`sklearn`)
   - **Random Forest Regressor** (`sklearn`)
   - **Extreme Gradient Boosting** (`xgboost`)
   - **Polynomial Time-Series Trend Regressor** for multi-quarter trajectory forecasting.

4. **Multiple Interfaces:**
   - **Interactive CLI Application:** Terminal-based interface with ASCII visualization and color-coded risk gauges.
   - **Interactive Web App (Streamlit):** Web UI dashboard with charts, trajectory timelines, parameter sliders, and health action plans.

---

## 📁 Project Structure

```
disease/
│
├── data/
│   └── disease_state_data.csv    # Curated historical epidemiological, search trend & news dataset
├── disease_predictor.py          # Primary Python script & interactive CLI engine
├── web_app.py                    # Streamlit Web Application Dashboard
├── test_run.py                   # Automated test script for sample runs
├── requirements.txt              # Project Python dependencies
└── README.md                     # Comprehensive documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Required Packages
Run the following command in PowerShell or Command Prompt:

```bash
python -m pip install -r requirements.txt
```

*(Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, `plotly`, `colorama`, `tabulate`)*

---

## 🚀 How to Run

### Option A: Interactive Command-Line Interface (CLI)

Run the main prediction script:

```bash
python disease_predictor.py
```

**How it works:**
1. Loads historical datasets and automatically trains and evaluates the ML models.
2. Prompts:
   - `Enter Indian State name: ` (e.g., `Kerala` or `Maharashtra` or press Enter to browse list)
   - `Enter Disease name: ` (e.g., `Dengue` or `Malaria` or press Enter to browse list)
3. Outputs:
   - **Current Outbreak Risk Percentage** (e.g. `14.91% [MODERATE]`)
   - **Future Direction** (e.g. `INCREASING` or `DECREASING`)
   - **6-Month and 12-Month Projected Risk Levels**
   - **Ensemble Model Comparison Table**
   - **Key Feature Importances**
   - **Public Health Preventive Recommendations**

---

### Option B: Interactive Web Application Dashboard

Launch the browser interface with Streamlit:

```bash
streamlit run web_app.py
```

Then open your browser to `http://localhost:8501`. You can select any State, Disease, and adjust Search Trends / Health News sliders interactively.

---

## 🧠 Machine Learning Methodology

### 1. Feature Engineering
The model processes multidimensional signals:
- **Search Trend Score ($S_t$):** Symptom and disease query volume normalized between 0–100.
- **News Coverage Intensity ($N_t$):** Digital media reports and outbreak keyword frequency (0–100).
- **Trend-News Interaction:** $(S_t \times N_t) / 100$ capturing media-search amplification.
- **Incidence Density:** Historical baseline cases per lakh population.
- **Vulnerability Indices:** Regional climate sensitivity index, sanitation access, and healthcare index.
- **Cyclical Seasonality:** $\sin(2\pi \cdot \text{month} / 12)$ and $\cos(2\pi \cdot \text{month} / 12)$ capturing monsoon and temperature patterns.

### 2. Risk Estimation Formula
The supervised regressors predict continuous vulnerability:
$$\text{Risk \%} = f(S_t, N_t, \text{CasesPerLakh}, \text{Climate}, \text{Sanitation}, \text{Healthcare}, \text{Seasonality})$$

### 3. Trajectory Forecaster (Increase / Decrease)
Future trajectory is calculated using polynomial regression over the temporal feature space:
$$\Delta_{\text{12M}} = \frac{\hat{Y}_{t+12} - Y_t}{\max(Y_t, 0.01)} \times 100\%$$
- If $\Delta_{\text{12M}} > +2\% \longrightarrow$ **INCREASING** 📈
- If $\Delta_{\text{12M}} < -2\% \longrightarrow$ **DECREASING** 📉
- Otherwise $\longrightarrow$ **STABLE** ➡️

---

## 📊 Sample CLI Output

```text
====================================================================
  [+]  DISEASE OUTBREAK RISK REPORT
  State   : Kerala
  Disease : Dengue
  Date    : 22 November 2026
--------------------------------------------------------------------
  ── CURRENT RISK ASSESSMENT ──

  Predicted Outbreak Risk  :  14.91%  [MODERATE]
  [███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 14.9%  ▶  MODERATE

  Search Trend Score : 56/100
  News Coverage Score: 49/100
  Climate Risk       : 0.70
  Sanitation Index   : 0.72
  Healthcare Index   : 0.80

  ┌─ ML Model Comparison ───────────────────────────────┐
  │ Random Forest           14.92%  ■■■■■■■
  │ Gradient Boosting       14.91%  ■■■■■■■
  │ XGBoost                 13.59%  ■■■■■■
  └──────────────────────────────────────────────────────┘

  ┌─ Future Trend Forecast ─────────────────────────────┐
  │ Direction     : /^\  INCREASING
  │ Confidence    : 71%
  │ 6-Month Fcast : 72.50%
  │ 12-Month Fcast: 76.26%
  │ 12M Change    : +411.5%
  │ Seasonal Peak : Oct
  │
  │ Dengue risk in Kerala is projected to sharply rise by 411.5% over the next 12 months. Peak season is typically around Oct.
  └──────────────────────────────────────────────────────┘
====================================================================
```
