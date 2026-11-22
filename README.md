---
title: Disease Outbreak Risk Prediction System
emoji: 🦠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

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
├── app.py                        # Main Streamlit application entry point
├── streamlit_app.py              # Streamlit Cloud entry point
├── web_app.py                    # Web Application Dashboard
├── disease_predictor.py          # Primary Python script & interactive CLI engine
├── requirements.txt              # Project Python dependencies
└── README.md                     # Comprehensive documentation & HF metadata
```

---

## ⚙️ Installation & Setup

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
