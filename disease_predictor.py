"""
==============================================================================
  DISEASE OUTBREAK RISK PREDICTION SYSTEM
  Using Search Trends & Health News Data for Indian States
==============================================================================
  Author  : Disease Risk Prediction System
  Version : 2.0
  Python  : 3.8+
==============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Back, Style, init

# Force UTF-8 output on Windows to handle special characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Initialize colorama for colored terminal output
init(autoreset=True)
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║          [~]  DISEASE OUTBREAK RISK PREDICTION SYSTEM  [~]          ║
║           India | Search Trends + Health News Signals            ║
║                     Machine Learning Powered                     ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "disease_state_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli", "Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

DISEASES = [
    "Dengue", "Malaria", "Cholera", "COVID-19",
    "Typhoid", "Nipah", "Chikungunya", "Tuberculosis",
    "Hepatitis B", "Japanese Encephalitis"
]

# Risk level thresholds
RISK_LEVELS = {
    (0,   10):  ("LOW",      Fore.GREEN),
    (10,  25):  ("MODERATE", Fore.YELLOW),
    (25,  50):  ("HIGH",     Fore.RED),
    (50, float('inf')): ("CRITICAL", Fore.MAGENTA),
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class DataManager:
    """Loads, validates, and enriches the disease dataset."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        self._enrich()
        return self.df

    def _enrich(self):
        """Add derived features used by all models."""
        df = self.df
        df['cases_per_lakh'] = (df['cases'] / df['population']) * 100000
        df['composite_score'] = (
            df['search_trend_score'] * 0.35
            + df['news_coverage_score'] * 0.25
            + df['cases_per_lakh'] * 0.20
            + df['climate_risk'] * 0.10
            + (1 - df['sanitation_score']) * 0.05
            + (1 - df['healthcare_index']) * 0.05
        )
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['trend_news_interaction'] = (
            df['search_trend_score'] * df['news_coverage_score'] / 100
        )
        self.df = df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """Trains ML models (Random Forest + XGBoost) for risk % prediction."""

    FEATURES = [
        'search_trend_score', 'news_coverage_score', 'cases_per_lakh',
        'climate_risk', 'sanitation_score', 'healthcare_index',
        'composite_score', 'sin_month', 'cos_month',
        'trend_news_interaction', 'month', 'year'
    ]
    TARGET = 'risk_percentage'

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.scalers = {}
        self.trained = False

    def train(self):
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        try:
            from xgboost import XGBRegressor
            HAS_XGB = True
        except ImportError:
            HAS_XGB = False

        X = self.df[self.FEATURES]
        y = self.df[self.TARGET]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['standard'] = scaler

        # ── Random Forest ──────────────────────────────────────────────────
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=3,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        rf.fit(X_scaled, y)
        rf_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
        self.models['random_forest'] = {
            'model': rf, 'name': 'Random Forest',
            'cv_r2': rf_scores.mean(), 'cv_std': rf_scores.std()
        }

        # ── Gradient Boosting ──────────────────────────────────────────────
        gbm = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=6,
            min_samples_split=4, random_state=42
        )
        gbm.fit(X_scaled, y)
        gbm_scores = cross_val_score(gbm, X_scaled, y, cv=5, scoring='r2')
        self.models['gradient_boosting'] = {
            'model': gbm, 'name': 'Gradient Boosting',
            'cv_r2': gbm_scores.mean(), 'cv_std': gbm_scores.std()
        }

        # ── XGBoost (optional) ─────────────────────────────────────────────
        if HAS_XGB:
            xgb = XGBRegressor(
                n_estimators=200, learning_rate=0.08, max_depth=6,
                colsample_bytree=0.8, subsample=0.8, random_state=42,
                verbosity=0
            )
            xgb.fit(X_scaled, y)
            xgb_scores = cross_val_score(xgb, X_scaled, y, cv=5, scoring='r2')
            self.models['xgboost'] = {
                'model': xgb, 'name': 'XGBoost',
                'cv_r2': xgb_scores.mean(), 'cv_std': xgb_scores.std()
            }

        self.trained = True
        self._select_best()

    def _select_best(self):
        """Pick the model with highest cross-validated R²."""
        best_key = max(self.models, key=lambda k: self.models[k]['cv_r2'])
        self.best_model_key = best_key
        self.best = self.models[best_key]

    def predict(self, features_dict: dict) -> float:
        """Predict risk % using the best ensemble model."""
        X = pd.DataFrame([features_dict])[self.FEATURES]
        X_scaled = self.scalers['standard'].transform(X)
        pred = self.best['model'].predict(X_scaled)[0]
        return max(0.0, round(float(pred), 2))

    def predict_all(self, features_dict: dict) -> dict:
        """Get predictions from every trained model."""
        X = pd.DataFrame([features_dict])[self.FEATURES]
        X_scaled = self.scalers['standard'].transform(X)
        results = {}
        for key, info in self.models.items():
            pred = info['model'].predict(X_scaled)[0]
            results[info['name']] = max(0.0, round(float(pred), 2))
        return results

    def feature_importance(self) -> pd.Series:
        m = self.best['model']
        if hasattr(m, 'feature_importances_'):
            return pd.Series(
                m.feature_importances_, index=self.FEATURES
            ).sort_values(ascending=False)
        return pd.Series()


# ─────────────────────────────────────────────────────────────────────────────
# TREND FORECASTER  (time-series regression)
# ─────────────────────────────────────────────────────────────────────────────

class TrendForecaster:
    """
    Builds a polynomial regression + seasonal decomposition to forecast
    whether disease risk will INCREASE or DECREASE over the next 12 months.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def forecast(self, state: str, disease: str, current_risk: float) -> dict:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        sub = self.df[
            (self.df['state'] == state) & (self.df['disease'] == disease)
        ].copy()

        result = {
            'direction': 'UNKNOWN',
            'confidence': 0,
            'slope': 0,
            'forecast_next_6m': current_risk,
            'forecast_next_12m': current_risk,
            'seasonal_peak': 'Unknown',
            'trend_summary': 'Insufficient data for trend analysis.',
            'data_points': len(sub)
        }

        if len(sub) < 3:
            result['trend_summary'] = (
                "Limited historical data available. Prediction based on "
                "regional baseline and national averages."
            )
            # Fallback: use search trend direction
            avg_trend = sub['search_trend_score'].mean() if len(sub) > 0 else 50
            if avg_trend > 65:
                result['direction'] = 'INCREASING'
                result['confidence'] = 55
                result['forecast_next_6m'] = current_risk * 1.1
                result['forecast_next_12m'] = current_risk * 1.18
            else:
                result['direction'] = 'DECREASING'
                result['confidence'] = 55
                result['forecast_next_6m'] = current_risk * 0.92
                result['forecast_next_12m'] = current_risk * 0.85
            return result

        # Build a time index
        sub = sub.sort_values(['year', 'month']).reset_index(drop=True)
        sub['t'] = sub['year'] * 12 + sub['month']
        t_min = sub['t'].min()
        sub['t_norm'] = sub['t'] - t_min

        X = sub[['t_norm']].values
        y = sub['risk_percentage'].values

        # Polynomial degree 2 regression
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        reg = LinearRegression()
        reg.fit(X_poly, y)

        # Forecast 6 and 12 months ahead
        t_last = sub['t_norm'].max()
        t_6 = poly.transform([[t_last + 6]])
        t_12 = poly.transform([[t_last + 12]])

        f6  = max(0.0, round(float(reg.predict(t_6)[0]),  2))
        f12 = max(0.0, round(float(reg.predict(t_12)[0]), 2))

        slope = float(reg.coef_[0])  # linear coefficient direction
        pct_change = ((f12 - current_risk) / max(current_risk, 0.01)) * 100

        # Seasonal peak month
        if len(sub) >= 6:
            peak_month = sub.loc[sub['cases'].idxmax(), 'month']
            month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                           'Jul','Aug','Sep','Oct','Nov','Dec']
            result['seasonal_peak'] = month_names[int(peak_month) - 1]

        # Confidence: based on R² and data volume
        try:
            from sklearn.metrics import r2_score
            y_pred = reg.predict(X_poly)
            r2 = max(0.0, r2_score(y, y_pred))
            conf = int(50 + r2 * 40 + min(len(sub), 10) * 1.5)
        except Exception:
            conf = 60

        result.update({
            'direction': 'INCREASING' if pct_change > 2 else
                         'DECREASING' if pct_change < -2 else 'STABLE',
            'confidence': min(conf, 95),
            'slope': round(slope, 4),
            'forecast_next_6m': f6,
            'forecast_next_12m': f12,
            'pct_change': round(pct_change, 1),
            'trend_summary': self._make_summary(disease, state, f12,
                                                 pct_change, result['seasonal_peak'])
        })
        return result

    @staticmethod
    def _make_summary(disease, state, f12, pct_change, peak_month):
        direction = "rise" if pct_change > 2 else "fall" if pct_change < -2 else "remain stable"
        magnitude = abs(pct_change)
        severity = "sharply" if magnitude > 30 else "moderately" if magnitude > 10 else "slightly"
        return (
            f"{disease} risk in {state} is projected to {severity} {direction} "
            f"by {magnitude:.1f}% over the next 12 months. "
            f"{'Peak season is typically around ' + peak_month + '.' if peak_month != 'Unknown' else ''}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# NEWS SENTIMENT SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class NewsSentimentSimulator:
    """
    Simulates real-time health news sentiment (search-trend proxy).
    In production, replace with actual Google Trends / NewsAPI calls.
    """

    DISEASE_KEYWORDS = {
        'Dengue':              ['dengue', 'aedes mosquito', 'dengue outbreak', 'dengue fever'],
        'Malaria':             ['malaria', 'plasmodium', 'malaria outbreak', 'anopheles'],
        'Cholera':             ['cholera', 'diarrhoea outbreak', 'water contamination'],
        'COVID-19':            ['covid', 'coronavirus', 'sars-cov-2', 'omicron'],
        'Typhoid':             ['typhoid', 'salmonella typhi', 'enteric fever'],
        'Nipah':               ['nipah', 'nipah virus', 'bat virus'],
        'Chikungunya':         ['chikungunya', 'chikungunya fever', 'aedes'],
        'Tuberculosis':        ['tuberculosis', 'tb', 'mycobacterium'],
        'Hepatitis B':         ['hepatitis b', 'hbv', 'liver infection'],
        'Japanese Encephalitis': ['japanese encephalitis', 'je', 'brain fever', 'culex mosquito'],
    }

    @staticmethod
    def get_simulated_scores(disease: str, state: str, month: int) -> dict:
        """Return plausible search trend + news scores based on seasonality."""
        # Monsoon amplifier (Jun-Oct)
        is_monsoon = 1 if 6 <= month <= 10 else 0
        is_winter  = 1 if month in [1, 2, 11, 12] else 0

        base_trend = {
            'Dengue':              60 + 20 * is_monsoon,
            'Malaria':             55 + 25 * is_monsoon,
            'Cholera':             45 + 15 * is_monsoon,
            'COVID-19':            35 + 10 * is_winter,
            'Typhoid':             50 + 10 * is_monsoon,
            'Nipah':               85,
            'Chikungunya':         55 + 15 * is_monsoon,
            'Tuberculosis':        40,
            'Hepatitis B':         35,
            'Japanese Encephalitis': 50 + 20 * is_monsoon,
        }.get(disease, 50)

        noise = np.random.randint(-8, 8)
        trend = int(np.clip(base_trend + noise, 10, 100))
        news  = int(np.clip(trend * 0.85 + np.random.randint(-5, 5), 10, 100))
        return {'search_trend_score': trend, 'news_coverage_score': news}


# ─────────────────────────────────────────────────────────────────────────────
# STATE PROFILE  (environmental / demographic meta-data)
# ─────────────────────────────────────────────────────────────────────────────

STATE_PROFILES = {
    "Andhra Pradesh":  dict(population=52221000,  climate_risk=0.75, sanitation_score=0.52, healthcare_index=0.60),
    "Arunachal Pradesh": dict(population=1570000, climate_risk=0.82, sanitation_score=0.30, healthcare_index=0.40),
    "Assam":           dict(population=35607039,  climate_risk=0.80, sanitation_score=0.38, healthcare_index=0.48),
    "Bihar":           dict(population=124799926, climate_risk=0.65, sanitation_score=0.28, healthcare_index=0.38),
    "Chhattisgarh":    dict(population=29436231,  climate_risk=0.78, sanitation_score=0.32, healthcare_index=0.48),
    "Goa":             dict(population=1586250,   climate_risk=0.72, sanitation_score=0.68, healthcare_index=0.72),
    "Gujarat":         dict(population=63872399,  climate_risk=0.60, sanitation_score=0.55, healthcare_index=0.62),
    "Haryana":         dict(population=28204000,  climate_risk=0.50, sanitation_score=0.60, healthcare_index=0.62),
    "Himachal Pradesh":dict(population=7451955,   climate_risk=0.25, sanitation_score=0.72, healthcare_index=0.70),
    "Jharkhand":       dict(population=38593948,  climate_risk=0.80, sanitation_score=0.30, healthcare_index=0.42),
    "Karnataka":       dict(population=67562686,  climate_risk=0.65, sanitation_score=0.60, healthcare_index=0.68),
    "Kerala":          dict(population=35699443,  climate_risk=0.70, sanitation_score=0.72, healthcare_index=0.80),
    "Madhya Pradesh":  dict(population=85358965,  climate_risk=0.68, sanitation_score=0.35, healthcare_index=0.50),
    "Maharashtra":     dict(population=124904071, climate_risk=0.62, sanitation_score=0.60, healthcare_index=0.68),
    "Manipur":         dict(population=3091545,   climate_risk=0.80, sanitation_score=0.32, healthcare_index=0.42),
    "Meghalaya":       dict(population=3366710,   climate_risk=0.85, sanitation_score=0.30, healthcare_index=0.42),
    "Mizoram":         dict(population=1239244,   climate_risk=0.82, sanitation_score=0.32, healthcare_index=0.42),
    "Nagaland":        dict(population=2249695,   climate_risk=0.78, sanitation_score=0.32, healthcare_index=0.42),
    "Odisha":          dict(population=46356334,  climate_risk=0.80, sanitation_score=0.32, healthcare_index=0.50),
    "Punjab":          dict(population=30141373,  climate_risk=0.42, sanitation_score=0.62, healthcare_index=0.68),
    "Rajasthan":       dict(population=80466000,  climate_risk=0.50, sanitation_score=0.42, healthcare_index=0.52),
    "Sikkim":          dict(population=690251,    climate_risk=0.72, sanitation_score=0.52, healthcare_index=0.52),
    "Tamil Nadu":      dict(population=77841267,  climate_risk=0.72, sanitation_score=0.60, healthcare_index=0.68),
    "Telangana":       dict(population=37220000,  climate_risk=0.72, sanitation_score=0.52, healthcare_index=0.62),
    "Tripura":         dict(population=4169794,   climate_risk=0.80, sanitation_score=0.32, healthcare_index=0.42),
    "Uttar Pradesh":   dict(population=241066874, climate_risk=0.62, sanitation_score=0.30, healthcare_index=0.40),
    "Uttarakhand":     dict(population=11250858,  climate_risk=0.48, sanitation_score=0.60, healthcare_index=0.60),
    "West Bengal":     dict(population=99609303,  climate_risk=0.68, sanitation_score=0.52, healthcare_index=0.58),
    "Andaman and Nicobar Islands": dict(population=397000, climate_risk=0.78, sanitation_score=0.60, healthcare_index=0.62),
    "Chandigarh":      dict(population=1158473,   climate_risk=0.40, sanitation_score=0.75, healthcare_index=0.80),
    "Dadra and Nagar Haveli": dict(population=615724, climate_risk=0.65, sanitation_score=0.55, healthcare_index=0.55),
    "Daman and Diu":   dict(population=243247,    climate_risk=0.65, sanitation_score=0.60, healthcare_index=0.60),
    "Delhi":           dict(population=32941309,  climate_risk=0.52, sanitation_score=0.62, healthcare_index=0.70),
    "Jammu and Kashmir": dict(population=13606320,climate_risk=0.22, sanitation_score=0.60, healthcare_index=0.60),
    "Ladakh":          dict(population=290492,    climate_risk=0.18, sanitation_score=0.68, healthcare_index=0.58),
    "Lakshadweep":     dict(population=70000,     climate_risk=0.60, sanitation_score=0.68, healthcare_index=0.60),
    "Puducherry":      dict(population=1413542,   climate_risk=0.70, sanitation_score=0.65, healthcare_index=0.68),
}

# Default profile for unrecognised states
DEFAULT_PROFILE = dict(population=5000000, climate_risk=0.60,
                        sanitation_score=0.50, healthcare_index=0.55)

# Disease baseline cases per lakh (national average)
DISEASE_BASELINES = {
    'Dengue':              15.0,
    'Malaria':             20.0,
    'Cholera':             4.0,
    'COVID-19':            25.0,
    'Typhoid':             8.0,
    'Nipah':               0.01,
    'Chikungunya':         10.0,
    'Tuberculosis':        18.0,
    'Hepatitis B':         5.0,
    'Japanese Encephalitis': 3.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_separator(char='─', width=68, color=Fore.CYAN):
    print(color + char * width + Style.RESET_ALL)

def get_risk_label(risk: float):
    for (lo, hi), (label, color) in RISK_LEVELS.items():
        if lo <= risk < hi:
            return label, color
    return "CRITICAL", Fore.MAGENTA

def print_risk_bar(risk: float, width: int = 50):
    label, color = get_risk_label(risk)
    filled = int(min(risk / 100, 1.0) * width)
    bar = '█' * filled + '░' * (width - filled)
    print(f"  {color}[{bar}] {risk:.1f}%  ▶  {label}{Style.RESET_ALL}")

def fuzzy_match(query: str, choices: list) -> str | None:
    """Case-insensitive partial match."""
    q = query.strip().lower()
    for c in choices:
        if q == c.lower():
            return c
    for c in choices:
        if q in c.lower() or c.lower().startswith(q):
            return c
    return None

def print_model_comparison(predictions: dict):
    print(f"\n{Fore.CYAN}  ┌─ ML Model Comparison ───────────────────────────────┐{Style.RESET_ALL}")
    for name, val in predictions.items():
        _, color = get_risk_label(val)
        bar = '■' * int(val / 2)
        print(f"  │ {name:<22} {color}{val:6.2f}%  {bar}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  └──────────────────────────────────────────────────────┘{Style.RESET_ALL}")

def print_feature_importance(importance: pd.Series):
    if importance.empty:
        return
    print(f"\n{Fore.CYAN}  ┌─ Top Feature Importances ───────────────────────────┐{Style.RESET_ALL}")
    for feat, val in importance.head(6).items():
        bar = '▓' * int(val * 60)
        print(f"  │ {feat:<30} {val:.3f}  {bar}")
    print(f"{Fore.CYAN}  └──────────────────────────────────────────────────────┘{Style.RESET_ALL}")

def print_forecast_panel(forecast: dict):
    d = forecast['direction']
    color = Fore.RED if d == 'INCREASING' else Fore.GREEN if d == 'DECREASING' else Fore.YELLOW
    arrow = '/^\\' if d == 'INCREASING' else '\\v/' if d == 'DECREASING' else '->'

    print(f"\n{Fore.CYAN}  ┌─ Future Trend Forecast ─────────────────────────────┐{Style.RESET_ALL}")
    print(f"  │ Direction     : {color}{arrow}  {d}{Style.RESET_ALL}")
    print(f"  │ Confidence    : {forecast['confidence']}%")
    print(f"  │ 6-Month Fcast : {forecast['forecast_next_6m']:.2f}%")
    print(f"  │ 12-Month Fcast: {forecast['forecast_next_12m']:.2f}%")
    if forecast.get('pct_change') is not None:
        pct = forecast['pct_change']
        p_color = Fore.RED if pct > 0 else Fore.GREEN
        print(f"  │ 12M Change    : {p_color}{'+' if pct > 0 else ''}{pct:.1f}%{Style.RESET_ALL}")
    if forecast['seasonal_peak'] != 'Unknown':
        print(f"  │ Seasonal Peak : {forecast['seasonal_peak']}")
    print(f"  │")
    print(f"  │ {Fore.WHITE}{forecast['trend_summary']}{Style.RESET_ALL}")
    if forecast['data_points'] < 3:
        print(f"  │ {Fore.YELLOW}! Based on limited data ({forecast['data_points']} records){Style.RESET_ALL}")
    print(f"{Fore.CYAN}  └──────────────────────────────────────────────────────┘{Style.RESET_ALL}")

def print_recommendations(disease: str, risk: float, direction: str):
    label, _ = get_risk_label(risk)
    print(f"\n{Fore.CYAN}  ┌─ Public Health Recommendations ─────────────────────┐{Style.RESET_ALL}")

    disease_tips = {
        'Dengue':    ["Eliminate stagnant water (flower pots, tyres, coolers)",
                      "Use mosquito nets and repellents",
                      "Wear full-sleeve clothing during peak hours"],
        'Malaria':   ["Use insecticide-treated bed nets (ITNs)",
                      "Indoor residual spraying (IRS) of walls",
                      "Take prophylactic medication if travelling"],
        'Cholera':   ["Drink only boiled/filtered water",
                      "Wash hands with soap before eating",
                      "Avoid raw seafood and street food"],
        'COVID-19':  ["Wear masks in crowded indoor spaces",
                      "Ensure adequate ventilation",
                      "Keep vaccinations up to date"],
        'Typhoid':   ["Vaccinate high-risk populations",
                      "Avoid unhygienic food and water sources",
                      "Early antibiotic treatment on diagnosis"],
        'Nipah':     ["Avoid contact with bats and sick animals",
                      "Do not consume raw date palm sap",
                      "Isolate confirmed cases strictly"],
        'Chikungunya':["Eliminate Aedes mosquito breeding sites",
                       "Use repellents and protective clothing",
                       "Report clusters to local health authorities"],
        'Tuberculosis':["Complete full antibiotic course (DOTS)",
                        "Improve ventilation in living spaces",
                        "Screen household contacts of TB patients"],
        'Hepatitis B': ["Vaccinate infants and high-risk adults",
                        "Use sterile needles; screen blood donors",
                        "Safe sexual practices"],
        'Japanese Encephalitis': ["Vaccinate children in endemic areas",
                                   "Control Culex mosquito breeding (rice fields, ponds)",
                                   "Use bed nets during evening hours"],
    }
    tips = disease_tips.get(disease, ["Consult local health authorities",
                                       "Maintain hygiene and sanitation",
                                       "Report unusual disease clusters"])

    if label in ('HIGH', 'CRITICAL') or direction == 'INCREASING':
        print(f"  │ {Fore.RED}! ALERT: Elevated risk detected – immediate action needed!{Style.RESET_ALL}")

    for i, tip in enumerate(tips, 1):
        print(f"  │  {i}. {tip}")

    if direction == 'INCREASING':
        print(f"  │  [!] Trend is rising – strengthen surveillance and response.")
    elif direction == 'DECREASING':
        print(f"  │  [OK] Downward trend – continue current preventive measures.")

    print(f"{Fore.CYAN}  └──────────────────────────────────────────────────────┘{Style.RESET_ALL}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PredictionEngine:

    def __init__(self):
        self.data_manager   = DataManager(DATA_PATH)
        self.model_trainer  = None
        self.forecaster     = None
        self.news_sim       = NewsSentimentSimulator()

    def initialise(self):
        print(f"\n{Fore.YELLOW}  [*]  Loading data and training models …{Style.RESET_ALL}")
        df = self.data_manager.load()
        self.model_trainer = ModelTrainer(df)
        self.model_trainer.train()
        self.forecaster    = TrendForecaster(df)
        best = self.model_trainer.best
        print(f"  {Fore.GREEN}[OK]  Best model : {best['name']}"
              f"  │  CV R² = {best['cv_r2']:.3f} ± {best['cv_std']:.3f}{Style.RESET_ALL}")
        return df

    def build_feature_vector(self, state: str, disease: str, month: int, year: int, search_trend_score: float = None, news_coverage_score: float = None) -> dict:
        profile = STATE_PROFILES.get(state, DEFAULT_PROFILE)
        scores  = self.news_sim.get_simulated_scores(disease, state, month)
        baseline_cases_lakh = DISEASE_BASELINES.get(disease, 10.0)

        c_risk  = profile['climate_risk']
        san     = profile['sanitation_score']
        hc      = profile['healthcare_index']
        trend   = float(search_trend_score) if search_trend_score is not None else scores['search_trend_score']
        news    = float(news_coverage_score) if news_coverage_score is not None else scores['news_coverage_score']

        composite = (
            trend * 0.35 + news * 0.25 + baseline_cases_lakh * 0.20
            + c_risk * 0.10 + (1 - san) * 0.05 + (1 - hc) * 0.05
        )

        return {
            'search_trend_score':    trend,
            'news_coverage_score':   news,
            'cases_per_lakh':        baseline_cases_lakh,
            'climate_risk':          c_risk,
            'sanitation_score':      san,
            'healthcare_index':      hc,
            'composite_score':       composite,
            'sin_month':             np.sin(2 * np.pi * month / 12),
            'cos_month':             np.cos(2 * np.pi * month / 12),
            'trend_news_interaction': trend * news / 100,
            'month':                 month,
            'year':                  year,
        }

    def run_query(self, state: str, disease: str, month: int = None, year: int = None, search_trend_score: float = None, news_coverage_score: float = None):
        now    = datetime.now()
        month  = month or now.month
        year   = year or now.year

        profile = STATE_PROFILES.get(state, DEFAULT_PROFILE)
        scores  = self.news_sim.get_simulated_scores(disease, state, month)
        if search_trend_score is not None:
            scores['search_trend_score'] = int(search_trend_score)
        if news_coverage_score is not None:
            scores['news_coverage_score'] = int(news_coverage_score)

        fv       = self.build_feature_vector(state, disease, month, year, search_trend_score, news_coverage_score)
        risk_pct = self.model_trainer.predict(fv)
        all_preds = self.model_trainer.predict_all(fv)
        importance = self.model_trainer.feature_importance()
        forecast = self.forecaster.forecast(state, disease, risk_pct)

        # ── Print results ──────────────────────────────────────────────────
        print_separator('═')
        print(f"\n{Fore.WHITE}  [+]  DISEASE OUTBREAK RISK REPORT{Style.RESET_ALL}")
        print(f"  State   : {Fore.CYAN}{state}{Style.RESET_ALL}")
        print(f"  Disease : {Fore.YELLOW}{disease}{Style.RESET_ALL}")
        print(f"  Date    : {now.strftime('%d %B %Y')}")
        print_separator()

        print(f"\n{Fore.WHITE}  ── CURRENT RISK ASSESSMENT ──{Style.RESET_ALL}")
        print(f"\n  Predicted Outbreak Risk  :  ", end="")
        label, color = get_risk_label(risk_pct)
        print(f"{color}{risk_pct:.2f}%  [{label}]{Style.RESET_ALL}")
        print_risk_bar(risk_pct)

        print(f"\n  Search Trend Score : {scores['search_trend_score']}/100")
        print(f"  News Coverage Score: {scores['news_coverage_score']}/100")
        print(f"  Climate Risk       : {profile['climate_risk']:.2f}")
        print(f"  Sanitation Index   : {profile['sanitation_score']:.2f}")
        print(f"  Healthcare Index   : {profile['healthcare_index']:.2f}")

        print_model_comparison(all_preds)
        print_feature_importance(importance)
        print_forecast_panel(forecast)
        print_recommendations(disease, risk_pct, forecast['direction'])
        print_separator('═')


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────────────────────

def select_from_list(prompt_text: str, choices: list, page_size: int = 10) -> str:
    """Paginated numbered-list picker."""
    while True:
        print(f"\n{Fore.CYAN}{prompt_text}{Style.RESET_ALL}")
        total = len(choices)
        pages = (total + page_size - 1) // page_size
        page  = 0

        while True:
            start = page * page_size
            end   = min(start + page_size, total)
            for i, item in enumerate(choices[start:end], start + 1):
                print(f"  {Fore.WHITE}{i:>2}.{Style.RESET_ALL} {item}")

            nav_parts = []
            if page > 0:            nav_parts.append("[P] Prev")
            if page < pages - 1:    nav_parts.append("[N] Next")
            nav_parts.append("[Q] Search by typing")
            print(f"\n  {' | '.join(nav_parts)}")
            choice = input(f"\n  Enter number or option: ").strip()

            if choice.upper() == 'N' and page < pages - 1:
                page += 1
            elif choice.upper() == 'P' and page > 0:
                page -= 1
            elif choice.upper() == 'Q':
                break
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < total:
                    return choices[idx]
                print(f"  {Fore.RED}Invalid number. Try again.{Style.RESET_ALL}")
            else:
                print(f"  {Fore.RED}Invalid option. Try again.{Style.RESET_ALL}")

        # Free-text search
        query = input(f"  Search: ").strip()
        matched = fuzzy_match(query, choices)
        if matched:
            print(f"  {Fore.GREEN}[v] Matched: {matched}{Style.RESET_ALL}")
            return matched
        print(f"  {Fore.RED}No match for '{query}'. Please try again.{Style.RESET_ALL}")


def run_cli(engine: PredictionEngine):
    while True:
        print(f"\n{Fore.CYAN}  ══ NEW QUERY ════════════════════════════════════════════{Style.RESET_ALL}")
        print(f"  Type a name or number to select, or press {Fore.YELLOW}Ctrl+C{Style.RESET_ALL} to exit.\n")

        # ── State selection ────────────────────────────────────────────────
        state = None
        raw_state = input(f"  {Fore.GREEN}Enter Indian State name: {Style.RESET_ALL}").strip()
        if raw_state:
            state = fuzzy_match(raw_state, INDIAN_STATES)
            if not state:
                print(f"  {Fore.RED}State not recognised. Showing list…{Style.RESET_ALL}")

        if not state:
            state = select_from_list("Select Indian State:", INDIAN_STATES)

        # ── Disease selection ──────────────────────────────────────────────
        disease = None
        raw_disease = input(f"\n  {Fore.GREEN}Enter Disease name: {Style.RESET_ALL}").strip()
        if raw_disease:
            disease = fuzzy_match(raw_disease, DISEASES)
            if not disease:
                print(f"  {Fore.RED}Disease not recognised. Showing list…{Style.RESET_ALL}")

        if not disease:
            disease = select_from_list("Select Disease:", DISEASES)

        # ── Optional Manual Month & Year ──────────────────────────────────
        now = datetime.now()
        raw_m = input(f"  Month (1-12, press Enter for current {now.month}): ").strip()
        month = int(raw_m) if raw_m.isdigit() and 1 <= int(raw_m) <= 12 else now.month

        raw_y = input(f"  Year (e.g. {now.year}, press Enter for current): ").strip()
        year = int(raw_y) if raw_y.isdigit() and 2000 <= int(raw_y) <= 2040 else now.year

        # ── Optional Manual Surveillance Scores ───────────────────────────
        raw_st = input("  Manual Search Trend Value (0-100, press Enter to auto-calculate): ").strip()
        st_val = float(raw_st) if raw_st.replace('.', '', 1).isdigit() else None

        raw_nw = input("  Manual News Media Mentions (0-100, press Enter to auto-calculate): ").strip()
        nw_val = float(raw_nw) if raw_nw.replace('.', '', 1).isdigit() else None

        # ── Run prediction ─────────────────────────────────────────────────
        engine.run_query(state, disease, month=month, year=year, search_trend_score=st_val, news_coverage_score=nw_val)

        # ── Continue? ──────────────────────────────────────────────────────
        again = input(f"\n  {Fore.YELLOW}Run another query? (Y/N): {Style.RESET_ALL}").strip().upper()
        if again != 'Y':
            print(f"\n{Fore.CYAN}  Thank you for using the Disease Risk Prediction System!{Style.RESET_ALL}\n")
            break


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    clear_screen()
    print(BANNER)
    engine = PredictionEngine()
    try:
        engine.initialise()
        run_cli(engine)
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}  Exiting …  Goodbye!{Style.RESET_ALL}\n")
        sys.exit(0)
    except FileNotFoundError:
        print(f"\n{Fore.RED}  ERROR: Data file not found at:{Style.RESET_ALL}\n  {DATA_PATH}")
        print(f"  Make sure 'data/disease_state_data.csv' exists in the project folder.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n{Fore.RED}  Unexpected error: {exc}{Style.RESET_ALL}")
        raise


if __name__ == "__main__":
    main()
