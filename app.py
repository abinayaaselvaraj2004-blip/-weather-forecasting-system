"""
AtmosBlend AI - Web Application
Adaptive Intelligence for Multi-Model Weather Forecasting
Problem Statement: 26081 | Ministry of Earth Sciences (MoES) / NCMRWF
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import DataPreprocessor
from src.ai_model import AIWeatherModel
from src.ensemble import EnsembleForecastEngine
from src.weighting import AdaptiveWeightingEngine
from src.extreme_weather import ExtremeWeatherDetector
from src.evaluation import ForecastEvaluator

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AtmosBlend AI | NCMRWF-MoES",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Crisp Light & Bright UI Theme
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Crisp Light Background */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        color: #0f172a;
    }

    /* Metric Card Styling - Bright Clean White */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #38bdf8;
        box-shadow: 0 12px 28px -4px rgba(2, 132, 199, 0.12);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #0284c7;
        margin-top: 6px;
        font-weight: 600;
    }

    /* Welcome Hero Banner */
    .welcome-hero {
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 50%, #f0fdf4 100%);
        border: 1px solid #bfdbfe;
        border-radius: 20px;
        padding: 32px 36px;
        box-shadow: 0 10px 30px -5px rgba(2, 132, 199, 0.1);
        margin-bottom: 24px;
    }
    .hero-badge {
        background: #dbeafe;
        color: #1d4ed8;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 14px;
        border: 1px solid #bfdbfe;
    }

    /* Feature & Module Cards */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 15px -2px rgba(15, 23, 42, 0.04);
        transition: all 0.22s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: #0284c7;
        box-shadow: 0 12px 24px -4px rgba(2, 132, 199, 0.14);
    }
    .scenario-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .scenario-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 8px 20px rgba(2, 132, 199, 0.12);
        transform: translateY(-2px);
    }

    /* Severe Weather Alert Card */
    .alert-card {
        padding: 18px 24px;
        border-radius: 14px;
        margin-bottom: 20px;
        border-left: 6px solid;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
    }
    .alert-red {
        background: #fef2f2;
        border-color: #ef4444;
        color: #991b1b;
    }
    .alert-orange {
        background: #fff7ed;
        border-color: #f97316;
        color: #9a3412;
    }
    .alert-yellow {
        background: #fefce8;
        border-color: #eab308;
        color: #854d0e;
    }
    .alert-green {
        background: #f0fdf4;
        border-color: #22c55e;
        color: #166534;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar aesthetics - Bright & Clean */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.03);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #334155 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #cbd5e1;
        background: #ffffff;
        color: #0f172a;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #0284c7;
        color: #0284c7;
        background: #f0f9ff;
        transform: translateY(-1px);
    }
    
    /* Disclaimer Tag */
    .demo-watermark {
        background: #fffbeb;
        border: 1px dashed #f59e0b;
        padding: 8px 14px;
        border-radius: 8px;
        color: #b45309;
        font-size: 0.78rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INITIALIZE ENGINES & DATA CACHE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system_engines():
    data_processed_path = os.path.join(BASE_DIR, "data", "processed", "weather_cleaned.csv")
    if not os.path.exists(data_processed_path):
        from src.data_generator import generate_weather_data
        from src.preprocessing import DataPreprocessor
        raw_df = generate_weather_data(n_samples=1600, random_seed=42)
        prep = DataPreprocessor()
        clean_df, _ = prep.clean_and_preprocess(raw_df)
        prep.save_processed(clean_df)

    ai_engine = AIWeatherModel(data_processed_path)
    if not ai_engine.load_saved_models():
        ai_engine.train_models()

    weight_engine = AdaptiveWeightingEngine(data_processed_path)
    evaluator = ForecastEvaluator(data_processed_path)
    df_clean = pd.read_csv(data_processed_path)
    return ai_engine, weight_engine, evaluator, df_clean

ai_engine, weight_engine, evaluator, df_clean = load_system_engines()

# -----------------------------------------------------------------------------
# PRESET SCENARIO HANDLER & SESSION STATE
# -----------------------------------------------------------------------------
SCENARIOS = {
    "Scenario 1: Tamil Nadu NE Monsoon (Heavy Rain)": {
        "region": "Tamil Nadu",
        "season": "Northeast Monsoon",
        "lead_time": "24 hours",
        "regime": "Heavy Rain",
        "variable": "Rainfall"
    },
    "Scenario 2: Rajasthan Summer (Heat Wave)": {
        "region": "Rajasthan",
        "season": "Summer",
        "lead_time": "48 hours",
        "regime": "Heat Wave",
        "variable": "Temperature"
    },
    "Scenario 3: Kerala SW Monsoon (Heavy Rain)": {
        "region": "Kerala",
        "season": "Southwest Monsoon",
        "lead_time": "24 hours",
        "regime": "Heavy Rain",
        "variable": "Rainfall"
    },
    "Scenario 4: Coastal Odisha (High Wind / Cyclone)": {
        "region": "Odisha",
        "season": "Post-Monsoon",
        "lead_time": "24 hours",
        "regime": "Storm/Cyclone",
        "variable": "Wind"
    }
}

if "region" not in st.session_state:
    st.session_state.region = "Tamil Nadu"
if "season" not in st.session_state:
    st.session_state.season = "Northeast Monsoon"
if "lead_time" not in st.session_state:
    st.session_state.lead_time = "24 hours"
if "regime" not in st.session_state:
    st.session_state.regime = "Heavy Rain"
if "variable" not in st.session_state:
    st.session_state.variable = "Rainfall"
if "forecast_date" not in st.session_state:
    st.session_state.forecast_date = datetime.date(2024, 11, 15)

def apply_scenario(sc_key):
    sc = SCENARIOS[sc_key]
    st.session_state.region = sc["region"]
    st.session_state.season = sc["season"]
    st.session_state.lead_time = sc["lead_time"]
    st.session_state.regime = sc["regime"]
    st.session_state.variable = sc["variable"]

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <div style="background: linear-gradient(135deg, #0284c7, #38bdf8); width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 22px;">🌦️</div>
            <div>
                <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">ATMOSBLEND AI</h2>
                <span style="font-size: 0.72rem; color: #0284c7; font-weight: 700; text-transform: uppercase;">MoES • NCMRWF PS-26081</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="demo-watermark">
            ⚠️ <b>SIMULATED / DEMO DATA</b><br/>
            Atmospheric Research Prototype for Hackathon Evaluation.
        </div>
    """, unsafe_allow_html=True)

    # Preset Quick-Launch Scenarios
    st.markdown("<h4 style='font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;'>🎯 Preset Demo Scenarios</h4>", unsafe_allow_html=True)
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        if st.button("🌊 TN Heavy Rain", use_container_width=True, help="Tamil Nadu, NE Monsoon, 24h, Heavy Rain"):
            apply_scenario("Scenario 1: Tamil Nadu NE Monsoon (Heavy Rain)")
            st.rerun()
        if st.button("⛈️ Kerala SW Monsoon", use_container_width=True, help="Kerala, SW Monsoon, 24h, Heavy Rain"):
            apply_scenario("Scenario 3: Kerala SW Monsoon (Heavy Rain)")
            st.rerun()
    with col_sc2:
        if st.button("🔥 Rajasthan Heat", use_container_width=True, help="Rajasthan, Summer, 48h, Heat Wave"):
            apply_scenario("Scenario 2: Rajasthan Summer (Heat Wave)")
            st.rerun()
        if st.button("🌀 Odisha Storm", use_container_width=True, help="Odisha, Post-Monsoon, 24h, Storm/Cyclone"):
            apply_scenario("Scenario 4: Coastal Odisha (High Wind / Cyclone)")
            st.rerun()

    st.markdown("---")
    st.markdown("<h4 style='font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;'>⚙️ Operational Filters</h4>", unsafe_allow_html=True)

    regions_list = ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana", "Maharashtra", "Rajasthan", "Odisha"]
    st.session_state.region = st.selectbox("Geographic Region", regions_list, index=regions_list.index(st.session_state.region))

    variables_list = ["Rainfall", "Temperature", "Wind"]
    st.session_state.variable = st.selectbox("Target Variable", variables_list, index=variables_list.index(st.session_state.variable))

    seasons_list = ["Northeast Monsoon", "Southwest Monsoon", "Summer", "Winter", "Post-Monsoon"]
    st.session_state.season = st.selectbox("Meteorological Season", seasons_list, index=seasons_list.index(st.session_state.season))

    lead_times_list = ["6 hours", "12 hours", "24 hours", "48 hours", "72 hours"]
    st.session_state.lead_time = st.selectbox("Forecast Lead Time", lead_times_list, index=lead_times_list.index(st.session_state.lead_time))

    regimes_list = ["Heavy Rain", "Heat Wave", "High Wind", "Storm/Cyclone", "Normal"]
    st.session_state.regime = st.selectbox("Synoptic Weather Regime", regimes_list, index=regimes_list.index(st.session_state.regime))

    st.session_state.forecast_date = st.date_input("Forecast Reference Date", value=st.session_state.forecast_date)

    st.markdown("---")
    page_options = [
        "🏠 Home (Welcome)",
        "1. Overview",
        "2. Forecast Curves",
        "3. Model Intelligence",
        "4. Adaptive Weights",
        "5. Regional Weight Map",
        "6. Extreme Weather Center",
        "7. Performance Verification",
        "8. Methodology & Pipeline"
    ]

    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Home (Welcome)"

    if "nav_target" in st.session_state:
        st.session_state.current_page = st.session_state.nav_target
        del st.session_state["nav_target"]

    target_idx = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
    selected_page = st.radio("Dashboard Module", page_options, index=target_idx, key="sidebar_module_nav")
    st.session_state.current_page = selected_page

    st.markdown("---")
    csv_file = os.path.join(BASE_DIR, "outputs", "powerbi_export.csv")
    if os.path.exists(csv_file):
        with open(csv_file, "rb") as f:
            st.download_button(
                label="📊 Download Power BI Dataset",
                data=f,
                file_name="AtmosBlend_PowerBI_Dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

# -----------------------------------------------------------------------------
# DYNAMIC METEOROLOGICAL COMPUTATIONS
# -----------------------------------------------------------------------------
cur_region = st.session_state.region
cur_var = st.session_state.variable
cur_season = st.session_state.season
cur_lead_time = st.session_state.lead_time
cur_regime = st.session_state.regime

slice_df = df_clean[
    (df_clean["Region"] == cur_region) &
    (df_clean["Season"] == cur_season) &
    (df_clean["Weather_Regime"] == cur_regime) &
    (df_clean["Lead_Time"] == cur_lead_time)
]

if len(slice_df) > 0:
    sample_row = slice_df.iloc[0].to_dict()
else:
    fallback_df = df_clean[
        (df_clean["Region"] == cur_region) &
        (df_clean["Weather_Regime"] == cur_regime)
    ]
    sample_row = fallback_df.iloc[0].to_dict() if len(fallback_df) > 0 else df_clean.iloc[0].to_dict()

nwp_rain = float(sample_row.get("NWP_Rainfall", 45.0))
nwp_temp = float(sample_row.get("NWP_Temperature", 31.0))
nwp_wind = float(sample_row.get("NWP_Wind", 22.0))

ai_rain = float(sample_row.get("AI_Rainfall", 52.0))
ai_temp = float(sample_row.get("AI_Temperature", 30.5))
ai_wind = float(sample_row.get("AI_Wind", 24.0))

ens_rain = float(sample_row.get("Ensemble_Rainfall", 48.0))
ens_temp = float(sample_row.get("Ensemble_Temperature", 30.8))
ens_wind = float(sample_row.get("Ensemble_Wind", 23.5))

actual_rain = float(sample_row.get("Actual_Rainfall", 50.0))
actual_temp = float(sample_row.get("Actual_Temperature", 30.7))
actual_wind = float(sample_row.get("Actual_Wind", 23.0))

weights_rain, errors_rain, rationale_rain = weight_engine.compute_adaptive_weights(
    cur_region, cur_season, cur_lead_time, cur_regime, "Rainfall"
)
weights_temp, errors_temp, rationale_temp = weight_engine.compute_adaptive_weights(
    cur_region, cur_season, cur_lead_time, cur_regime, "Temperature"
)
weights_wind, errors_wind, rationale_wind = weight_engine.compute_adaptive_weights(
    cur_region, cur_season, cur_lead_time, cur_regime, "Wind"
)

hybrid_rain = weight_engine.compute_hybrid_forecast(nwp_rain, ai_rain, ens_rain, weights_rain, "Rainfall")
hybrid_temp = weight_engine.compute_hybrid_forecast(nwp_temp, ai_temp, ens_temp, weights_temp, "Temperature")
hybrid_wind = weight_engine.compute_hybrid_forecast(nwp_wind, ai_wind, ens_wind, weights_wind, "Wind")

if cur_var == "Rainfall":
    active_weights = weights_rain
    active_errors = errors_rain
    active_rationale = rationale_rain
    active_hybrid = hybrid_rain
    active_nwp = nwp_rain
    active_ai = ai_rain
    active_ens = ens_rain
    active_actual = actual_rain
    unit = "mm"
elif cur_var == "Temperature":
    active_weights = weights_temp
    active_errors = errors_temp
    active_rationale = rationale_temp
    active_hybrid = hybrid_temp
    active_nwp = nwp_temp
    active_ai = ai_temp
    active_ens = ens_temp
    active_actual = actual_temp
    unit = "°C"
else:
    active_weights = weights_wind
    active_errors = errors_wind
    active_rationale = rationale_wind
    active_hybrid = hybrid_wind
    active_nwp = nwp_wind
    active_ai = ai_wind
    active_ens = ens_wind
    active_actual = actual_wind
    unit = "km/h"

dominant_model = max(active_weights, key=active_weights.get)
dominant_weight = active_weights[dominant_model]

risk_info = ExtremeWeatherDetector.evaluate_risks(
    hybrid_rain=hybrid_rain,
    hybrid_temp=hybrid_temp,
    hybrid_wind=hybrid_wind,
    region=cur_region,
    lead_time=cur_lead_time,
    weather_regime=cur_regime
)

ens_breakdown = EnsembleForecastEngine.generate_or_retrieve_members(sample_row)

# -----------------------------------------------------------------------------
# TOP HEADER SECTION
# -----------------------------------------------------------------------------
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #e2e8f0;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #0f172a, #0284c7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ATMOSBLEND AI
            </h1>
            <p style="margin: 4px 0 0 0; color: #475569; font-size: 1.05rem; font-weight: 400;">
                Adaptive Intelligence for Multi-Model Weather Forecasting • <span style="color: #0284c7; font-weight: 600;">Ministry of Earth Sciences (MoES) / NCMRWF</span>
            </p>
        </div>
        <div style="text-align: right;">
            <div class="badge" style="background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd;">
                📍 {cur_region}
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                Lead Time: <b>{cur_lead_time}</b> | Regime: <b>{cur_regime}</b>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# KPI SUMMARY CARDS
# -----------------------------------------------------------------------------
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Final Hybrid Forecast ({cur_var})</div>
            <div class="kpi-value">{active_hybrid:.1f} <span style="font-size: 1.2rem; color: #64748b;">{unit}</span></div>
            <div class="kpi-sub">Blended NWP + AI + Ensemble</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    mae_dom = active_errors[dominant_model]["MAE"]
    confidence_score = max(55, min(96, int(100 - (mae_dom * 4.2))))
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Forecast Confidence</div>
            <div class="kpi-value">{confidence_score}%</div>
            <div class="kpi-sub">Spread: ±{ens_breakdown[cur_var]['spread']:.2f} {unit}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Dominant Model</div>
            <div class="kpi-value" style="color: #0284c7;">{dominant_model}</div>
            <div class="kpi-sub">Assigned Weight: <b>{dominant_weight:.1f}%</b></div>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    risk_lvl = risk_info["overall_risk"]
    risk_clr = risk_info["risk_color"]
    st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {risk_clr};">
            <div class="kpi-title">Extreme Weather Risk</div>
            <div class="kpi-value" style="color: {risk_clr};">{risk_lvl.upper()}</div>
            <div class="kpi-sub">Threat Score: <b>{risk_info['risk_score']}/100</b></div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

primary_alert = risk_info["alerts"][0]
alert_class = "alert-red" if risk_lvl == "Severe" else ("alert-orange" if risk_lvl == "High" else ("alert-yellow" if risk_lvl == "Moderate" else "alert-green"))
st.markdown(f"""
    <div class="alert-card {alert_class}">
        <div style="font-weight: 700; font-size: 1.05rem; letter-spacing: 0.02em;">
            🚨 {primary_alert['title']}
        </div>
        <div style="font-size: 0.92rem; margin-top: 6px; opacity: 0.95;">
            {primary_alert['message']}
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 0: HOME & WELCOME
# -----------------------------------------------------------------------------
if selected_page == "🏠 Home (Welcome)":
    st.markdown("""
        <div class="welcome-hero">
            <span class="hero-badge">✨ WELCOME TO ATMOSBLEND AI • PS-26081</span>
            <h2 style="margin: 0 0 10px 0; font-size: 2.3rem; font-weight: 800; color: #0f172a; line-height: 1.2;">
                Adaptive Intelligence for Multi-Model Weather Forecasting
            </h2>
            <p style="margin: 0 0 16px 0; font-size: 1.08rem; color: #334155; line-height: 1.6; max-width: 950px;">
                Welcome to India's premier multi-model atmospheric synthesis platform designed for the 
                <b>Ministry of Earth Sciences (MoES)</b> and <b>NCMRWF</b>. AtmosBlend AI dynamically fuses physics-based 
                Numerical Weather Prediction (NWP), Machine Learning Regressors, and 5-Member Ensembles using real-time 
                inverse-error reliability weighting to eliminate extreme forecast error and trigger automated hazard warnings.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span class="badge" style="background: #dbeafe; color: #1e40af;">🚀 3-Engine Model Fusion</span>
                <span class="badge" style="background: #dcfce7; color: #166534;">🛡️ IMD Threshold Hazard Warning</span>
                <span class="badge" style="background: #fef3c7; color: #92400e;">⚡ Dynamic Regime Optimization</span>
                <span class="badge" style="background: #f3e8ff; color: #6b21a8;">📊 Power BI Ready Dataset</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_c1, col_c2 = st.columns([1.6, 1])
    with col_c1:
        st.markdown(f"""
            <div class="feature-card" style="padding: 16px 20px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Active Telemetry State</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 2px;">
                            📍 {cur_region} &nbsp;|&nbsp; 🗓️ {cur_season} &nbsp;|&nbsp; ⏱️ {cur_lead_time}
                        </div>
                        <div style="font-size: 0.85rem; color: #0284c7; margin-top: 4px;">
                            Current Weather Regime: <b>{cur_regime}</b> &nbsp;•&nbsp; Target Variable: <b>{cur_var} ({unit})</b>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_c2:
        if st.button("🚀 Enter Live Forecast Dashboard ➡️", type="primary", use_container_width=True):
            st.session_state.nav_target = "1. Overview"
            st.rerun()
        if st.button("🚨 Inspect Extreme Hazards Center ⚠️", use_container_width=True):
            st.session_state.nav_target = "6. Extreme Weather Center"
            st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;'>🎯 Interactive Weather Event Demonstrations</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 0.92rem; margin-bottom: 14px;'>Click any scenario below to automatically reconfigure the atmospheric models and examine real-time blending:</p>", unsafe_allow_html=True)

    sc_cols = st.columns(4)
    with sc_cols[0]:
        st.markdown("""
            <div class="scenario-card">
                <span class="badge" style="background: #e0f2fe; color: #0369a1;">🌊 HEAVY RAIN</span>
                <h4 style="margin: 8px 0 4px 0; color: #0f172a; font-size: 1.05rem;">Tamil Nadu NE Monsoon</h4>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Active coastal depression with 24h lead time. Extreme precipitation surge.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Test Scenario 1", key="hm_sc1", use_container_width=True):
            apply_scenario("Scenario 1: Tamil Nadu NE Monsoon (Heavy Rain)")
            st.session_state.nav_target = "1. Overview"
            st.rerun()

    with sc_cols[1]:
        st.markdown("""
            <div class="scenario-card">
                <span class="badge" style="background: #fef2f2; color: #b91c1c;">🔥 HEAT WAVE</span>
                <h4 style="margin: 8px 0 4px 0; color: #0f172a; font-size: 1.05rem;">Rajasthan Summer</h4>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Persistent high-temperature anticyclone with 48h lead time. AI regressor weighted.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Test Scenario 2", key="hm_sc2", use_container_width=True):
            apply_scenario("Scenario 2: Rajasthan Summer (Heat Wave)")
            st.session_state.nav_target = "1. Overview"
            st.rerun()

    with sc_cols[2]:
        st.markdown("""
            <div class="scenario-card">
                <span class="badge" style="background: #f0fdf4; color: #15803d;">⛈️ MONSOON SURGE</span>
                <h4 style="margin: 8px 0 4px 0; color: #0f172a; font-size: 1.05rem;">Kerala SW Monsoon</h4>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Orographic Western Ghats precipitation surge with high ensemble consensus.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Test Scenario 3", key="hm_sc3", use_container_width=True):
            apply_scenario("Scenario 3: Kerala SW Monsoon (Heavy Rain)")
            st.session_state.nav_target = "1. Overview"
            st.rerun()

    with sc_cols[3]:
        st.markdown("""
            <div class="scenario-card">
                <span class="badge" style="background: #fff7ed; color: #c2410c;">🌀 CYCLONE GALE</span>
                <h4 style="margin: 8px 0 4px 0; color: #0f172a; font-size: 1.05rem;">Coastal Odisha Storm</h4>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Post-monsoon Bay of Bengal low-pressure system triggering gale warnings.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Test Scenario 4", key="hm_sc4", use_container_width=True):
            apply_scenario("Scenario 4: Coastal Odisha (High Wind / Cyclone)")
            st.session_state.nav_target = "1. Overview"
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;'>🔬 Core Pillars of AtmosBlend AI</h3>", unsafe_allow_html=True)
    pil_cols = st.columns(3)
    with pil_cols[0]:
        st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 10px;">🧩</div>
                <h4 style="margin: 0 0 6px 0; color: #0f172a; font-size: 1.1rem;">Multi-Source Model Synergy</h4>
                <p style="color: #64748b; font-size: 0.88rem; line-height: 1.5; margin: 0;">
                    Instead of relying solely on physical NWP or pure data-driven AI, AtmosBlend fuses Numerical Weather Prediction, Random Forest Regressors, and 5-Member Ensembles into a unified prediction pipeline.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with pil_cols[1]:
        st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 10px;">⚖️</div>
                <h4 style="margin: 0 0 6px 0; color: #0f172a; font-size: 1.1rem;">Adaptive Regime Weighting</h4>
                <p style="color: #64748b; font-size: 0.88rem; line-height: 1.5; margin: 0;">
                    Weights dynamically shift according to historical verification errors: <code>Reliability = 1 / (MAE + 0.5)</code> conditioned on geographic region, lead time, season, and active weather regimes.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with pil_cols[2]:
        st.markdown("""
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 10px;">🚨</div>
                <h4 style="margin: 0 0 6px 0; color: #0f172a; font-size: 1.1rem;">IMD-Calibrated Risk Sentry</h4>
                <p style="color: #64748b; font-size: 0.88rem; line-height: 1.5; margin: 0;">
                    Automated multi-hazard threat scoring aligned with Indian Meteorological Department (IMD) standards for Heavy Rain (>204mm), Heat Waves (>45°C), and Gale Winds (>62km/h).
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;'>📂 Explore Dashboard Capabilities</h3>", unsafe_allow_html=True)
    mod_cols1 = st.columns(4)
    with mod_cols1[0]:
        st.markdown("""
            <div class="feature-card">
                <h5 style="margin: 0 0 4px 0; color: #0f172a;">1. Executive Overview</h5>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Multi-model synthesis, comparison charts, and component forecast matrix.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Overview", key="hm_mod1", use_container_width=True):
            st.session_state.nav_target = "1. Overview"
            st.rerun()

    with mod_cols1[1]:
        st.markdown("""
            <div class="feature-card">
                <h5 style="margin: 0 0 4px 0; color: #0f172a;">2. Forecast Curves</h5>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Lead-time trajectories from 6h to 72h with 5-member uncertainty dispersion.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Curves", key="hm_mod2", use_container_width=True):
            st.session_state.nav_target = "2. Forecast Curves"
            st.rerun()

    with mod_cols1[2]:
        st.markdown("""
            <div class="feature-card">
                <h5 style="margin: 0 0 4px 0; color: #0f172a;">3. Model Intelligence</h5>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Historical MAE/RMSE comparisons & AI feature importance breakdown.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Intelligence", key="hm_mod3", use_container_width=True):
            st.session_state.nav_target = "3. Model Intelligence"
            st.rerun()

    with mod_cols1[3]:
        st.markdown("""
            <div class="feature-card">
                <h5 style="margin: 0 0 4px 0; color: #0f172a;">4. Adaptive Weights</h5>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Mathematical weight derivation and meteorological rationale explanations.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Weights", key="hm_mod4", use_container_width=True):
            st.session_state.nav_target = "4. Adaptive Weights"
            st.rerun()

    mod_cols2 = st.columns(4)
    with mod_cols2[0]:
        st.markdown("""
            <div class="feature-card" style="margin-top: 10px;">
                <h5 style="margin: 0 0 4px 0; color: #0f172a;">5. Regional Weight Map</h5>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Pan-India geospatial distribution of dominant model allocations.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Map", key="hm_mod5", use_container_width=True):
            st.session_state.nav_target = "5. Regional Weight Map"
            st.rerun()

    with mod_cols2[1]:
        st.markdown("""
            <div class="feature-card" style="margin-top: 10px;">
                <h5 style="margin: 0 0 4px 0; color: #0f172a;">6. Extreme Weather</h5>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Multi-hazard thresholds, active color warnings, and advisory bulletins.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Hazard Center", key="hm_mod6", use_container_width=True):
            st.session_state.nav_target = "6. Extreme Weather Center"
            st.rerun()

    with mod_cols2[2]:
        st.markdown("""
            <div class="feature-card" style="margin-top: 10px;">
                <h5 style="margin: 0 0 4px 0; color: #0f172a;">7. Verification</h5>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Rigorous benchmark comparing Hybrid blend against individual systems.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Verification", key="hm_mod7", use_container_width=True):
            st.session_state.nav_target = "7. Performance Verification"
            st.rerun()

    with mod_cols2[3]:
        st.markdown("""
            <div class="feature-card" style="margin-top: 10px;">
                <h5 style="margin: 0 0 4px 0; color: #0f172a;">8. Pipeline Architecture</h5>
                <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">Complete 9-stage scientific architecture and LaTeX formulations.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Open Architecture", key="hm_mod8", use_container_width=True):
            st.session_state.nav_target = "8. Methodology & Pipeline"
            st.rerun()

# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -----------------------------------------------------------------------------
elif selected_page == "1. Overview":
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;'>Executive Meteorological Overview</h3>", unsafe_allow_html=True)
    
    col_ov1, col_ov2 = st.columns([1.5, 1])
    with col_ov1:
        st.markdown("#### Multi-Source Forecast Synthesis")
        models_data = {
            "Model Source": ["NWP Baseline", "AI/ML (Random Forest)", "Ensemble (5-Member)", "Hybrid Forecast (AtmosBlend)"],
            f"Forecast Value ({unit})": [active_nwp, active_ai, active_ens, active_hybrid],
            "Type": ["Constituent", "Constituent", "Constituent", "Blended Result"]
        }
        df_comp = pd.DataFrame(models_data)
        fig_bar = px.bar(
            df_comp,
            x="Model Source",
            y=f"Forecast Value ({unit})",
            color="Model Source",
            color_discrete_sequence=["#60a5fa", "#34d399", "#a78bfa", "#38bdf8"],
            text_auto=True
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e293b", family="Outfit, sans-serif"),
            showlegend=False,
            height=340,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        fig_bar.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_ov2:
        st.markdown("#### Adaptive Weight Allocation")
        w_df = pd.DataFrame({
            "Model": ["NWP", "AI/ML", "Ensemble"],
            "Weight": [active_weights["NWP"], active_weights["AI"], active_weights["Ensemble"]]
        })
        fig_donut = px.pie(
            w_df,
            names="Model",
            values="Weight",
            hole=0.55,
            color="Model",
            color_discrete_map={"NWP": "#60a5fa", "AI/ML": "#34d399", "Ensemble": "#a78bfa"}
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e293b", family="Outfit, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("#### Component Model Forecast Matrix")
    overview_table = pd.DataFrame({
        "Meteorological Variable": ["Rainfall", "Temperature", "Wind Speed"],
        "NWP Baseline": [f"{nwp_rain:.1f} mm", f"{nwp_temp:.1f} °C", f"{nwp_wind:.1f} km/h"],
        "AI/ML Regressor": [f"{ai_rain:.1f} mm", f"{ai_temp:.1f} °C", f"{ai_wind:.1f} km/h"],
        "Ensemble Mean (±Spread)": [
            f"{ens_rain:.1f} mm (±{ens_breakdown['Rainfall']['spread']:.1f})",
            f"{ens_temp:.1f} °C (±{ens_breakdown['Temperature']['spread']:.1f})",
            f"{ens_wind:.1f} km/h (±{ens_breakdown['Wind']['spread']:.1f})"
        ],
        "Hybrid Blend": [f"{hybrid_rain:.1f} mm", f"{hybrid_temp:.1f} °C", f"{hybrid_wind:.1f} km/h"],
        "Dominant Model": [
            f"{max(weights_rain, key=weights_rain.get)} ({weights_rain[max(weights_rain, key=weights_rain.get)]:.1f}%)",
            f"{max(weights_temp, key=weights_temp.get)} ({weights_temp[max(weights_temp, key=weights_temp.get)]:.1f}%)",
            f"{max(weights_wind, key=weights_wind.get)} ({weights_wind[max(weights_wind, key=weights_wind.get)]:.1f}%)"
        ],
        "Historical Actual (Ref)": [f"{actual_rain:.1f} mm", f"{actual_temp:.1f} °C", f"{actual_wind:.1f} km/h"]
    })
    st.dataframe(overview_table, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PAGE 2: FORECAST CURVES
# -----------------------------------------------------------------------------
elif selected_page == "2. Forecast Curves":
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;'>Multi-Model Lead Time Trajectory</h3>", unsafe_allow_html=True)
    st.markdown("Visualizing NWP, AI, Ensemble, and the resulting Hybrid blend across forecast horizons (6h to 72h).")

    lt_horizons = ["6 hours", "12 hours", "24 hours", "48 hours", "72 hours"]
    lt_data = []
    for lt in lt_horizons:
        sub_lt = df_clean[
            (df_clean["Region"] == cur_region) &
            (df_clean["Season"] == cur_season) &
            (df_clean["Weather_Regime"] == cur_regime) &
            (df_clean["Lead_Time"] == lt)
        ]
        row_lt = sub_lt.iloc[0] if len(sub_lt) > 0 else sample_row
        v_nwp = float(row_lt.get(f"NWP_{cur_var}", active_nwp))
        v_ai = float(row_lt.get(f"AI_{cur_var}", active_ai))
        v_ens = float(row_lt.get(f"Ensemble_{cur_var}", active_ens))
        v_act = float(row_lt.get(f"Actual_{cur_var}", active_actual))
        w_lt, _, _ = weight_engine.compute_adaptive_weights(cur_region, cur_season, lt, cur_regime, cur_var)
        v_hyb = weight_engine.compute_hybrid_forecast(v_nwp, v_ai, v_ens, w_lt, cur_var)

        lt_data.append({
            "Lead Time": lt,
            "NWP Forecast": v_nwp,
            "AI/ML Forecast": v_ai,
            "Ensemble Forecast": v_ens,
            "Hybrid Blend": v_hyb,
            "Actual Observation": v_act
        })

    df_lt = pd.DataFrame(lt_data)
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=df_lt["Lead Time"], y=df_lt["NWP Forecast"], mode="lines+markers", name="NWP (Physics Baseline)", line=dict(color="#60a5fa", dash="dot", width=2)))
    fig_line.add_trace(go.Scatter(x=df_lt["Lead Time"], y=df_lt["AI/ML Forecast"], mode="lines+markers", name="AI/ML (Random Forest)", line=dict(color="#34d399", dash="dash", width=2)))
    fig_line.add_trace(go.Scatter(x=df_lt["Lead Time"], y=df_lt["Ensemble Forecast"], mode="lines+markers", name="Ensemble (5-Member Mean)", line=dict(color="#a78bfa", dash="dashdot", width=2)))
    fig_line.add_trace(go.Scatter(x=df_lt["Lead Time"], y=df_lt["Hybrid Blend"], mode="lines+markers", name="Hybrid Forecast (AtmosBlend)", line=dict(color="#38bdf8", width=4)))
    fig_line.add_trace(go.Scatter(x=df_lt["Lead Time"], y=df_lt["Actual Observation"], mode="markers", name="Actual Observation", marker=dict(color="#f43f5e", size=9, symbol="diamond")))

    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", family="Outfit, sans-serif"),
        title=f"Evolution of {cur_var} Forecasts Across Lead Times ({cur_region} - {cur_regime})",
        xaxis_title="Forecast Lead Time",
        yaxis_title=f"{cur_var} ({unit})",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=450
    )
    fig_line.update_xaxes(gridcolor="#e2e8f0")
    fig_line.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("#### 5-Member Ensemble Uncertainty Spread")
    ens_members = ens_breakdown[cur_var]["members"]
    ens_df = pd.DataFrame({
        "Member": list(ens_members.keys()),
        f"Value ({unit})": list(ens_members.values())
    })
    
    col_e1, col_e2 = st.columns([1.5, 1])
    with col_e1:
        fig_ens = px.bar(
            ens_df,
            x="Member",
            y=f"Value ({unit})",
            color="Member",
            color_discrete_sequence=px.colors.sequential.Purples_r,
            text_auto=True
        )
        fig_ens.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e293b", family="Outfit, sans-serif"),
            showlegend=False,
            height=280
        )
        st.plotly_chart(fig_ens, use_container_width=True)

    with col_e2:
        st.markdown(f"""
            <div class="kpi-card" style="margin-top: 10px;">
                <div class="kpi-title">Ensemble Uncertainty Statistics</div>
                <div style="font-size: 0.95rem; margin-top: 10px; line-height: 1.8;">
                    • <b>Source:</b> Ensemble Forecast – Prototype<br/>
                    • <b>Ensemble Mean:</b> {ens_breakdown[cur_var]['mean']} {unit}<br/>
                    • <b>Ensemble Spread (σ):</b> ±{ens_breakdown[cur_var]['spread']} {unit}<br/>
                    • <b>Member Min / Max:</b> {min(ens_members.values())} to {max(ens_members.values())} {unit}<br/>
                    • <b>Spread Interpretation:</b> {'High dispersion (Atmospheric Bifurcation)' if ens_breakdown[cur_var]['spread'] > 3.0 else 'High consensus / tight cluster'}
                </div>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 3: MODEL INTELLIGENCE
# -----------------------------------------------------------------------------
elif selected_page == "3. Model Intelligence":
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;'>Historical Model Performance & Verification</h3>", unsafe_allow_html=True)
    st.markdown(f"Comparing NWP, AI/ML, and Ensemble historical errors under **{cur_region} | {cur_season} | {cur_lead_time} | {cur_regime}**.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        err_table = pd.DataFrame({
            "Forecast Source": ["NWP Baseline", "AI/ML (Random Forest)", "Ensemble (Prototype)"],
            f"MAE ({unit})": [active_errors["NWP"]["MAE"], active_errors["AI"]["MAE"], active_errors["Ensemble"]["MAE"]],
            f"RMSE ({unit})": [active_errors["NWP"]["RMSE"], active_errors["AI"]["RMSE"], active_errors["Ensemble"]["RMSE"]],
            "Historical Samples": [active_errors["NWP"]["samples_count"], active_errors["AI"]["samples_count"], active_errors["Ensemble"]["samples_count"]]
        })
        st.markdown("#### Historical Error Metrics Table")
        st.dataframe(err_table, use_container_width=True, hide_index=True)
        st.caption("ℹ️ Lower error indicates higher historical accuracy for this contextual weather pattern.")

    with col_m2:
        plot_err = pd.DataFrame([
            {"Model": "NWP", "Metric": "MAE", "Error": active_errors["NWP"]["MAE"]},
            {"Model": "NWP", "Metric": "RMSE", "Error": active_errors["NWP"]["RMSE"]},
            {"Model": "AI/ML", "Metric": "MAE", "Error": active_errors["AI"]["MAE"]},
            {"Model": "AI/ML", "Metric": "RMSE", "Error": active_errors["AI"]["RMSE"]},
            {"Model": "Ensemble", "Metric": "MAE", "Error": active_errors["Ensemble"]["MAE"]},
            {"Model": "Ensemble", "Metric": "RMSE", "Error": active_errors["Ensemble"]["RMSE"]}
        ])
        fig_err = px.bar(
            plot_err,
            x="Model",
            y="Error",
            color="Metric",
            barmode="group",
            color_discrete_map={"MAE": "#38bdf8", "RMSE": "#f43f5e"},
            text_auto=True
        )
        fig_err.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e293b", family="Outfit, sans-serif"),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.markdown("#### Error Comparison Chart")
        st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("---")
    st.markdown("#### AI/ML Model Top Feature Importances (Random Forest)")
    if cur_var in ai_engine.feature_importances:
        feat_dict = ai_engine.feature_importances[cur_var]
        feat_df = pd.DataFrame({
            "Feature": list(feat_dict.keys()),
            "Importance Weight": list(feat_dict.values())
        }).sort_values(by="Importance Weight", ascending=True)

        fig_feat = px.bar(
            feat_df,
            x="Importance Weight",
            y="Feature",
            orientation="h",
            color="Importance Weight",
            color_continuous_scale="Viridis"
        )
        fig_feat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e293b", family="Outfit, sans-serif"),
            height=320,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_feat, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4: ADAPTIVE WEIGHTS
# -----------------------------------------------------------------------------
elif selected_page == "4. Adaptive Weights":
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;'>Dynamic Weight Allocation Engine</h3>", unsafe_allow_html=True)
    st.markdown(
        "Weights are dynamically recomputed in real time based on historical model error: "
        "`Reliability = 1 / (MAE + 0.5)` normalized so `W_NWP + W_AI + W_Ensemble = 100%`."
    )

    col_w1, col_w2 = st.columns([1, 1])
    with col_w1:
        st.markdown("#### Assigned Weights Distribution")
        w_chart_df = pd.DataFrame({
            "Model": ["NWP Baseline", "AI/ML Regressor", "Ensemble (Prototype)"],
            "Weight (%)": [active_weights["NWP"], active_weights["AI"], active_weights["Ensemble"]],
            "Color": ["#60a5fa", "#34d399", "#a78bfa"]
        })
        fig_wbar = px.bar(
            w_chart_df,
            x="Model",
            y="Weight (%)",
            color="Model",
            color_discrete_map={"NWP Baseline": "#60a5fa", "AI/ML Regressor": "#34d399", "Ensemble (Prototype)": "#a78bfa"},
            text_auto=True
        )
        fig_wbar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1e293b", family="Outfit, sans-serif"),
            showlegend=False,
            height=320
        )
        st.plotly_chart(fig_wbar, use_container_width=True)

    with col_w2:
        st.markdown("#### Mathematical Reliability Scaling")
        rel_nwp = 1.0 / (active_errors["NWP"]["MAE"] + 0.5)
        rel_ai = 1.0 / (active_errors["AI"]["MAE"] + 0.5)
        rel_ens = 1.0 / (active_errors["Ensemble"]["MAE"] + 0.5)
        sum_rel = rel_nwp + rel_ai + rel_ens

        st.markdown(f"""
            <div class="kpi-card" style="font-size: 0.9rem; line-height: 1.7;">
                <div class="kpi-title">Reliability Derivation (ε = 0.5)</div>
                • <b>NWP Reliability:</b> 1 / ({active_errors['NWP']['MAE']} + 0.5) = <b>{rel_nwp:.4f}</b><br/>
                • <b>AI Reliability:</b> 1 / ({active_errors['AI']['MAE']} + 0.5) = <b>{rel_ai:.4f}</b><br/>
                • <b>Ensemble Reliability:</b> 1 / ({active_errors['Ensemble']['MAE']} + 0.5) = <b>{rel_ens:.4f}</b><br/>
                • <b>Sum of Reliabilities:</b> {sum_rel:.4f}<br/>
                <hr style="border-color: #e2e8f0; margin: 8px 0;"/>
                • <b>NWP Weight:</b> ({rel_nwp:.4f} / {sum_rel:.4f}) × 100 = <b style="color: #60a5fa;">{active_weights['NWP']}%</b><br/>
                • <b>AI Weight:</b> ({rel_ai:.4f} / {sum_rel:.4f}) × 100 = <b style="color: #34d399;">{active_weights['AI']}%</b><br/>
                • <b>Ensemble Weight:</b> ({rel_ens:.4f} / {sum_rel:.4f}) × 100 = <b style="color: #a78bfa;">{active_weights['Ensemble']}%</b>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💡 Meteorological Rationale: *Why these weights?*")
    st.info(active_rationale)

# -----------------------------------------------------------------------------
# PAGE 5: REGIONAL WEIGHT MAP
# -----------------------------------------------------------------------------
elif selected_page == "5. Regional Weight Map":
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;'>Pan-India Dominant Model Weight Map</h3>", unsafe_allow_html=True)
    st.caption("Clearly Labeled: Prototype Model Weight Map (Simulated Geographic Distribution)")

    from src.data_generator import REGIONS_META
    map_records = []
    for reg_name, meta in REGIONS_META.items():
        w_reg, err_reg, _ = weight_engine.compute_adaptive_weights(
            reg_name, cur_season, cur_lead_time, cur_regime, cur_var
        )
        dom = max(w_reg, key=w_reg.get)
        dom_w = w_reg[dom]
        map_records.append({
            "Region": reg_name,
            "Latitude": meta["lat"],
            "Longitude": meta["lon"],
            "Dominant Model": f"{dom} Dominant",
            "Dominant Weight (%)": dom_w,
            "NWP Weight": f"{w_reg['NWP']}%",
            "AI Weight": f"{w_reg['AI']}%",
            "Ensemble Weight": f"{w_reg['Ensemble']}%",
            "Marker Size": dom_w * 0.8
        })

    df_map = pd.DataFrame(map_records)
    fig_map = px.scatter_geo(
        df_map,
        lat="Latitude",
        lon="Longitude",
        color="Dominant Model",
        size="Dominant Weight (%)",
        hover_name="Region",
        hover_data={"Latitude": False, "Longitude": False, "Dominant Weight (%)": True, "NWP Weight": True, "AI Weight": True, "Ensemble Weight": True},
        color_discrete_map={
            "NWP Dominant": "#60a5fa",
            "AI Dominant": "#34d399",
            "Ensemble Dominant": "#a78bfa"
        },
        scope="asia"
    )
    fig_map.update_geos(
        center=dict(lat=21.0, lon=79.5),
        projection_scale=3.8,
        visible=False,
        showcountries=True,
        countrycolor="#94a3b8",
        showland=True,
        landcolor="#f8fafc",
        showocean=True,
        oceancolor="#e0f2fe",
        showsubunits=True,
        subunitcolor="#cbd5e1"
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b", family="Outfit, sans-serif"),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("#### Regional Weighting Breakdown")
    st.dataframe(df_map[["Region", "Dominant Model", "Dominant Weight (%)", "NWP Weight", "AI Weight", "Ensemble Weight"]], use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PAGE 6: EXTREME WEATHER CENTER
# -----------------------------------------------------------------------------
elif selected_page == "6. Extreme Weather Center":
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;'>Severe Weather Warning Center & Early Advisory</h3>", unsafe_allow_html=True)
    st.caption("Calibrated against MoES / IMD Prototype Thresholds for Multi-Hazard Disaster Mitigation")

    h_cols = st.columns(4)
    with h_cols[0]:
        rain_val = hybrid_rain
        if rain_val > 204.4:
            r_tag, r_col = "Red Alert (>204mm)", "#ef4444"
        elif rain_val >= 115.6:
            r_tag, r_col = "Orange Alert (115-204mm)", "#f97316"
        elif rain_val >= 64.5:
            r_tag, r_col = "Yellow Alert (64-115mm)", "#eab308"
        else:
            r_tag, r_col = "Green (Normal)", "#22c55e"
            
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid {r_col};">
                <div class="kpi-title">🌧️ Heavy Rain Threat</div>
                <div class="kpi-value">{rain_val:.1f} <span style="font-size: 1rem;">mm</span></div>
                <div style="margin-top: 8px; color: {r_col}; font-weight: 700; font-size: 0.85rem;">{r_tag}</div>
            </div>
        """, unsafe_allow_html=True)

    with h_cols[1]:
        temp_val = hybrid_temp
        if temp_val >= 45.0:
            t_tag, t_col = "Red Alert (Severe Heat)", "#ef4444"
        elif temp_val >= 43.0:
            t_tag, t_col = "Orange Alert (Heat Wave)", "#f97316"
        elif temp_val >= 40.0:
            t_tag, t_col = "Yellow Alert (Hot Day)", "#eab308"
        else:
            t_tag, t_col = "Green (Normal)", "#22c55e"

        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid {t_col};">
                <div class="kpi-title">🌡️ Heat Wave Threat</div>
                <div class="kpi-value">{temp_val:.1f} <span style="font-size: 1rem;">°C</span></div>
                <div style="margin-top: 8px; color: {t_col}; font-weight: 700; font-size: 0.85rem;">{t_tag}</div>
            </div>
        """, unsafe_allow_html=True)

    with h_cols[2]:
        wind_val = hybrid_wind
        if wind_val >= 88.0:
            w_tag, w_col = "Red Alert (Storm Force)", "#ef4444"
        elif wind_val >= 62.0:
            w_tag, w_col = "Orange Alert (Gale Wind)", "#f97316"
        elif wind_val >= 45.0:
            w_tag, w_col = "Yellow Alert (Strong Breeze)", "#eab308"
        else:
            w_tag, w_col = "Green (Normal)", "#22c55e"

        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid {w_col};">
                <div class="kpi-title">💨 High Wind Threat</div>
                <div class="kpi-value">{wind_val:.1f} <span style="font-size: 1rem;">km/h</span></div>
                <div style="margin-top: 8px; color: {w_col}; font-weight: 700; font-size: 0.85rem;">{w_tag}</div>
            </div>
        """, unsafe_allow_html=True)

    with h_cols[3]:
        is_cyclone = (wind_val >= 62.0 and rain_val >= 60.0) or (cur_regime == "Storm/Cyclone" and wind_val >= 50.0)
        c_tag = "ACTIVE CYCLONE SIGNAL" if is_cyclone else "NO CYCLONE DETECTED"
        c_col = "#ef4444" if is_cyclone else "#22c55e"

        st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid {c_col};">
                <div class="kpi-title">🌀 Storm/Cyclone Signal</div>
                <div class="kpi-value" style="font-size: 1.4rem; color: {c_col};">{c_tag}</div>
                <div style="margin-top: 8px; color: #94a3b8; font-size: 0.8rem;">Compound Rain + Wind Index</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Active Hazard Advisory Bulletins")
    for al in risk_info["alerts"]:
        a_border = "alert-red" if al["risk"] == "Severe" else ("alert-orange" if al["risk"] == "High" else ("alert-yellow" if al["risk"] == "Moderate" else "alert-green"))
        st.markdown(f"""
            <div class="alert-card {a_border}">
                <div style="font-weight: 800; font-size: 1.1rem;">⚡ {al['title']}</div>
                <div style="margin-top: 6px; font-size: 0.95rem;">{al['message']}</div>
                <div style="margin-top: 8px; font-size: 0.82rem; opacity: 0.85;">
                    <b>Region:</b> {cur_region} | <b>Lead Time:</b> {cur_lead_time} | <b>Forecast Values:</b> Rain: {hybrid_rain:.1f} mm, Temp: {hybrid_temp:.1f} °C, Wind: {hybrid_wind:.1f} km/h
                </div>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 7: PERFORMANCE VERIFICATION
# -----------------------------------------------------------------------------
elif selected_page == "7. Performance Verification":
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;'>Multi-Model Verification: NWP vs AI vs Ensemble vs Hybrid</h3>", unsafe_allow_html=True)
    st.markdown("Rigorous benchmark evaluating whether the adaptive **Hybrid Forecast** systematically outperforms individual forecast systems.")

    overall_metrics = evaluator.evaluate_overall_models()
    for var_name in ["Rainfall", "Temperature", "Wind"]:
        m_dict = overall_metrics[var_name]
        u = "mm" if var_name == "Rainfall" else ("°C" if var_name == "Temperature" else "km/h")
        
        st.markdown(f"#### {var_name} Forecast Verification")
        comp_df = pd.DataFrame({
            "Model System": ["NWP Baseline", "AI/ML (Random Forest)", "Ensemble (Prototype)", "Hybrid Blend (AtmosBlend)"],
            f"MAE ({u})": [m_dict["NWP"]["MAE"], m_dict["AI"]["MAE"], m_dict["Ensemble"]["MAE"], m_dict["Hybrid"]["MAE"]],
            f"RMSE ({u})": [m_dict["NWP"]["RMSE"], m_dict["AI"]["RMSE"], m_dict["Ensemble"]["RMSE"], m_dict["Hybrid"]["RMSE"]]
        })
        
        col_p1, col_p2 = st.columns([1.6, 1])
        with col_p1:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        with col_p2:
            st.markdown(f"""
                <div class="kpi-card" style="padding: 14px 18px;">
                    <div class="kpi-title">{var_name} Hybrid Skill Score</div>
                    <div style="font-size: 0.95rem; margin-top: 6px; line-height: 1.6;">
                        • <b>Best Constituent:</b> {m_dict['best_individual_model']}<br/>
                        • <b>Hybrid MAE:</b> {m_dict['Hybrid']['MAE']} {u}<br/>
                        • <b>Hybrid RMSE:</b> {m_dict['Hybrid']['RMSE']} {u}<br/>
                        • <b>Error Reduction:</b> Blending eliminates outlier extreme drift across multi-regime transitions.
                    </div>
                </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 8: METHODOLOGY & PIPELINE
# -----------------------------------------------------------------------------
elif selected_page == "8. Methodology & Pipeline":
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;'>System Architecture & Scientific Methodology</h3>", unsafe_allow_html=True)
    st.markdown("Problem Statement: 26081 • Ministry of Earth Sciences (MoES) / NCMRWF")

    st.markdown("""
    ```
    ┌───────────────────────────┐
    │ 1. Data Collection        │  Observational AWS Data + NWP Forecast Feeds
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │ 2. Data Preprocessing     │  Missing Value Imputation, Bounds Check, Cyclical Features
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │ 3. Constituent Forecasts  │  NWP (Physics) + AI/ML (Random Forest) + Ensemble (5 Members)
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │ 4. Error Calculation      │  Stratified Historical MAE / RMSE Evaluation
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │ 5. Context Analysis       │  Conditioned on Region, Season, Lead Time, Weather Regime
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │ 6. Adaptive Weighting     │  Reliability = 1 / (MAE + ε)  ->  Normalized Sum = 100%
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │ 7. Hybrid Forecast Blend  │  Linear Optimal Combination across Rain, Temp, Wind
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │ 8. Extreme Weather Engine │  IMD Standard Hazard Thresholds -> Risk & Alert Bulletins
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │ 9. Interactive Dashboard  │  Streamlit Executive Decision Support + Power BI Export
    └───────────────────────────┘
    ```
    """)

    st.markdown("#### Mathematical Formulation")
    st.latex(r"""
    	ext{Reliability}_{m} = rac{1}{	ext{Error}_{m}(	ext{Region}, 	ext{Season}, 	ext{LeadTime}, 	ext{Regime}) + \epsilon}
    """)
    st.latex(r"""
    W_{m} = rac{	ext{Reliability}_{m}}{\sum_{k \in \{	ext{NWP}, 	ext{AI}, 	ext{ENS}\}} 	ext{Reliability}_{k}} 	imes 100\%
    """)
    st.latex(r"""
    	ext{Hybrid Forecast} = \left( W_{	ext{NWP}} 	imes 	ext{NWP} ight) + \left( W_{	ext{AI}} 	imes 	ext{AI} ight) + \left( W_{	ext{ENS}} 	imes 	ext{ENS} ight)
    """)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748b;">
        <div><b>AtmosBlend AI v1.0.0</b> | Ministry of Earth Sciences (MoES) - NCMRWF Prototype</div>
        <div>Problem Statement: 26081 | Designed for Pair-Programming & Hackathon Evaluation</div>
    </div>
""", unsafe_allow_html=True)
