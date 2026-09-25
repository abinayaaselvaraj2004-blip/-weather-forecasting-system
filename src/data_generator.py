"""
Data Generator for AtmosBlend AI
Simulates meteorologically consistent multi-model forecast and observation datasets.
Clearly marked as: DEMO / SIMULATED DATA - Prototype for Hackathon Evaluation.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

REGIONS_META = {
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569, "coastal": True, "cyclone_prone": True, "ne_monsoon": True},
    "Kerala": {"lat": 10.8505, "lon": 76.2711, "coastal": True, "cyclone_prone": False, "sw_monsoon_heavy": True},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139, "coastal": True, "cyclone_prone": False},
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400, "coastal": True, "cyclone_prone": True},
    "Telangana": {"lat": 18.1124, "lon": 79.0193, "coastal": False, "cyclone_prone": False, "heat_prone": True},
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139, "coastal": True, "cyclone_prone": False},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179, "coastal": False, "cyclone_prone": False, "arid": True, "heat_prone": True},
    "Odisha": {"lat": 20.9517, "lon": 85.0985, "coastal": True, "cyclone_prone": True}
}

SEASONS = ["Winter", "Summer", "Southwest Monsoon", "Northeast Monsoon", "Post-Monsoon"]
LEAD_TIMES = ["6 hours", "12 hours", "24 hours", "48 hours", "72 hours"]
WEATHER_REGIMES = ["Normal", "Heavy Rain", "Heat Wave", "High Wind", "Storm/Cyclone"]

def get_season_for_date(dt: datetime, region: str) -> str:
    month = dt.month
    if month in [1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Southwest Monsoon"
    else:  # Oct, Nov, Dec
        if region in ["Tamil Nadu", "Andhra Pradesh"]:
            return "Northeast Monsoon"
        else:
            return "Post-Monsoon"

def generate_weather_data(n_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    start_date = datetime(2024, 1, 1)
    records = []
    
    lead_time_hours_map = {
        "6 hours": 6,
        "12 hours": 12,
        "24 hours": 24,
        "48 hours": 48,
        "72 hours": 72
    }
    
    regions_list = list(REGIONS_META.keys())
    
    for i in range(n_samples):
        # Pick random region and date
        region = np.random.choice(regions_list)
        meta = REGIONS_META[region]
        day_offset = np.random.randint(0, 365)
        hour = np.random.choice([0, 6, 12, 18])
        dt = start_date + timedelta(days=day_offset, hours=int(hour))
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M")
        
        season = get_season_for_date(dt, region)
        lead_time = np.random.choice(LEAD_TIMES)
        lt_hours = lead_time_hours_map[lead_time]
        
        # Decide regime with probabilistic weighting per season and region
        if season == "Summer" and meta.get("heat_prone", False):
            regime_probs = [0.40, 0.05, 0.45, 0.08, 0.02]
        elif (season in ["Southwest Monsoon", "Northeast Monsoon"]) and meta.get("coastal", False):
            regime_probs = [0.35, 0.40, 0.02, 0.15, 0.08]
        elif meta.get("cyclone_prone", False) and season in ["Northeast Monsoon", "Post-Monsoon", "Summer"]:
            regime_probs = [0.50, 0.20, 0.05, 0.15, 0.10]
        else:
            regime_probs = [0.65, 0.15, 0.10, 0.08, 0.02]
            
        regime = np.random.choice(WEATHER_REGIMES, p=regime_probs)
        
        # Base realistic physical climatology
        # Base temperature by region & season
        if season == "Winter":
            base_temp = 20.0 if not meta.get("arid", False) else 15.0
        elif season == "Summer":
            base_temp = 41.0 if meta.get("heat_prone", False) else 34.0
        elif season == "Southwest Monsoon":
            base_temp = 28.0
        else:
            base_temp = 27.0
            
        # Diurnal cycle
        if hour == 12:
            diurnal = 4.5
        elif hour in [6, 18]:
            diurnal = 0.5
        else:
            diurnal = -3.5
            
        actual_temp = base_temp + diurnal + np.random.normal(0, 1.8)
        
        # Base wind speed (km/h)
        base_wind = 12.0
        if meta.get("coastal", False):
            base_wind += 6.0
            
        # Base rainfall (mm)
        base_rain = 0.0
        if season == "Southwest Monsoon":
            base_rain = 25.0 if region in ["Kerala", "Maharashtra", "Karnataka"] else 12.0
        elif season == "Northeast Monsoon" and region in ["Tamil Nadu", "Andhra Pradesh"]:
            base_rain = 35.0
            
        # Apply Weather Regime shifts
        if regime == "Heavy Rain":
            actual_rain = max(0.0, base_rain + np.random.uniform(50.0, 130.0))
            actual_temp -= np.random.uniform(3.0, 6.0)
            actual_wind = max(5.0, base_wind + np.random.uniform(15.0, 35.0))
        elif regime == "Heat Wave":
            actual_temp = max(actual_temp, 42.0 + np.random.uniform(1.5, 6.5))
            actual_rain = 0.0
            actual_wind = max(3.0, base_wind + np.random.normal(0, 3.0))
        elif regime == "High Wind":
            actual_wind = max(35.0, base_wind + np.random.uniform(30.0, 55.0))
            actual_rain = max(0.0, base_rain * 0.5 + np.random.uniform(0.0, 20.0))
            actual_temp += np.random.normal(0, 2.0)
        elif regime == "Storm/Cyclone":
            actual_rain = max(40.0, base_rain + np.random.uniform(80.0, 180.0))
            actual_wind = max(60.0, base_wind + np.random.uniform(45.0, 85.0))
            actual_temp = 25.0 + np.random.normal(0, 2.0)
        else: # Normal
            actual_rain = max(0.0, base_rain * 0.3 + np.random.exponential(4.0)) if base_rain > 0 else max(0.0, np.random.exponential(1.2) - 0.8)
            actual_wind = max(3.0, base_wind + np.random.normal(0, 4.0))
            
        # Bound physically realistic limits
        actual_temp = np.clip(actual_temp, 8.0, 52.0)
        actual_rain = np.clip(round(actual_rain, 1), 0.0, 350.0)
        actual_wind = np.clip(round(actual_wind, 1), 2.0, 160.0)
        actual_temp = round(actual_temp, 1)

        # -------------------------------------------------------------
        # Simulate Multi-Model Forecasts with realistic error dynamics
        # -------------------------------------------------------------
        # Error degradation factor by lead time
        lt_err_factor = 1.0 + (lt_hours / 72.0) * 0.8
        
        # 1. NWP Forecast: Strong physical dynamics, performs great at 48h-72h synoptic,
        # but can suffer localized convective displacement for rain at short lead times.
        nwp_rain_err = np.random.normal(0, 4.0 * lt_err_factor)
        if regime in ["Heavy Rain", "Storm/Cyclone"]:
            nwp_rain_err += np.random.normal(-5.0, 12.0)  # slight underestimation of convective peaks
        nwp_temp_err = np.random.normal(0, 1.2 * lt_err_factor)
        nwp_wind_err = np.random.normal(0, 3.5 * lt_err_factor)
        
        nwp_rain = max(0.0, round(actual_rain + nwp_rain_err, 1))
        nwp_temp = round(actual_temp + nwp_temp_err, 1)
        nwp_wind = max(1.0, round(actual_wind + nwp_wind_err, 1))
        
        # 2. AI/ML Forecast: Excels at short lead times (6-24h) and non-linear local features,
        # but slightly larger drift at 72h.
        ai_lt_factor = 0.8 if lt_hours <= 24 else (1.4 + (lt_hours - 24)/48.0)
        ai_rain_err = np.random.normal(0, 3.2 * ai_lt_factor)
        ai_temp_err = np.random.normal(0, 0.9 * ai_lt_factor)
        ai_wind_err = np.random.normal(0, 2.8 * ai_lt_factor)
        
        ai_rain = max(0.0, round(actual_rain + ai_rain_err, 1))
        ai_temp = round(actual_temp + ai_temp_err, 1)
        ai_wind = max(1.0, round(actual_wind + ai_wind_err, 1))
        
        # 3. Ensemble Forecast: 5 members with perturbations
        ens_members_rain = []
        ens_members_temp = []
        ens_members_wind = []
        
        for m in range(1, 6):
            # Member perturbations
            m_spread = 1.0 + (m - 3) * 0.15
            m_rain = max(0.0, round(actual_rain + np.random.normal(0, 3.5 * lt_err_factor * m_spread), 1))
            m_temp = round(actual_temp + np.random.normal(0, 1.1 * lt_err_factor * m_spread), 1)
            m_wind = max(1.0, round(actual_wind + np.random.normal(0, 3.0 * lt_err_factor * m_spread), 1))
            ens_members_rain.append(m_rain)
            ens_members_temp.append(m_temp)
            ens_members_wind.append(m_wind)
            
        ens_rain = round(float(np.mean(ens_members_rain)), 1)
        ens_temp = round(float(np.mean(ens_members_temp)), 1)
        ens_wind = round(float(np.mean(ens_members_wind)), 1)
        
        record = {
            "Date": date_str,
            "Time": time_str,
            "Region": region,
            "Latitude": meta["lat"],
            "Longitude": meta["lon"],
            "Season": season,
            "Lead_Time": lead_time,
            "Weather_Regime": regime,
            
            # Observations
            "Actual_Rainfall": actual_rain,
            "Actual_Temperature": actual_temp,
            "Actual_Wind": actual_wind,
            
            # NWP Model
            "NWP_Rainfall": nwp_rain,
            "NWP_Temperature": nwp_temp,
            "NWP_Wind": nwp_wind,
            
            # AI Model
            "AI_Rainfall": ai_rain,
            "AI_Temperature": ai_temp,
            "AI_Wind": ai_wind,
            
            # Ensemble Mean
            "Ensemble_Rainfall": ens_rain,
            "Ensemble_Temperature": ens_temp,
            "Ensemble_Wind": ens_wind,
            
            # Ensemble 5 Individual Members
            "Ens_M1_Rainfall": ens_members_rain[0],
            "Ens_M1_Temp": ens_members_temp[0],
            "Ens_M1_Wind": ens_members_wind[0],
            "Ens_M2_Rainfall": ens_members_rain[1],
            "Ens_M2_Temp": ens_members_temp[1],
            "Ens_M2_Wind": ens_members_wind[1],
            "Ens_M3_Rainfall": ens_members_rain[2],
            "Ens_M3_Temp": ens_members_temp[2],
            "Ens_M3_Wind": ens_members_wind[2],
            "Ens_M4_Rainfall": ens_members_rain[3],
            "Ens_M4_Temp": ens_members_temp[3],
            "Ens_M4_Wind": ens_members_wind[3],
            "Ens_M5_Rainfall": ens_members_rain[4],
            "Ens_M5_Temp": ens_members_temp[4],
            "Ens_M5_Wind": ens_members_wind[4],
            
            "Data_Source": "DEMO / SIMULATED DATA - Atmospheric Dynamics Prototype"
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    
    # Intentionally inject controlled missing values and duplicates to demonstrate cleaning pipeline
    # 1. Duplicates
    dup_indices = np.random.choice(len(df), size=25, replace=False)
    dup_rows = df.iloc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # 2. Missing values in a few rows
    missing_indices = np.random.choice(len(df), size=30, replace=False)
    for idx in missing_indices[:15]:
        df.loc[idx, "NWP_Rainfall"] = np.nan
    for idx in missing_indices[15:25]:
        df.loc[idx, "Actual_Temperature"] = np.nan
    for idx in missing_indices[25:]:
        df.loc[idx, "AI_Wind"] = np.nan
        
    return df

def main():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "demo_weather_raw.csv")
    
    print(f"Generating realistic simulated demo weather dataset...")
    df = generate_weather_data(n_samples=1600, random_seed=42)
    df.to_csv(out_path, index=False)
    print(f"Successfully generated {len(df)} records saved to: {out_path}")
    print(f"Regions: {df['Region'].nunique()}, Seasons: {df['Season'].unique().tolist()}")
    print(f"Regimes: {df['Weather_Regime'].unique().tolist()}, Lead Times: {df['Lead_Time'].unique().tolist()}")

if __name__ == "__main__":
    main()
