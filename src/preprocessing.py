"""
Data Preprocessing Module for AtmosBlend AI
Handles missing values, duplicate removal, physical bounds validation,
datetime conversions, meteorological season/regime verification,
and feature engineering for ML model consumption.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class DataPreprocessor:
    def __init__(self, raw_filepath: str = None, processed_filepath: str = None):
        self.raw_filepath = raw_filepath or os.path.join("data", "raw", "demo_weather_raw.csv")
        self.processed_filepath = processed_filepath or os.path.join("data", "processed", "weather_cleaned.csv")
        self.cleaning_stats: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_filepath):
            raise FileNotFoundError(f"Raw data file not found at: {self.raw_filepath}")
        return pd.read_csv(self.raw_filepath)

    def clean_and_preprocess(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df is None:
            df = self.load_data()
            
        initial_count = len(df)
        initial_missing = int(df.isnull().sum().sum())
        missing_by_col_initial = df.isnull().sum().to_dict()
        
        # 1. Remove duplicate records
        duplicates_count = int(df.duplicated(subset=["Date", "Time", "Region", "Lead_Time"]).sum())
        df = df.drop_duplicates(subset=["Date", "Time", "Region", "Lead_Time"]).copy()
        
        # 2. Date and Time parsing
        df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
        df = df.dropna(subset=["DateTime"]).copy()
        
        # Cyclical temporal feature engineering
        df["Month"] = df["DateTime"].dt.month
        df["Day"] = df["DateTime"].dt.day
        df["Hour"] = df["DateTime"].dt.hour
        df["DayOfWeek"] = df["DateTime"].dt.dayofweek
        
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
        df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
        df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
        
        # Lead time numeric conversion
        lead_time_map = {
            "6 hours": 6,
            "12 hours": 12,
            "24 hours": 24,
            "48 hours": 48,
            "72 hours": 72
        }
        df["Lead_Time_Hours"] = df["Lead_Time"].map(lead_time_map).fillna(24).astype(int)

        # 3. Handle missing values
        # For numerical columns, impute with regional-seasonal median
        num_cols = [
            "Actual_Rainfall", "Actual_Temperature", "Actual_Wind",
            "NWP_Rainfall", "NWP_Temperature", "NWP_Wind",
            "AI_Rainfall", "AI_Temperature", "AI_Wind",
            "Ensemble_Rainfall", "Ensemble_Temperature", "Ensemble_Wind"
        ]
        
        for col in num_cols:
            if col in df.columns:
                # Group median imputation by Region and Season
                df[col] = df.groupby(["Region", "Season"])[col].transform(lambda x: x.fillna(x.median()))
                # Fallback to global median if still NaN
                df[col] = df[col].fillna(df[col].median())
                
        # Handle ensemble members missing if any
        ens_member_cols = [c for c in df.columns if c.startswith("Ens_M")]
        for col in ens_member_cols:
            df[col] = df.groupby(["Region", "Season"])[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

        # 4. Handle incorrect / out-of-physical-bounds values
        # Rainfall >= 0 mm, Temp between 0 and 55 C, Wind >= 0 km/h
        rainfall_cols = [c for c in df.columns if "Rain" in c]
        temp_cols = [c for c in df.columns if "Temp" in c]
        wind_cols = [c for c in df.columns if "Wind" in c]
        
        for c in rainfall_cols:
            df[c] = df[c].clip(lower=0.0, upper=500.0)
        for c in temp_cols:
            df[c] = df[c].clip(lower=2.0, upper=54.0)
        for c in wind_cols:
            df[c] = df[c].clip(lower=0.0, upper=250.0)
            
        final_count = len(df)
        final_missing = int(df.isnull().sum().sum())
        
        self.cleaning_stats = {
            "initial_records": initial_count,
            "final_records": final_count,
            "duplicates_removed": duplicates_count,
            "initial_missing_values": initial_missing,
            "final_missing_values": final_missing,
            "missing_by_col_initial": missing_by_col_initial,
            "regions_count": int(df["Region"].nunique()),
            "seasons": df["Season"].unique().tolist(),
            "regimes": df["Weather_Regime"].unique().tolist(),
            "lead_times": df["Lead_Time"].unique().tolist()
        }
        
        return df, self.cleaning_stats

    def save_processed(self, df: pd.DataFrame) -> str:
        os.makedirs(os.path.dirname(self.processed_filepath), exist_ok=True)
        df.to_csv(self.processed_filepath, index=False)
        return self.processed_filepath

def main():
    preprocessor = DataPreprocessor()
    print("Executing Data Preprocessing Pipeline...")
    cleaned_df, stats = preprocessor.clean_and_preprocess()
    saved_path = preprocessor.save_processed(cleaned_df)
    
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Initial Records: {stats['initial_records']}")
    print(f"Duplicates Removed: {stats['duplicates_removed']}")
    print(f"Missing Values Imputed: {stats['initial_missing_values']} -> {stats['final_missing_values']}")
    print(f"Cleaned Records Saved: {stats['final_records']} to {saved_path}")
    print("Data Preprocessing successfully completed.")

if __name__ == "__main__":
    main()
