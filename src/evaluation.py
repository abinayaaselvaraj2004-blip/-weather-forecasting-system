"""
Evaluation and Power BI Export Module for AtmosBlend AI
Computes overall and slice-based MAE, RMSE, and Skill Scores for:
NWP vs AI/ML vs Ensemble vs Hybrid Forecast.
Generates Power BI-compatible CSV and Excel export datasets.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weighting import AdaptiveWeightingEngine
from src.extreme_weather import ExtremeWeatherDetector

class ForecastEvaluator:
    def __init__(self, data_path: str = None, outputs_dir: str = None):
        self.data_path = data_path or os.path.join("data", "processed", "weather_cleaned.csv")
        self.outputs_dir = outputs_dir or "outputs"
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.weight_engine = AdaptiveWeightingEngine(self.data_path)
        self.df = None
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = None

    def evaluate_overall_models(self) -> Dict[str, Any]:
        """
        Computes comparative MAE, RMSE, and Hybrid improvement across all records.
        """
        if self.df is None:
            self._load_data()
            
        df = self.df.copy()
        
        # Precompute hybrid forecasts across the dataset
        hybrid_rain = []
        hybrid_temp = []
        hybrid_wind = []
        
        # Cache weights by (Region, Season, Lead_Time, Weather_Regime)
        weight_cache = {}
        
        for _, row in df.iterrows():
            key = (row["Region"], row["Season"], row["Lead_Time"], row["Weather_Regime"])
            if key not in weight_cache:
                w_rain, _, _ = self.weight_engine.compute_adaptive_weights(
                    row["Region"], row["Season"], row["Lead_Time"], row["Weather_Regime"], "Rainfall"
                )
                w_temp, _, _ = self.weight_engine.compute_adaptive_weights(
                    row["Region"], row["Season"], row["Lead_Time"], row["Weather_Regime"], "Temperature"
                )
                w_wind, _, _ = self.weight_engine.compute_adaptive_weights(
                    row["Region"], row["Season"], row["Lead_Time"], row["Weather_Regime"], "Wind"
                )
                weight_cache[key] = (w_rain, w_temp, w_wind)
                
            w_r, w_t, w_w = weight_cache[key]
            
            hr = self.weight_engine.compute_hybrid_forecast(
                row["NWP_Rainfall"], row["AI_Rainfall"], row["Ensemble_Rainfall"], w_r, "Rainfall"
            )
            ht = self.weight_engine.compute_hybrid_forecast(
                row["NWP_Temperature"], row["AI_Temperature"], row["Ensemble_Temperature"], w_t, "Temperature"
            )
            hw = self.weight_engine.compute_hybrid_forecast(
                row["NWP_Wind"], row["AI_Wind"], row["Ensemble_Wind"], w_w, "Wind"
            )
            
            hybrid_rain.append(hr)
            hybrid_temp.append(ht)
            hybrid_wind.append(hw)
            
        df["Hybrid_Rainfall"] = hybrid_rain
        df["Hybrid_Temperature"] = hybrid_temp
        df["Hybrid_Wind"] = hybrid_wind
        
        metrics = {}
        for var in ["Rainfall", "Temperature", "Wind"]:
            actual = df[f"Actual_{var}"]
            nwp = df[f"NWP_{var}"]
            ai = df[f"AI_{var}"]
            ens = df[f"Ensemble_{var}"]
            hyb = df[f"Hybrid_{var}"]
            
            nwp_mae = float(np.mean(np.abs(actual - nwp)))
            nwp_rmse = float(np.sqrt(np.mean((actual - nwp) ** 2)))
            
            ai_mae = float(np.mean(np.abs(actual - ai)))
            ai_rmse = float(np.sqrt(np.mean((actual - ai) ** 2)))
            
            ens_mae = float(np.mean(np.abs(actual - ens)))
            ens_rmse = float(np.sqrt(np.mean((actual - ens) ** 2)))
            
            hyb_mae = float(np.mean(np.abs(actual - hyb)))
            hyb_rmse = float(np.sqrt(np.mean((actual - hyb) ** 2)))
            
            best_individual_rmse = min(nwp_rmse, ai_rmse, ens_rmse)
            best_individual_mae = min(nwp_mae, ai_mae, ens_mae)
            
            # Skill score improvement over best constituent model
            rmse_improvement_pct = round(((best_individual_rmse - hyb_rmse) / best_individual_rmse) * 100, 2)
            mae_improvement_pct = round(((best_individual_mae - hyb_mae) / best_individual_mae) * 100, 2)
            
            metrics[var] = {
                "NWP": {"MAE": round(nwp_mae, 2), "RMSE": round(nwp_rmse, 2)},
                "AI": {"MAE": round(ai_mae, 2), "RMSE": round(ai_rmse, 2)},
                "Ensemble": {"MAE": round(ens_mae, 2), "RMSE": round(ens_rmse, 2)},
                "Hybrid": {"MAE": round(hyb_mae, 2), "RMSE": round(hyb_rmse, 2)},
                "best_individual_model": "Ensemble" if best_individual_mae == ens_mae else ("AI" if best_individual_mae == ai_mae else "NWP"),
                "mae_improvement_pct": mae_improvement_pct,
                "rmse_improvement_pct": rmse_improvement_pct
            }
            
        return metrics

    def generate_powerbi_dataset(self) -> Tuple[str, str, pd.DataFrame]:
        """
        Creates full Power BI-ready CSV and Excel export containing:
        Date, Region, Season, Lead_Time, Weather_Regime,
        Actual_Rainfall, NWP_Rainfall, AI_Rainfall, Ensemble_Rainfall, Hybrid_Rainfall,
        NWP_Weight, AI_Weight, Ensemble_Weight,
        NWP_MAE, AI_MAE, Ensemble_MAE, Hybrid_MAE, Risk_Level
        """
        if self.df is None:
            self._load_data()
            
        df = self.df.copy()
        
        weight_cache = {}
        records = []
        
        for _, row in df.iterrows():
            key = (row["Region"], row["Season"], row["Lead_Time"], row["Weather_Regime"])
            if key not in weight_cache:
                w_rain, err_rain, _ = self.weight_engine.compute_adaptive_weights(
                    row["Region"], row["Season"], row["Lead_Time"], row["Weather_Regime"], "Rainfall"
                )
                weight_cache[key] = (w_rain, err_rain)
                
            w_rain, err_rain = weight_cache[key]
            
            hyb_rain = self.weight_engine.compute_hybrid_forecast(
                row["NWP_Rainfall"], row["AI_Rainfall"], row["Ensemble_Rainfall"], w_rain, "Rainfall"
            )
            hyb_temp = row.get("Actual_Temperature", 30.0) # placeholder if not available
            hyb_wind = row.get("NWP_Wind", 15.0)
            
            # Risk evaluation
            risk_info = ExtremeWeatherDetector.evaluate_risks(
                hybrid_rain=hyb_rain,
                hybrid_temp=float(row.get("NWP_Temperature", 30.0)),
                hybrid_wind=float(row.get("NWP_Wind", 15.0)),
                region=row["Region"],
                lead_time=row["Lead_Time"],
                weather_regime=row["Weather_Regime"]
            )
            
            # Hybrid MAE estimation
            hyb_mae = round(abs(row["Actual_Rainfall"] - hyb_rain), 2)
            
            rec = {
                "Date": row["Date"],
                "Time": row.get("Time", "12:00"),
                "Region": row["Region"],
                "Season": row["Season"],
                "Lead_Time": row["Lead_Time"],
                "Weather_Regime": row["Weather_Regime"],
                "Actual_Rainfall": row["Actual_Rainfall"],
                "NWP_Rainfall": row["NWP_Rainfall"],
                "AI_Rainfall": row["AI_Rainfall"],
                "Ensemble_Rainfall": row["Ensemble_Rainfall"],
                "Hybrid_Rainfall": hyb_rain,
                "NWP_Weight": w_rain["NWP"],
                "AI_Weight": w_rain["AI"],
                "Ensemble_Weight": w_rain["Ensemble"],
                "NWP_MAE": err_rain["NWP"]["MAE"],
                "AI_MAE": err_rain["AI"]["MAE"],
                "Ensemble_MAE": err_rain["Ensemble"]["MAE"],
                "Hybrid_MAE": hyb_mae,
                "Risk_Level": risk_info["overall_risk"],
                "Risk_Score": risk_info["risk_score"]
            }
            records.append(rec)
            
        pbi_df = pd.DataFrame(records)
        
        csv_path = os.path.join(self.outputs_dir, "powerbi_export.csv")
        xlsx_path = os.path.join(self.outputs_dir, "powerbi_export.xlsx")
        
        pbi_df.to_csv(csv_path, index=False)
        pbi_df.to_excel(xlsx_path, index=False, engine="openpyxl")
        
        return csv_path, xlsx_path, pbi_df

def main():
    evaluator = ForecastEvaluator()
    print("Evaluating Multi-Model Performance vs Hybrid Forecast...")
    metrics = evaluator.evaluate_overall_models()
    
    print("\n=== MODEL COMPARISON (RAINFALL) ===")
    for model, m in metrics["Rainfall"].items():
        if isinstance(m, dict):
            print(f"Model: {model:<10} | MAE: {m['MAE']:<6} | RMSE: {m['RMSE']:<6}")
            
    print(f"Skill Score Improvement of Hybrid over Best Model: {metrics['Rainfall']['mae_improvement_pct']}%")
    
    print("\nGenerating Power BI Export Files...")
    csv_p, xlsx_p, pbi_df = evaluator.generate_powerbi_dataset()
    print(f"Exported {len(pbi_df)} rows to:")
    print(f"1. CSV: {csv_p}")
    print(f"2. Excel: {xlsx_p}")

if __name__ == "__main__":
    main()
