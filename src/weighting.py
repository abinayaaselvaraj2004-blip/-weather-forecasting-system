"""
Adaptive Weighting and Hybrid Blending Engine for AtmosBlend AI
Dynamically assigns weights to NWP, AI/ML, and Ensemble forecasts based on:
Region, Season, Forecast Lead Time, Weather Regime, and Historical Model Errors.
Implements: Reliability = 1 / (Error + epsilon), normalized to 100%.
Generates comprehensive meteorological rationales.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class AdaptiveWeightingEngine:
    def __init__(self, data_path: str = None, epsilon: float = 0.5):
        self.data_path = data_path or os.path.join("data", "processed", "weather_cleaned.csv")
        self.epsilon = epsilon
        self.df = None
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = None

    def calculate_historical_errors(
        self,
        region: str,
        season: str,
        lead_time: str,
        regime: str,
        variable: str = "Rainfall"
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates context-specific historical MAE & RMSE for NWP, AI, and Ensemble.
        Hierarchically falls back if specific slice is too small (< 5 samples).
        """
        if self.df is None:
            self._load_data()
            
        df = self.df
        var_suffix = variable  # e.g., 'Rainfall', 'Temperature', 'Wind'
        actual_col = f"Actual_{var_suffix}"
        nwp_col = f"NWP_{var_suffix}"
        ai_col = f"AI_{var_suffix}"
        ens_col = f"Ensemble_{var_suffix}"

        # Level 1: Full contextual intersection
        subset = df[
            (df["Region"] == region) &
            (df["Season"] == season) &
            (df["Lead_Time"] == lead_time) &
            (df["Weather_Regime"] == regime)
        ]
        
        # Level 2 fallback: Region + Regime + Lead Time
        if len(subset) < 5:
            subset = df[
                (df["Region"] == region) &
                (df["Weather_Regime"] == regime)
            ]
            
        # Level 3 fallback: Season + Regime
        if len(subset) < 5:
            subset = df[
                (df["Season"] == season) &
                (df["Weather_Regime"] == regime)
            ]
            
        # Level 4 fallback: Entire Regime
        if len(subset) < 5:
            subset = df[df["Weather_Regime"] == regime]
            
        # Level 5 fallback: Global
        if len(subset) < 5:
            subset = df

        models = {
            "NWP": nwp_col,
            "AI": ai_col,
            "Ensemble": ens_col
        }
        
        errors = {}
        for m_name, col in models.items():
            err_abs = (subset[actual_col] - subset[col]).abs()
            mae = float(err_abs.mean())
            rmse = float(np.sqrt((err_abs ** 2).mean()))
            errors[m_name] = {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "samples_count": len(subset)
            }
            
        return errors

    def compute_adaptive_weights(
        self,
        region: str,
        season: str,
        lead_time: str,
        regime: str,
        variable: str = "Rainfall"
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], str]:
        """
        Computes dynamic reliability weights:
        Reliability = 1 / (MAE + epsilon)
        Weights = Reliability / Sum(Reliability) * 100
        """
        errors = self.calculate_historical_errors(region, season, lead_time, regime, variable)
        
        reliabilities = {}
        for m_name, err_dict in errors.items():
            mae = err_dict["MAE"]
            rel = 1.0 / (mae + self.epsilon)
            reliabilities[m_name] = rel
            
        total_rel = sum(reliabilities.values())
        raw_weights = {m_name: (rel / total_rel) * 100.0 for m_name, rel in reliabilities.items()}
        
        # Round and normalize to ensure exact 100.0% sum
        rounded_weights = {m_name: round(val, 1) for m_name, val in raw_weights.items()}
        diff = round(100.0 - sum(rounded_weights.values()), 1)
        # Adjust largest weight slightly for exact 100% rounding
        dominant_model = max(rounded_weights, key=rounded_weights.get)
        rounded_weights[dominant_model] = round(rounded_weights[dominant_model] + diff, 1)
        
        rationale = self._generate_weight_rationale(
            region, season, lead_time, regime, variable, errors, rounded_weights, dominant_model
        )
        
        return rounded_weights, errors, rationale

    def _generate_weight_rationale(
        self,
        region: str,
        season: str,
        lead_time: str,
        regime: str,
        variable: str,
        errors: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        dominant_model: str
    ) -> str:
        """
        Generates meteorological justification for why these weights were assigned.
        """
        dom_mae = errors[dominant_model]["MAE"]
        second_model = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)[1]
        second_mae = errors[second_model]["MAE"]
        diff_pct = round(((second_mae - dom_mae) / (second_mae + 1e-5)) * 100, 1)
        
        unit = "mm" if variable == "Rainfall" else ("°C" if variable == "Temperature" else "km/h")
        
        # Meteorological insights
        context_notes = []
        if regime in ["Heavy Rain", "Storm/Cyclone"]:
            if dominant_model == "AI":
                context_notes.append("Machine learning captures high-frequency convective precipitation peaks with lower spatial displacement bias.")
            elif dominant_model == "Ensemble":
                context_notes.append("Ensemble perturbation members effectively capture atmospheric instability spread during extreme storm events.")
            else:
                context_notes.append("NWP hydrodynamic equations resolve large-scale synoptic moisture convergence effectively.")
        elif regime == "Heat Wave":
            if dominant_model == "NWP":
                context_notes.append("Physical radiative transfer schemes in NWP accurately forecast surface sensible heat fluxes and land-surface coupling.")
            else:
                context_notes.append(f"{dominant_model} forecast demonstrates superior calibration against extreme surface temperatures.")
        elif lead_time in ["48 hours", "72 hours"]:
            context_notes.append(f"At medium-range lead time ({lead_time}), synoptic boundary conditions give {dominant_model} an edge in stability.")
        else: # 6h or 12h
            context_notes.append(f"At short lead time ({lead_time}), rapid nowcasting data integration yields minimal bias.")

        note_str = " ".join(context_notes)
        
        text = (
            f"**{dominant_model} Model Dominates ({weights[dominant_model]}% weight)**\n\n"
            f"• **Historical Precision**: In {region} during {season} ({lead_time}, {regime} regime), "
            f"{dominant_model} achieved the lowest historical error for {variable} "
            f"(MAE: {dom_mae} {unit}), outperforming {second_model} (MAE: {second_mae} {unit}) by {abs(diff_pct):.1f}%.\n"
            f"• **Atmospheric Physics Rationale**: {note_str}\n"
            f"• **Formula Applied**: Reliability = 1 / (MAE + {self.epsilon}), normalized across NWP ({weights['NWP']}%), "
            f"AI ({weights['AI']}%), and Ensemble ({weights['Ensemble']}%)."
        )
        return text

    def compute_hybrid_forecast(
        self,
        nwp_val: float,
        ai_val: float,
        ens_val: float,
        weights: Dict[str, float],
        variable: str = "Rainfall"
    ) -> float:
        """
        Hybrid = (NWP * W_NWP/100) + (AI * W_AI/100) + (Ensemble * W_ENS/100)
        """
        hybrid_val = (
            (nwp_val * weights["NWP"]) +
            (ai_val * weights["AI"]) +
            (ens_val * weights["Ensemble"])
        ) / 100.0
        
        if variable == "Rainfall":
            return max(0.0, round(hybrid_val, 1))
        elif variable == "Wind":
            return max(1.0, round(hybrid_val, 1))
        else:
            return round(hybrid_val, 1)

def main():
    engine = AdaptiveWeightingEngine()
    print("Testing Adaptive Weighting Engine...")
    
    # Test Scenario 1: Tamil Nadu, Northeast Monsoon, 24 hours, Heavy Rain
    w1, e1, r1 = engine.compute_adaptive_weights("Tamil Nadu", "Northeast Monsoon", "24 hours", "Heavy Rain", "Rainfall")
    print("\n--- SCENARIO 1: Tamil Nadu | NE Monsoon | 24h | Heavy Rain ---")
    print("Weights:", w1)
    print("Errors:", e1)
    print("Rationale:\n", r1)
    
    # Test Scenario 2: Rajasthan, Summer, 48 hours, Heat Wave
    w2, e2, r2 = engine.compute_adaptive_weights("Rajasthan", "Summer", "48 hours", "Heat Wave", "Temperature")
    print("\n--- SCENARIO 2: Rajasthan | Summer | 48h | Heat Wave ---")
    print("Weights:", w2)
    print("Errors:", e2)

if __name__ == "__main__":
    main()
