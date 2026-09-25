"""
Ensemble Forecast Module for AtmosBlend AI
Simulates and manages 5-member ensemble forecasts representing initial condition uncertainties.
Computes Ensemble Mean, Spread (standard deviation), and Uncertainty metrics.
Clearly labeled as: "Ensemble Forecast – Prototype"
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class EnsembleForecastEngine:
    """
    Manages 5-member ensemble forecasting with physical perturbation schemes:
    Member 1: Control Run (deterministic baseline)
    Member 2: Positive convective/moisture perturbation
    Member 3: Negative convective/moisture perturbation
    Member 4: Baroclinic / kinetic wind shear perturbation
    Member 5: Synoptic thermal boundary perturbation
    """
    
    LABEL = "Ensemble Forecast – Prototype"

    @staticmethod
    def compute_ensemble_stats(members: List[float]) -> Dict[str, float]:
        arr = np.array(members, dtype=float)
        mean_val = float(np.mean(arr))
        spread_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return {
            "mean": round(mean_val, 1),
            "spread": round(spread_val, 2),
            "min": round(float(np.min(arr)), 1),
            "max": round(float(np.max(arr)), 1)
        }

    @classmethod
    def generate_or_retrieve_members(cls, row_or_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves existing 5 members from record or synthesizes realistic members.
        """
        has_rain_members = all(f"Ens_M{i}_Rainfall" in row_or_dict for i in range(1, 6))
        
        if has_rain_members:
            rain_members = [float(row_or_dict[f"Ens_M{i}_Rainfall"]) for i in range(1, 6)]
            temp_members = [float(row_or_dict[f"Ens_M{i}_Temp"]) for i in range(1, 6)]
            wind_members = [float(row_or_dict[f"Ens_M{i}_Wind"]) for i in range(1, 6)]
        else:
            # Generate realistic perturbed members around base value
            base_rain = float(row_or_dict.get("Ensemble_Rainfall", row_or_dict.get("NWP_Rainfall", 10.0)))
            base_temp = float(row_or_dict.get("Ensemble_Temperature", row_or_dict.get("NWP_Temperature", 30.0)))
            base_wind = float(row_or_dict.get("Ensemble_Wind", row_or_dict.get("NWP_Wind", 15.0)))
            
            # Perturbations
            rain_factors = [1.0, 1.15, 0.88, 1.05, 0.95]
            temp_offsets = [0.0, 0.6, -0.7, 0.3, -0.4]
            wind_offsets = [0.0, 3.2, -2.8, 4.5, -1.8]
            
            rain_members = [max(0.0, round(base_rain * f + np.random.normal(0, 1.2), 1)) for f in rain_factors]
            temp_members = [round(base_temp + t + np.random.normal(0, 0.3), 1) for t in temp_offsets]
            wind_members = [max(1.0, round(base_wind + w + np.random.normal(0, 0.8), 1)) for w in wind_offsets]

        rain_stats = cls.compute_ensemble_stats(rain_members)
        temp_stats = cls.compute_ensemble_stats(temp_members)
        wind_stats = cls.compute_ensemble_stats(wind_members)
        
        return {
            "source_label": cls.LABEL,
            "Rainfall": {
                "mean": rain_stats["mean"],
                "spread": rain_stats["spread"],
                "members": {f"Member {i+1}": rain_members[i] for i in range(5)}
            },
            "Temperature": {
                "mean": temp_stats["mean"],
                "spread": temp_stats["spread"],
                "members": {f"Member {i+1}": temp_members[i] for i in range(5)}
            },
            "Wind": {
                "mean": wind_stats["mean"],
                "spread": wind_stats["spread"],
                "members": {f"Member {i+1}": wind_members[i] for i in range(5)}
            }
        }
