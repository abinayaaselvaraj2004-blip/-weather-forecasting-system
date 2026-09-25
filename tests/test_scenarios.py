"""
Comprehensive Scenario & Pipeline Verification Script for AtmosBlend AI
Tests all 4 required scenarios, dynamic adaptive weighting transitions,
extreme weather alert triggers, and Power BI output formats.
"""

import os
import sys
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.weighting import AdaptiveWeightingEngine
from src.extreme_weather import ExtremeWeatherDetector
from src.ensemble import EnsembleForecastEngine
from src.evaluation import ForecastEvaluator

def run_tests():
    print("=" * 70)
    print("[RUNNING] ATMOSBLEND AI - AUTOMATED SCENARIO & SYSTEM VERIFICATION")
    print("=" * 70)

    weight_engine = AdaptiveWeightingEngine()
    evaluator = ForecastEvaluator()

    # -------------------------------------------------------------
    # TEST 1: SCENARIO 1 (Tamil Nadu, NE Monsoon, 24h, Heavy Rain)
    # -------------------------------------------------------------
    print("\n[TEST 1] Scenario 1: Tamil Nadu | NE Monsoon | 24 Hours | Heavy Rain")
    w1, e1, r1 = weight_engine.compute_adaptive_weights("Tamil Nadu", "Northeast Monsoon", "24 hours", "Heavy Rain", "Rainfall")
    dom1 = max(w1, key=w1.get)
    hyb_rain1 = weight_engine.compute_hybrid_forecast(75.0, 85.0, 82.0, w1, "Rainfall")
    risk1 = ExtremeWeatherDetector.evaluate_risks(hyb_rain1, 27.0, 35.0, "Tamil Nadu", "24 hours", "Heavy Rain")
    
    print(f"  * Weights: NWP={w1['NWP']}%, AI={w1['AI']}%, Ensemble={w1['Ensemble']}%")
    print(f"  * Dominant Model: {dom1} ({w1[dom1]}%)")
    print(f"  * Hybrid Rainfall Forecast: {hyb_rain1:.1f} mm")
    print(f"  * Risk Level: {risk1['overall_risk']} (Threat Score: {risk1['risk_score']}/100)")
    print(f"  * Active Alert: {risk1['alerts'][0]['title']}")
    assert abs(sum(w1.values()) - 100.0) < 0.1, "Weights must sum to 100%"
    assert risk1["overall_risk"] in ["Moderate", "High", "Severe"], "Should trigger heavy rain alert"
    print("  [PASS] Scenario 1 PASSED")

    # -------------------------------------------------------------
    # TEST 2: SCENARIO 2 (Rajasthan, Summer, 48h, Heat Wave)
    # -------------------------------------------------------------
    print("\n[TEST 2] Scenario 2: Rajasthan | Summer | 48 Hours | Heat Wave")
    w2, e2, r2 = weight_engine.compute_adaptive_weights("Rajasthan", "Summer", "48 hours", "Heat Wave", "Temperature")
    dom2 = max(w2, key=w2.get)
    hyb_temp2 = weight_engine.compute_hybrid_forecast(44.5, 43.8, 44.2, w2, "Temperature")
    risk2 = ExtremeWeatherDetector.evaluate_risks(0.0, hyb_temp2, 18.0, "Rajasthan", "48 hours", "Heat Wave")

    print(f"  * Weights: NWP={w2['NWP']}%, AI={w2['AI']}%, Ensemble={w2['Ensemble']}%")
    print(f"  * Dominant Model: {dom2} ({w2[dom2]}%)")
    print(f"  * Hybrid Temperature Forecast: {hyb_temp2:.1f} C")
    print(f"  * Risk Level: {risk2['overall_risk']} (Threat Score: {risk2['risk_score']}/100)")
    print(f"  * Active Alert: {risk2['alerts'][0]['title']}")
    assert abs(sum(w2.values()) - 100.0) < 0.1, "Weights must sum to 100%"
    assert hyb_temp2 >= 43.0, "Should reflect extreme temperature"
    assert "HEAT" in risk2["alerts"][0]["title"], "Should trigger heat wave alert"
    print("  [PASS] Scenario 2 PASSED")

    # Dynamic Adaptiveness Verification
    assert (w1 != w2), "Weights must adapt and change when switching scenarios!"
    print("  [PASS] Dynamic Adaptive Weight Shift Verified (w1 != w2)")

    # -------------------------------------------------------------
    # TEST 3: SCENARIO 3 (Kerala, SW Monsoon, 24h, Heavy Rain)
    # -------------------------------------------------------------
    print("\n[TEST 3] Scenario 3: Kerala | Southwest Monsoon | 24 Hours | Heavy Rain")
    w3, e3, _ = weight_engine.compute_adaptive_weights("Kerala", "Southwest Monsoon", "24 hours", "Heavy Rain", "Rainfall")
    hyb_rain3 = weight_engine.compute_hybrid_forecast(110.0, 125.0, 118.0, w3, "Rainfall")
    risk3 = ExtremeWeatherDetector.evaluate_risks(hyb_rain3, 26.5, 40.0, "Kerala", "24 hours", "Heavy Rain")
    print(f"  * Weights: NWP={w3['NWP']}%, AI={w3['AI']}%, Ensemble={w3['Ensemble']}%")
    print(f"  * Hybrid Rainfall: {hyb_rain3:.1f} mm")
    print(f"  * Active Alert: {risk3['alerts'][0]['title']}")
    assert risk3["overall_risk"] in ["High", "Severe"], "Should trigger high/severe rain alert"
    print("  [PASS] Scenario 3 PASSED")

    # -------------------------------------------------------------
    # TEST 4: SCENARIO 4 (Odisha, Post-Monsoon, 24h, Storm/Cyclone)
    # -------------------------------------------------------------
    print("\n[TEST 4] Scenario 4: Coastal Odisha | Post-Monsoon | 24 Hours | Storm/Cyclone")
    w4, e4, _ = weight_engine.compute_adaptive_weights("Odisha", "Post-Monsoon", "24 hours", "Storm/Cyclone", "Wind")
    hyb_wind4 = weight_engine.compute_hybrid_forecast(78.0, 72.0, 81.0, w4, "Wind")
    hyb_rain4 = 95.0
    risk4 = ExtremeWeatherDetector.evaluate_risks(hyb_rain4, 25.0, hyb_wind4, "Odisha", "24 hours", "Storm/Cyclone")
    print(f"  * Weights (Wind): NWP={w4['NWP']}%, AI={w4['AI']}%, Ensemble={w4['Ensemble']}%")
    print(f"  * Hybrid Wind Forecast: {hyb_wind4:.1f} km/h")
    print(f"  * Active Alert: {risk4['alerts'][0]['title']}")
    assert "CYCLON" in risk4["alerts"][0]["title"] or "WIND" in risk4["alerts"][0]["title"], "Should trigger storm/cyclone signal"
    print("  [PASS] Scenario 4 PASSED")

    # -------------------------------------------------------------
    # TEST 5: 5-Member Ensemble Spread Verification
    # -------------------------------------------------------------
    print("\n[TEST 5] Ensemble 5-Member Perturbation Engine")
    ens_data = EnsembleForecastEngine.generate_or_retrieve_members({
        "NWP_Rainfall": 80.0, "NWP_Temperature": 30.0, "NWP_Wind": 45.0
    })
    rain_members = ens_data["Rainfall"]["members"]
    assert len(rain_members) == 5, "Must generate exactly 5 ensemble members"
    print(f"  * 5 Members: {list(rain_members.values())}")
    print(f"  * Ensemble Mean: {ens_data['Rainfall']['mean']} mm, Spread: +/-{ens_data['Rainfall']['spread']} mm")
    print("  [PASS] 5-Member Ensemble Engine PASSED")

    # -------------------------------------------------------------
    # TEST 6: Power BI File Export Verification
    # -------------------------------------------------------------
    print("\n[TEST 6] Power BI CSV / Excel Export Schema Verification")
    csv_path = os.path.join("outputs", "powerbi_export.csv")
    xlsx_path = os.path.join("outputs", "powerbi_export.xlsx")
    assert os.path.exists(csv_path), f"Missing {csv_path}"
    assert os.path.exists(xlsx_path), f"Missing {xlsx_path}"
    
    df_pbi = pd.read_csv(csv_path)
    expected_cols = [
        "Date", "Region", "Season", "Lead_Time", "Weather_Regime",
        "Actual_Rainfall", "NWP_Rainfall", "AI_Rainfall", "Ensemble_Rainfall", "Hybrid_Rainfall",
        "NWP_Weight", "AI_Weight", "Ensemble_Weight",
        "NWP_MAE", "AI_MAE", "Ensemble_MAE", "Hybrid_MAE", "Risk_Level"
    ]
    for col in expected_cols:
        assert col in df_pbi.columns, f"Missing required column: {col}"
    print(f"  * Exported Rows: {len(df_pbi)}")
    print(f"  * Columns Verified: {len(df_pbi.columns)} (Includes all {len(expected_cols)} mandatory columns)")
    print("  [PASS] Power BI Export PASSED")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL 6 TEST SUITES PASSED FLAWLESSLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
