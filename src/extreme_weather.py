"""
Extreme Weather Detection Module for AtmosBlend AI
Detects Heavy Rain, Heat Wave, High Wind, and Cyclone Signals using standardized meteorological thresholds.
Generates Risk Levels, Probability/Risk Scores, and Alert Bulletins.
Clearly displays prototype thresholds calibrated for MoES / NCMRWF operational context.
"""

from typing import Dict, Any, List

class ExtremeWeatherDetector:
    """
    Standard Prototype Alert Thresholds (Calibrated from IMD/MoES Standards):
    - Heavy Rain:
        * Moderate: 15.6 - 64.4 mm
        * Heavy (Yellow Alert): 64.5 - 115.5 mm
        * Very Heavy (Orange Alert): 115.6 - 204.4 mm
        * Extremely Heavy (Red Alert): > 204.4 mm
    - Heat Wave:
        * Heat Alert: 40.0 - 43.0 °C
        * Heat Wave (Orange Alert): 43.1 - 45.0 °C
        * Severe Heat Wave (Red Alert): > 45.0 °C
    - High Wind:
        * Strong Wind (Yellow Alert): 45.0 - 61.0 km/h
        * Gale Force (Orange Alert): 62.0 - 88.0 km/h
        * Storm Force (Red Alert): > 88.0 km/h
    - Cyclone / Storm Compound Signal:
        * Co-occurrence of Gale Wind (>62 km/h) + Heavy Rain (>60 mm) or Severe Regime
    """

    @classmethod
    def evaluate_risks(
        cls,
        hybrid_rain: float,
        hybrid_temp: float,
        hybrid_wind: float,
        region: str,
        lead_time: str,
        weather_regime: str = "Normal"
    ) -> Dict[str, Any]:
        alerts: List[Dict[str, Any]] = []
        risk_scores = []

        # 1. Rainfall Risk Analysis
        if hybrid_rain > 204.4:
            r_risk = "Severe"
            r_score = 95
            r_alert = {
                "type": "Heavy Rain",
                "title": "EXTREMELY HEAVY RAIN ALERT (RED ALERT)",
                "risk": r_risk,
                "score": r_score,
                "color": "#FF2B2B",
                "message": (
                    f"Forecast rainfall of {hybrid_rain:.1f} mm exceeds emergency threshold (>204.4 mm). "
                    f"Imminent danger of flash flooding, severe urban inundation, and landslides."
                )
            }
            alerts.append(r_alert)
            risk_scores.append(r_score)
        elif hybrid_rain >= 115.6:
            r_risk = "High"
            r_score = 80
            r_alert = {
                "type": "Heavy Rain",
                "title": "VERY HEAVY RAINFALL WARNING (ORANGE ALERT)",
                "risk": r_risk,
                "score": r_score,
                "color": "#FF8C00",
                "message": (
                    f"Forecast rainfall of {hybrid_rain:.1f} mm in {region} over {lead_time}. "
                    f"High likelihood of localized waterlogging, crop inundation, and road traffic disruption."
                )
            }
            alerts.append(r_alert)
            risk_scores.append(r_score)
        elif hybrid_rain >= 64.5:
            r_risk = "Moderate"
            r_score = 65
            r_alert = {
                "type": "Heavy Rain",
                "title": "HEAVY RAIN ADVISORY (YELLOW ALERT)",
                "risk": r_risk,
                "score": r_score,
                "color": "#E6B800",
                "message": (
                    f"Forecast rainfall of {hybrid_rain:.1f} mm in {region}. "
                    f"Residents and civic authorities advised to monitor drainage infrastructure."
                )
            }
            alerts.append(r_alert)
            risk_scores.append(r_score)
        else:
            r_risk = "Low"
            r_score = max(5, int(min(60, (hybrid_rain / 64.5) * 50)))
            risk_scores.append(r_score)

        # 2. Temperature / Heat Wave Risk Analysis
        if hybrid_temp >= 45.0:
            t_risk = "Severe"
            t_score = 92
            t_alert = {
                "type": "Heat Wave",
                "title": "SEVERE HEAT WAVE WARNING (RED ALERT)",
                "risk": t_risk,
                "score": t_score,
                "color": "#FF2B2B",
                "message": (
                    f"Critical temperature forecast of {hybrid_temp:.1f} °C in {region}. "
                    f"Extreme risk of heat stroke, dehydration, and power grid peak load stress."
                )
            }
            alerts.append(t_alert)
            risk_scores.append(t_score)
        elif hybrid_temp >= 43.0:
            t_risk = "High"
            t_score = 78
            t_alert = {
                "type": "Heat Wave",
                "title": "HEAT WAVE ALERT (ORANGE ALERT)",
                "risk": t_risk,
                "score": t_score,
                "color": "#FF8C00",
                "message": (
                    f"Extreme heat forecast of {hybrid_temp:.1f} °C. "
                    f"High thermal stress condition; avoid prolonged midday sun exposure."
                )
            }
            alerts.append(t_alert)
            risk_scores.append(t_score)
        elif hybrid_temp >= 40.0:
            t_risk = "Moderate"
            t_score = 55
            t_alert = {
                "type": "Heat Wave",
                "title": "WARM WEATHER ADVISORY (YELLOW ALERT)",
                "risk": t_risk,
                "score": t_score,
                "color": "#E6B800",
                "message": f"Elevated temperature ({hybrid_temp:.1f} °C) approaching heat wave thresholds."
            }
            alerts.append(t_alert)
            risk_scores.append(t_score)
        else:
            t_risk = "Low"
            t_score = 10
            risk_scores.append(t_score)

        # 3. Wind Speed Analysis
        if hybrid_wind >= 88.0:
            w_risk = "Severe"
            w_score = 95
            w_alert = {
                "type": "High Wind",
                "title": "STORM FORCE WIND WARNING (RED ALERT)",
                "risk": w_risk,
                "score": w_score,
                "color": "#FF2B2B",
                "message": (
                    f"Dangerous gusting winds reaching {hybrid_wind:.1f} km/h. "
                    f"Structural damage, uprooted trees, and transmission line failures anticipated."
                )
            }
            alerts.append(w_alert)
            risk_scores.append(w_score)
        elif hybrid_wind >= 62.0:
            w_risk = "High"
            w_score = 80
            w_alert = {
                "type": "High Wind",
                "title": "GALE FORCE WIND ALERT (ORANGE ALERT)",
                "risk": w_risk,
                "score": w_score,
                "color": "#FF8C00",
                "message": (
                    f"Gale force wind speed of {hybrid_wind:.1f} km/h forecast. "
                    f"Suspension of fishing operations and maritime cautionary signals advised."
                )
            }
            alerts.append(w_alert)
            risk_scores.append(w_score)
        elif hybrid_wind >= 45.0:
            w_risk = "Moderate"
            w_score = 60
            w_alert = {
                "type": "High Wind",
                "title": "SQUALLY WIND ADVISORY (YELLOW ALERT)",
                "risk": w_risk,
                "score": w_score,
                "color": "#E6B800",
                "message": f"Gusty winds up to {hybrid_wind:.1f} km/h. Drive cautiously and secure loose objects."
            }
            alerts.append(w_alert)
            risk_scores.append(w_score)
        else:
            w_risk = "Low"
            w_score = 12
            risk_scores.append(w_score)

        # 4. Storm / Cyclone Compound Signal
        cyclone_trigger = (hybrid_wind >= 62.0 and hybrid_rain >= 60.0) or (weather_regime == "Storm/Cyclone" and hybrid_wind >= 50.0)
        if cyclone_trigger:
            c_score = 98 if (hybrid_wind >= 85.0 or hybrid_rain >= 115.0) else 85
            c_alert = {
                "type": "Storm/Cyclone",
                "title": "CYCLONIC STORM / DEPRESSION THREAT (SPECIAL BULLETIN)",
                "risk": "Severe" if c_score >= 90 else "High",
                "score": c_score,
                "color": "#8B0000" if c_score >= 90 else "#FF4500",
                "message": (
                    f"Compound severe criteria met in {region} at lead time {lead_time}: "
                    f"Coincident rainfall of {hybrid_rain:.1f} mm and gale winds of {hybrid_wind:.1f} km/h. "
                    f"Triggering NCMRWF-IMD Coastal Emergency Protocol."
                )
            }
            alerts.insert(0, c_alert)
            risk_scores.append(c_score)

        # Composite overall risk calculation
        overall_score = max(risk_scores)
        if overall_score >= 88:
            overall_risk = "Severe"
            risk_color = "#FF2B2B"
        elif overall_score >= 70:
            overall_risk = "High"
            risk_color = "#FF8C00"
        elif overall_score >= 50:
            overall_risk = "Moderate"
            risk_color = "#E6B800"
        else:
            overall_risk = "Low"
            risk_color = "#28A745"

        if not alerts:
            alerts.append({
                "type": "Normal",
                "title": "NORMAL METEOROLOGICAL CONDITIONS (GREEN ALERT)",
                "risk": "Low",
                "score": overall_score,
                "color": "#28A745",
                "message": f"All forecast parameters within safe baseline ranges for {region} ({lead_time})."
            })

        return {
            "overall_risk": overall_risk,
            "risk_score": overall_score,
            "risk_color": risk_color,
            "region": region,
            "lead_time": lead_time,
            "hybrid_rain": hybrid_rain,
            "hybrid_temp": hybrid_temp,
            "hybrid_wind": hybrid_wind,
            "alerts": alerts,
            "thresholds_disclaimer": "Thresholds: MoES / IMD standard operational categories (Demo Calibrated Prototype)"
        }
