"""
AtmosBlend AI - 2D Anime Weather Visual Assets & Theme Engine
Provides SVG vector illustrations, CSS animations, and anime weather assistant components.
"""

def get_anime_character_svg(regime: str) -> str:
    """
    Returns a crisp, lightweight, high-quality 2D Anime Weather Analyst SVG illustration.
    Dynamically adjusts hair, face expression, umbrella posture, and headset glow
    based on the synoptic weather regime.
    """
    # Expression and accent variants
    if regime == "Heat Wave":
        mouth_path = "M 54 82 Q 60 88 66 82"  # Slightly exhausted / flushed
        eyes_svg = """
            <!-- Squinting / warm eyes -->
            <ellipse cx="48" cy="65" rx="5.5" ry="4" fill="#38bdf8" />
            <ellipse cx="72" cy="65" rx="5.5" ry="4" fill="#38bdf8" />
            <circle cx="46" cy="63" r="1.5" fill="#ffffff" />
            <circle cx="70" cy="63" r="1.5" fill="#ffffff" />
            <!-- Sweat drop -->
            <path d="M 80 56 Q 83 62 80 66 Q 77 62 80 56 Z" fill="#38bdf8" opacity="0.9" />
            <!-- Cheek blush -->
            <ellipse cx="42" cy="74" rx="4" ry="2" fill="#fb7185" opacity="0.6" />
            <ellipse cx="78" cy="74" rx="4" ry="2" fill="#fb7185" opacity="0.6" />
        """
        umbrella_color = "#f59e0b"
        headset_glow = "#f59e0b"
    elif regime == "High Wind":
        mouth_path = "M 55 83 Q 60 81 65 83"  # Resolute, braced
        eyes_svg = """
            <!-- Determined wide anime eyes -->
            <ellipse cx="48" cy="64" rx="6" ry="6.5" fill="#0284c7" />
            <ellipse cx="72" cy="64" rx="6" ry="6.5" fill="#0284c7" />
            <circle cx="46" cy="62" r="2.5" fill="#ffffff" />
            <circle cx="70" cy="62" r="2.5" fill="#ffffff" />
            <circle cx="50" cy="66" r="1.2" fill="#38bdf8" />
            <circle cx="74" cy="66" r="1.2" fill="#38bdf8" />
            <!-- Wind streak -->
            <path d="M 18 45 Q 35 42 50 48" stroke="#38bdf8" stroke-width="1.8" stroke-linecap="round" fill="none" opacity="0.75" />
            <path d="M 22 55 Q 38 52 48 57" stroke="#38bdf8" stroke-width="1.5" stroke-linecap="round" fill="none" opacity="0.6" />
        """
        umbrella_color = "#06b6d4"
        headset_glow = "#00f5d4"
    elif regime == "Storm/Cyclone":
        mouth_path = "M 55 84 L 65 84"  # Serious / alert line
        eyes_svg = """
            <!-- Focused alert eyes with cyber glow visor -->
            <ellipse cx="48" cy="64" rx="6" ry="6" fill="#818cf8" />
            <ellipse cx="72" cy="64" rx="6" ry="6" fill="#818cf8" />
            <circle cx="47" cy="62" r="2.2" fill="#ffffff" />
            <circle cx="71" cy="62" r="2.2" fill="#ffffff" />
            <!-- Visor scanning line -->
            <line x1="38" y1="64" x2="82" y2="64" stroke="#c084fc" stroke-width="1.5" stroke-dasharray="2,2" opacity="0.85" />
            <!-- Cheek blush -->
            <ellipse cx="42" cy="74" rx="3.5" ry="1.8" fill="#a855f7" opacity="0.4" />
            <ellipse cx="78" cy="74" rx="3.5" ry="1.8" fill="#a855f7" opacity="0.4" />
        """
        umbrella_color = "#8b5cf6"
        headset_glow = "#c084fc"
    elif regime == "Heavy Rain":
        mouth_path = "M 56 81 Q 60 86 64 81"  # Mild concerned smile
        eyes_svg = """
            <!-- Large expressive rainy anime eyes -->
            <ellipse cx="48" cy="64" rx="6" ry="7" fill="#0284c7" />
            <ellipse cx="72" cy="64" rx="6" ry="7" fill="#0284c7" />
            <circle cx="46" cy="61" r="2.5" fill="#ffffff" />
            <circle cx="70" cy="61" r="2.5" fill="#ffffff" />
            <circle cx="50" cy="67" r="1.5" fill="#38bdf8" />
            <circle cx="74" cy="67" r="1.5" fill="#38bdf8" />
            <!-- Water drop on cheek -->
            <path d="M 40 70 Q 42 74 40 77 Q 38 74 40 70 Z" fill="#38bdf8" opacity="0.85" />
            <ellipse cx="78" cy="74" rx="4" ry="2" fill="#f43f5e" opacity="0.4" />
        """
        umbrella_color = "#38bdf8"
        headset_glow = "#38bdf8"
    else:  # Normal
        mouth_path = "M 55 80 Q 60 87 65 80"  # Cheerful confident anime smile
        eyes_svg = """
            <!-- Sparkling anime eyes -->
            <ellipse cx="48" cy="63" rx="6" ry="7" fill="#0284c7" />
            <ellipse cx="72" cy="63" rx="6" ry="7" fill="#0284c7" />
            <circle cx="46" cy="60" r="2.8" fill="#ffffff" />
            <circle cx="70" cy="60" r="2.8" fill="#ffffff" />
            <circle cx="50" cy="67" r="1.4" fill="#38bdf8" />
            <circle cx="74" cy="67" r="1.4" fill="#38bdf8" />
            <!-- Rosy anime blush -->
            <ellipse cx="42" cy="73" rx="4.5" ry="2.2" fill="#fb7185" opacity="0.5" />
            <ellipse cx="78" cy="73" rx="4.5" ry="2.2" fill="#fb7185" opacity="0.5" />
        """
        umbrella_color = "#38bdf8"
        headset_glow = "#38bdf8"

    svg_content = f"""<svg viewBox="0 0 160 170" width="130" height="138" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Umbrella Canopy Gradient -->
    <linearGradient id="umbrellaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{umbrella_color}" stop-opacity="0.95" />
      <stop offset="100%" stop-color="#0f172a" stop-opacity="0.9" />
    </linearGradient>
    <!-- Raincoat Gradient -->
    <linearGradient id="coatGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.85" />
      <stop offset="100%" stop-color="#0284c7" stop-opacity="0.95" />
    </linearGradient>
    <!-- Hair Gradient -->
    <linearGradient id="hairGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e293b" />
      <stop offset="50%" stop-color="#0369a1" />
      <stop offset="100%" stop-color="#0284c7" />
    </linearGradient>
    <!-- Glow Filter -->
    <filter id="neonGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Umbrella (Behind Head) -->
  <g id="umbrella">
    <!-- Umbrella Stick -->
    <line x1="88" y1="20" x2="116" y2="140" stroke="#94a3b8" stroke-width="3" stroke-linecap="round" />
    <!-- Umbrella Rib Arc -->
    <path d="M 30 40 Q 88 -5 146 36 Q 120 32 95 38 Q 65 32 30 40 Z" fill="url(#umbrellaGrad)" stroke="{umbrella_color}" stroke-width="1.8" filter="url(#neonGlow)" />
    <!-- Rain drops bouncing on umbrella -->
    <circle cx="65" cy="18" r="2" fill="#e0f2fe" opacity="0.85" />
    <circle cx="108" cy="22" r="1.8" fill="#e0f2fe" opacity="0.85" />
    <path d="M 64 12 Q 66 16 64 19 Q 62 16 64 12 Z" fill="#bae6fd" opacity="0.75" />
  </g>

  <!-- Character Back Hair -->
  <path d="M 34 50 Q 22 90 32 125 Q 60 115 88 125 Q 98 90 86 50 Z" fill="url(#hairGrad)" />

  <!-- Body & Raincoat -->
  <g id="body">
    <!-- Raincoat Shoulders & Torso -->
    <path d="M 34 116 Q 60 106 86 116 L 98 165 L 22 165 Z" fill="url(#coatGrad)" />
    <!-- Raincoat Collar & Hood Fold -->
    <path d="M 38 116 Q 60 128 82 116 Q 60 110 38 116 Z" fill="#0c4a6e" />
    <!-- Coat Zipper & Neon Status Stripe -->
    <line x1="60" y1="120" x2="60" y2="165" stroke="{headset_glow}" stroke-width="2" stroke-dasharray="3,2" />
  </g>

  <!-- Neck -->
  <rect x="52" y="96" width="16" height="18" rx="4" fill="#fed7aa" />

  <!-- Head & Face -->
  <g id="face">
    <path d="M 36 60 Q 36 100 60 102 Q 84 100 84 60 Q 84 38 60 38 Q 36 38 36 60 Z" fill="#ffedd5" />
    
    <!-- Eyes, Blush & Expression -->
    {eyes_svg}

    <!-- Anime Nose -->
    <path d="M 60 72 L 59 75" stroke="#f97316" stroke-width="1.2" stroke-linecap="round" />

    <!-- Anime Mouth -->
    <path d="{mouth_path}" stroke="#e11d48" stroke-width="2" stroke-linecap="round" fill="none" />
  </g>

  <!-- Front Hair & Bangs -->
  <g id="frontHair">
    <path d="M 34 50 Q 42 70 46 72 Q 48 56 54 44 Q 60 68 64 68 Q 66 52 74 44 Q 78 68 84 52 Q 88 38 60 35 Q 36 38 34 50 Z" fill="url(#hairGrad)" />
    <!-- Cute hair highlight line -->
    <path d="M 44 45 Q 60 40 76 45" stroke="#7dd3fc" stroke-width="2" stroke-linecap="round" fill="none" opacity="0.8" />
  </g>

  <!-- Weather Monitoring Headset -->
  <g id="headset">
    <!-- Headband -->
    <path d="M 36 60 Q 60 32 84 60" stroke="#334155" stroke-width="4" fill="none" />
    <!-- Left Earphone -->
    <rect x="32" y="55" width="8" height="15" rx="3" fill="#0f172a" stroke="{headset_glow}" stroke-width="1.5" />
    <!-- Right Earphone with glowing sensor -->
    <rect x="80" y="55" width="8" height="15" rx="3" fill="#0f172a" stroke="{headset_glow}" stroke-width="1.5" />
    <circle cx="84" cy="62" r="2.5" fill="{headset_glow}" filter="url(#neonGlow)" />
    <!-- Microphone Arm -->
    <path d="M 82 66 Q 76 80 67 82" stroke="#64748b" stroke-width="2" fill="none" stroke-linecap="round" />
    <circle cx="66" cy="82" r="2" fill="{headset_glow}" />
  </g>

  <!-- Hand Holding Umbrella -->
  <ellipse cx="104" cy="132" rx="6" ry="6" fill="#fed7aa" />
</svg>"""
    return svg_content


def get_anime_speech(regime: str, dominant_model: str, hybrid_val: float, unit: str, var_name: str) -> dict:
    """
    Returns dynamically generated speech commentary and anime mood for the weather analyst.
    """
    if regime == "Heavy Rain":
        return {
            "mood": "Concerned & Alert",
            "icon": "🌧️",
            "dialogue": f"Heavy rainfall conditions detected ({hybrid_val:.1f} {unit})! The hybrid engine is dynamically prioritizing **{dominant_model}** due to its superior historical accuracy in mitigating convective over-prediction.",
            "subtext": "Atmospheric humidity saturation > 88% • Localized cloudburst risk monitored • Automated alerts dispatched."
        }
    elif regime == "Heat Wave":
        return {
            "mood": "Thermal Advisory Active",
            "icon": "🥵",
            "dialogue": f"Severe heat wave conditions detected ({hybrid_val:.1f} {unit})! **{dominant_model}** is holding primary reliability weight to capture persistent boundary-layer heat retention.",
            "subtext": "Surface thermal anomalies elevated • IMD Heat Hazard thresholds active • Hydration advisories recommended."
        }
    elif regime == "High Wind":
        return {
            "mood": "Wind Shear Warning",
            "icon": "💨",
            "dialogue": f"Strong gale-force wind conditions detected ({hybrid_val:.1f} {unit})! Adaptive weighting has recalibrated to mitigate physical boundary drag biases.",
            "subtext": "Wind gusts exceeding synoptic thresholds • Ensemble dispersion monitored • Coastal structural advisories active."
        }
    elif regime == "Storm/Cyclone":
        return {
            "mood": "Severe Storm Alert",
            "icon": "⛈️",
            "dialogue": f"Critical storm & cyclonic circulation detected! High compound hazard score. **{dominant_model}** is leading the multi-source fusion blend.",
            "subtext": "Compound rainfall + gale shear threshold exceeded • Emergency mitigation protocols aligned with MoES/IMD."
        }
    else:
        return {
            "mood": "Normal Operations",
            "icon": "🌤️",
            "dialogue": f"All atmospheric indicators are currently normal ({hybrid_val:.1f} {unit}). The multi-model blend is balanced across NWP physics, AI regressors, and ensemble spreads.",
            "subtext": "Model variance remains low • Forecast confidence is high • Continuous real-time telemetry streaming."
        }
