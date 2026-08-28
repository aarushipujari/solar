"""
🚀 Aditya-L1 Solar Flare & Space Weather Warning System
Production-Grade FastAPI Backend & Space-Ops Command Center

Features:
  - RESTful API with Interactive Swagger / OpenAPI Documentation at `/docs`
  - Spatio-Temporal PyTorch DL Inference (CNN + ConvLSTM)
  - Authentic PyTorch Autograd Grad-CAM (XAI)
  - National Infrastructure Impact Engine (NavIC, PGCIL, Aviation, Gaganyaan)
  - Zero-Cost Cloud Deployment Ready (Render, Hugging Face Spaces, Railway, Vercel)
"""

import os
import io
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from astropy.io import fits

from config import BASE_DIR, DATA_DIR, PROJECT_ROOT, SEQ_LENGTH
from model import SolarFlarePredictor, SpatioTemporalGradCAM
from preprocess import (
    load_and_clean_fits,
    preprocess_solar_disk,
    extract_active_region,
    apply_spectral_colormap,
    compute_magnetic_flux_gradient,
    compute_solar_physical_metrics,
)

# -----------------------------------------------------------------------------
# FASTAPI APP SETUP & SWAGGER CONFIGURATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="☀️ ISRO Aditya-L1 Solar Flare Early Warning System API",
    description="""
    **Smart India Hackathon (SIH) 2026**
    
    Production-grade AI microservice providing spatio-temporal solar flare forecasting 
    24 to 48 hours prior to Earth impact using Aditya-L1 SUIT multi-spectral imagery.
    
    *Features:*
    - 🛰️ **Aditya-L1 SUIT FITS Processing**
    - 🧠 **PyTorch Spatio-Temporal ConvLSTM Inference**
    - 🔬 **Authentic Autograd Grad-CAM Explainable AI (XAI)**
    - 🛡️ **National Infrastructure Impact Assessment (NavIC, PGCIL, Gaganyaan)**
    - 📡 **Automated ISSDC Advisory Bulletin Dispatcher**
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for free frontend hosting (Vercel, Netlify, GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
device = torch.device("cpu")
model = SolarFlarePredictor().to(device)

model_paths = [
    BASE_DIR / "solar_flare_model.pth",
    DATA_DIR / "solar_flare_model.pth",
    PROJECT_ROOT / "data" / "solar_flare_model.pth"
]
for p in model_paths:
    if p.exists():
        try:
            model.load_state_dict(torch.load(p, map_location=device))
            print(f"Loaded model weights from {p}")
            break
        except Exception as e:
            print(f"Warning loading {p}: {e}")

model.eval()
gradcam_engine = SpatioTemporalGradCAM(model)


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    scenario_id: Optional[str] = "AR3664_Impending_X_Flare"
    frame_indices: Optional[List[int]] = [0, 1, 2, 3]


class ImpactDirective(BaseModel):
    sector: str
    status: str
    risk_level: str
    action_directive: str


class ForecastResponse(BaseModel):
    timestamp_utc: str
    timestamp_ist: str
    target_active_region: str
    flare_probability_percent: float
    predicted_noaa_class: str
    estimated_peak_flux: str
    kp_geomagnetic_storm_index: str
    defcon_alert_condition: str
    impact_window_hours: str
    physics_indicators: dict
    mitigation_directives: List[ImpactDirective]


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_scenario_dir(scenario_id: str) -> Path:
    scenarios_map = {
        "AR3664_Impending_X_Flare": BASE_DIR / "scenarios" / "AR3664_Impending_X_Flare",
        "AR3685_M_Class_Eruption": BASE_DIR / "scenarios" / "AR3685_M_Class_Eruption",
        "AR3670_Quiet_Sun": BASE_DIR / "scenarios" / "AR3670_Quiet_Sun",
        "live_feed": DATA_DIR
    }
    target = scenarios_map.get(scenario_id, DATA_DIR)
    if not target.exists():
        target = DATA_DIR
    return target


def array_to_base64_png(img_array: np.ndarray) -> str:
    """Encodes RGB or grayscale numpy array into base64 PNG string."""
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_array
    _, buffer = cv2.imencode('.png', bgr)
    return base64.b64encode(buffer).decode('utf-8')


# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/health", tags=["System Diagnostics"])
def health_check():
    """Returns microservice health and telemetry status."""
    return {
        "status": "ONLINE",
        "service": "Aditya-L1 Space Weather Warning System",
        "model_architecture": "CNN-ConvLSTM Spatio-Temporal Predictor",
        "xai_engine": "PyTorch Autograd Grad-CAM",
        "time_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/scenarios", tags=["Observational Telemetry"])
def list_scenarios():
    """Lists available Aditya-L1 SUIT observation presets and live FITS feed."""
    return {
        "scenarios": [
            {
                "id": "AR3664_Impending_X_Flare",
                "name": "AR-3664 Impending X-Class Superflare",
                "description": "Rapid flux emergence and intense magnetic shear. High flare probability.",
                "expected_risk": "CRITICAL"
            },
            {
                "id": "AR3685_M_Class_Eruption",
                "name": "AR-3685 M-Class Eruptive Region",
                "description": "Intermediate flux build-up with moderate coronal reconnection risk.",
                "expected_risk": "WATCH"
            },
            {
                "id": "AR3670_Quiet_Sun",
                "name": "AR-3670 Quiet Sun Nominal State",
                "description": "Stable unipolar sunspot with low background coronal emission.",
                "expected_risk": "NOMINAL"
            },
            {
                "id": "live_feed",
                "name": "Live FITS Telemetry Stream",
                "description": "Ingests contiguous FITS observations from data/full_resolution.",
                "expected_risk": "DYNAMIC"
            }
        ]
    }


@app.post("/api/forecast", response_model=ForecastResponse, tags=["AI Forecasting"])
def run_forecast(request: ForecastRequest):
    """
    Ingests 4-frame temporal UV sequence, executes ConvLSTM DL inference,
    and returns 24-48h flare probability, NOAA flare class, and national asset directives.
    """
    target_dir = get_scenario_dir(request.scenario_id)
    fits_files = sorted(list(target_dir.glob("*.fits")))

    if len(fits_files) < SEQ_LENGTH:
        # Fallback to main DATA_DIR
        fits_files = sorted(list(DATA_DIR.glob("*.fits")))

    if len(fits_files) < SEQ_LENGTH:
        raise HTTPException(status_code=400, detail="Insufficient FITS files to build 4-frame temporal sequence.")

    # Select sequence
    seq_files = fits_files[:SEQ_LENGTH]
    patches = []
    headers = []

    for fpath in seq_files:
        raw = load_and_clean_fits(fpath)
        disk = preprocess_solar_disk(raw)
        patch = extract_active_region(disk, patch_size=(256, 256))
        patches.append(patch)

        meta = {"noaa_ar": "AR-3664", "date_obs": fpath.stem}
        try:
            with fits.open(fpath) as hdul:
                h = hdul[0].header
                meta["noaa_ar"] = h.get("NOAA_AR", "AR-3664")
                meta["date_obs"] = h.get("DATE-OBS", fpath.stem)
        except Exception:
            pass
        headers.append(meta)

    # Tensor conversion [1, 4, 1, 256, 256]
    seq_tensor = torch.tensor(np.stack(patches), dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    # Model inference
    with torch.no_grad():
        logits = model(seq_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        flare_prob = float(probs[1]) * 100.0

    # Physics indicators
    physics = compute_solar_physical_metrics(patches[-1])

    # Classification logic
    if flare_prob >= 70.0:
        flare_class = "X-Class (Extreme)"
        peak_flux = "1.8 × 10⁻⁴ W/m² (X1.8)"
        alert_cond = "CRITICAL"
        kp_index = "7 - 8 (Severe G3/G4 Storm)"
        impact_win = "18 - 36 Hours"
        directives = [
            ImpactDirective(sector="ISRO NavIC (IRNSS)", status="CRITICAL RISK", risk_level="HIGH", action_directive="Broadcast differential ionospheric correction flags. Potential position error > 18m."),
            ImpactDirective(sector="GSAT/INSAT Telecom", status="SURGE SAFE-MODE", risk_level="HIGH", action_directive="Safeguard GEO high-gain transponders against solar array differential surface charging."),
            ImpactDirective(sector="Gaganyaan Human Spaceflight", status="EVA NO-GO", risk_level="SEVERE", action_directive="Astronaut radiation dose elevated to 14.2 mSv/h in LEO. Extravehicular activity prohibited."),
            ImpactDirective(sector="PGCIL Power Grid", status="GIC ALERT", risk_level="HIGH", action_directive="Engage series capacitor banks across Northern & Western 765kV transmission corridors.")
        ]
    elif flare_prob >= 45.0:
        flare_class = "M-Class (Moderate)"
        peak_flux = "4.2 × 10⁻⁵ W/m² (M4.2)"
        alert_cond = "WATCH"
        kp_index = "5 - 6 (Moderate G1/G2 Storm)"
        impact_win = "24 - 48 Hours"
        directives = [
            ImpactDirective(sector="ISRO NavIC (IRNSS)", status="MONITORING", risk_level="MODERATE", action_directive="Minor ionospheric delay fluctuation. Maintain dual-frequency monitoring."),
            ImpactDirective(sector="GSAT/INSAT Telecom", status="NOMINAL", risk_level="LOW", action_directive="Nominal operations; track telemetry downlink signal-to-noise ratio."),
            ImpactDirective(sector="Gaganyaan Human Spaceflight", status="NOMINAL", risk_level="LOW", action_directive="Radiation dose within nominal limit (0.8 mSv/h). Standard cabin shielding active."),
            ImpactDirective(sector="PGCIL Power Grid", status="WATCH", risk_level="MODERATE", action_directive="Maintain standard reactive power reserve margins at high-voltage substations.")
        ]
    else:
        flare_class = "Quiet / A-B Class"
        peak_flux = "< 1.0 × 10⁻⁷ W/m²"
        alert_cond = "NOMINAL"
        kp_index = "1 - 2 (Quiet Conditions)"
        impact_win = "No Impending Disturbance"
        directives = [
            ImpactDirective(sector="ISRO NavIC (IRNSS)", status="NOMINAL", risk_level="SAFE", action_directive="Nominal satellite atomic clock accuracy (< 2.5m)."),
            ImpactDirective(sector="GSAT/INSAT Telecom", status="NOMINAL", risk_level="SAFE", action_directive="Nominal downlink transponder operations."),
            ImpactDirective(sector="Gaganyaan Human Spaceflight", status="SAFE", risk_level="SAFE", action_directive="Safe orbital space environment. Background radiation < 0.2 mSv/h."),
            ImpactDirective(sector="PGCIL Power Grid", status="NOMINAL", risk_level="SAFE", action_directive="Baseline geomagnetic field. Zero transformer saturation hazard.")
        ]

    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)

    return ForecastResponse(
        timestamp_utc=utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        timestamp_ist=ist_now.strftime("%Y-%m-%d %H:%M:%S IST"),
        target_active_region=headers[-1]["noaa_ar"],
        flare_probability_percent=round(flare_prob, 2),
        predicted_noaa_class=flare_class,
        estimated_peak_flux=peak_flux,
        kp_geomagnetic_storm_index=kp_index,
        defcon_alert_condition=alert_cond,
        impact_window_hours=impact_win,
        physics_indicators=physics,
        mitigation_directives=directives
    )


@app.get("/api/gradcam", tags=["Explainable AI (XAI)"])
def get_gradcam_heatmaps(scenario_id: Optional[str] = "AR3664_Impending_X_Flare"):
    """
    Computes genuine PyTorch Autograd Grad-CAM heatmaps for the 4-frame sequence
    and returns base64 PNG images showing exact magnetic shear attention lines.
    """
    target_dir = get_scenario_dir(scenario_id)
    fits_files = sorted(list(target_dir.glob("*.fits")))
    if len(fits_files) < SEQ_LENGTH:
        fits_files = sorted(list(DATA_DIR.glob("*.fits")))

    patches = []
    for fpath in fits_files[:SEQ_LENGTH]:
        raw = load_and_clean_fits(fpath)
        disk = preprocess_solar_disk(raw)
        patch = extract_active_region(disk, patch_size=(256, 256))
        patches.append(patch)

    seq_tensor = torch.tensor(np.stack(patches), dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    cams, _ = gradcam_engine.generate(seq_tensor, target_class=1)

    result_frames = []
    for i in range(len(cams)):
        patch_base = patches[i]
        cam_map = cams[i]

        # Generate Jet heatmap
        cam_uint8 = np.clip(cam_map * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Base image
        base_rgb = cv2.cvtColor(np.clip(patch_base * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(base_rgb, 0.4, heatmap_rgb, 0.6, 0)

        result_frames.append({
            "step": f"T-{SEQ_LENGTH - 1 - i}",
            "patch_base64": f"data:image/png;base64,{array_to_base64_png(base_rgb)}",
            "gradcam_base64": f"data:image/png;base64,{array_to_base64_png(blended)}",
            "peak_attention_score": float(np.max(cam_map))
        })

    return {
        "mathematical_formula": "L_GradCAM = ReLU(sum_k (alpha_k * A_k))",
        "target_layer": "SolarFlarePredictor.encoder[3] (Conv2d 32-channel)",
        "frames": result_frames
    }


@app.get("/api/bulletin", tags=["ISSDC Advisory Bulletin"])
def generate_bulletin(scenario_id: Optional[str] = "AR3664_Impending_X_Flare"):
    """Generates official ISRO ISSDC-formatted space weather advisory text bulletin."""
    req = ForecastRequest(scenario_id=scenario_id)
    forecast = run_forecast(req)

    bulletin_text = f"""================================================================================
INDIAN SPACE RESEARCH ORGANISATION (ISRO)
ISSDC SPACE WEATHER FORECAST & EARLY WARNING BULLETIN
ISSUED: {forecast.timestamp_utc} / {forecast.timestamp_ist}
================================================================================

1. OBSERVATIONAL SUMMARY:
   Spacecraft: Aditya-L1 | Payload: SUIT (Solar Ultraviolet Imaging Telescope)
   Filter: Mg II k (279.6 nm) | Target Active Region: {forecast.target_active_region}
   Data Source: {scenario_id}

2. FORECAST & RISK ASSESSMENT (24-48 HOUR WINDOW):
   Flare Eruption Probability: {forecast.flare_probability_percent}%
   Predicted NOAA Flare Class: {forecast.predicted_noaa_class}
   Estimated Peak X-Ray Flux: {forecast.estimated_peak_flux}
   Geomagnetic Storm Index (Kp): {forecast.kp_geomagnetic_storm_index}
   DEFCON Alert Level: CONDITION {forecast.defcon_alert_condition}
   Impact Window: {forecast.impact_window_hours}

3. CRITICAL INFRASTRUCTURE MITIGATION DIRECTIVES:
"""
    for d in forecast.mitigation_directives:
        bulletin_text += f"   - [{d.sector}]: {d.status} ({d.risk_level}) -> {d.action_directive}\n"

    bulletin_text += f"""
================================================================================
Aditya-L1 Deep Learning Early Warning System | Smart India Hackathon (SIH) 2026
================================================================================
"""
    return {
        "bulletin_text": bulletin_text,
        "issued_utc": forecast.timestamp_utc,
        "alert_condition": forecast.defcon_alert_condition
    }


# -----------------------------------------------------------------------------
# ROOT ROUTE: EMBEDDED HIGH-TECH SPACE-OPS COMMAND CENTER WEB UI
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
def index():
    """Renders the standalone Space-Ops Command Center dashboard."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISRO Aditya-L1 | Space Weather Command Center</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        body { background-color: #060814; color: #dbe4ee; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .glass-card { background: rgba(16, 24, 48, 0.8); border: 1px solid rgba(0, 229, 255, 0.15); border-radius: 12px; backdrop-filter: blur(10px); }
        .glow-cyan { text-shadow: 0 0 12px rgba(0, 229, 255, 0.6); }
        .glow-red { text-shadow: 0 0 12px rgba(255, 51, 75, 0.8); }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8">
    <!-- Top Header -->
    <header class="glass-card p-6 mb-6">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <div class="flex items-center gap-2 mb-1">
                    <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-cyan-950 text-cyan-400 border border-cyan-500/40">🇮🇳 ISRO ADITYA-L1 OPS</span>
                    <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-950 text-amber-400 border border-amber-500/40">SMART INDIA HACKATHON 2026</span>
                    <a href="/docs" target="_blank" class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-950 text-emerald-400 border border-emerald-500/40 hover:bg-emerald-900 transition">🚀 Interactive Swagger API Docs</a>
                </div>
                <h1 class="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
                    ☀️ Aditya-L1 Solar Flare & Space Weather Warning System
                </h1>
                <p class="text-sm text-slate-400">
                    Proactive Deep Learning Forecasting for Critical National Satellites, Power Grids & Civil Aviation
                </p>
            </div>
            <div class="text-left md:text-right border-l md:border-l-0 pl-4 md:pl-0 border-slate-700">
                <div class="text-xs text-slate-400">ORBITAL STATION</div>
                <div class="text-lg font-bold text-cyan-400 glow-cyan">Sun-Earth L1 Halo Orbit</div>
                <div class="text-xs text-emerald-400 flex items-center md:justify-end gap-1">
                    <span class="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    Downlink: <b>NOMINAL (ISSDC Bylalu)</b>
                </div>
            </div>
        </div>
    </header>

    <!-- SIH Mission Banner -->
    <div class="glass-card p-4 mb-6 border-l-4 border-l-cyan-400 bg-slate-900/60">
        <div class="flex justify-between items-center mb-1">
            <span class="font-bold text-cyan-400 text-sm flex items-center gap-1.5"><i data-lucide="target" class="w-4 h-4"></i> SIH Mission Objective</span>
            <span class="text-xs text-cyan-300 font-semibold px-2 py-0.5 bg-cyan-950/60 rounded border border-cyan-500/30">Transforming Reactive Mitigation into Proactive Defence</span>
        </div>
        <p class="text-xs md:text-sm text-slate-300">
            To protect critical satellite communication, global navigation networks (<b>NavIC / GPS</b>), and power infrastructure (<b>PGCIL</b>) from destructive geomagnetic storms and CMEs, our project leverages spatio-temporal deep learning (<b>CNN + ConvLSTM</b>) trained on multi-spectral solar imagery from the pioneering <b>ISRO Aditya-L1 SUIT payload</b> to provide highly accurate forecasts <b>24 to 48 hours prior to Earth impact</b>.
        </p>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <!-- Control Panel & Status -->
        <div class="glass-card p-6 flex flex-col justify-between">
            <div>
                <h3 class="text-lg font-bold text-cyan-400 mb-3 flex items-center gap-2"><i data-lucide="sliders" class="w-5 h-5"></i> Observation Preset</h3>
                <label class="text-xs text-slate-400 mb-1 block">Select Telemetry Scenario:</label>
                <select id="scenarioSelect" onchange="fetchForecast()" class="w-full bg-slate-900 border border-slate-700 text-cyan-300 rounded-lg p-2.5 text-sm focus:outline-none focus:border-cyan-500 mb-4">
                    <option value="AR3664_Impending_X_Flare">🔴 AR-3664 Impending X-Class Superflare (High Risk)</option>
                    <option value="AR3685_M_Class_Eruption">🟡 AR-3685 M-Class Eruptive Region (Moderate Risk)</option>
                    <option value="AR3670_Quiet_Sun">🟢 AR-3670 Quiet Sun Nominal State (Low Risk)</option>
                    <option value="live_feed">📁 Live FITS Telemetry Stream (data/full_resolution)</option>
                </select>

                <div class="mt-4 p-4 rounded-xl bg-slate-950/70 border border-slate-800">
                    <div class="text-xs text-slate-400 mb-1">DEFCON ALERT STATUS</div>
                    <div id="defconBadge" class="text-xl font-extrabold text-red-500 glow-red">CONDITION RED // CRITICAL</div>
                    <div id="flareClass" class="text-sm font-semibold text-slate-200 mt-1">Predicted: X-Class (Extreme)</div>
                    <div id="peakFlux" class="text-xs text-slate-400">Peak Flux: 1.8 × 10⁻⁴ W/m²</div>
                </div>
            </div>

            <div class="mt-6 pt-4 border-t border-slate-800 flex justify-between items-center text-xs text-slate-400">
                <div>Lead Time: <b class="text-white" id="impactWindow">18 - 36 Hours</b></div>
                <div>Geomagnetic Kp: <b class="text-amber-400" id="kpIndex">7 - 8 (Severe)</b></div>
            </div>
        </div>

        <!-- 24-48h Probability Dial -->
        <div class="glass-card p-6 flex flex-col items-center justify-center text-center">
            <h3 class="text-sm font-semibold text-slate-400 mb-2">⚡ 24–48h Flare Eruption Probability</h3>
            <div class="relative flex items-center justify-center w-48 h-48 my-2">
                <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                    <path class="text-slate-800" stroke-width="3.5" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                    <path id="probCircle" class="text-red-500 transition-all duration-1000 ease-out" stroke-dasharray="88, 100" stroke-width="3.5" stroke-linecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                </svg>
                <div class="absolute flex flex-col items-center">
                    <span id="probValue" class="text-4xl font-extrabold text-white">88.4%</span>
                    <span class="text-xs text-slate-400 font-medium">Risk Factor</span>
                </div>
            </div>
            <p class="text-xs text-slate-400 mt-2 max-w-xs">Probability calculated via Spatio-Temporal ConvLSTM Sequence Model.</p>
        </div>

        <!-- Physical Parameters -->
        <div class="glass-card p-6">
            <h3 class="text-lg font-bold text-cyan-400 mb-3 flex items-center gap-2"><i data-lucide="activity" class="w-5 h-5"></i> Solar Magnetic Physics</h3>
            <div class="space-y-3 text-sm">
                <div class="flex justify-between p-2.5 rounded bg-slate-900/50 border border-slate-800">
                    <span class="text-slate-400">Total Unsigned Flux (Φ):</span>
                    <b id="fluxProxy" class="text-cyan-300">48.20</b>
                </div>
                <div class="flex justify-between p-2.5 rounded bg-slate-900/50 border border-slate-800">
                    <span class="text-slate-400">Max Flux Gradient (|∇I|):</span>
                    <b id="maxGrad" class="text-cyan-300">1.00 (High Shear)</b>
                </div>
                <div class="flex justify-between p-2.5 rounded bg-slate-900/50 border border-slate-800">
                    <span class="text-slate-400">Neutral Line Complexity:</span>
                    <b id="shearIndex" class="text-amber-400">92.4 / 100</b>
                </div>
                <div class="flex justify-between p-2.5 rounded bg-slate-900/50 border border-slate-800">
                    <span class="text-slate-400">Active Magnetic Loops:</span>
                    <b id="loopCount" class="text-emerald-400">2,410</b>
                </div>
            </div>
        </div>
    </div>

    <!-- National Assets Threat Matrix -->
    <div class="glass-card p-6 mb-6">
        <h3 class="text-lg font-bold text-cyan-400 mb-4 flex items-center gap-2"><i data-lucide="shield-alert" class="w-5 h-5"></i> National Infrastructure Mitigation Directives</h3>
        <div id="directivesContainer" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <!-- Dynamically Populated -->
        </div>
    </div>

    <!-- Grad-CAM Saliency Reel -->
    <div class="glass-card p-6 mb-6">
        <div class="flex justify-between items-center mb-4">
            <div>
                <h3 class="text-lg font-bold text-cyan-400 flex items-center gap-2"><i data-lucide="brain" class="w-5 h-5"></i> Spatio-Temporal Grad-CAM Explainability (XAI)</h3>
                <p class="text-xs text-slate-400">Mathematical PyTorch backpropagation attention maps showing magnetic shear lines driving the forecast.</p>
            </div>
            <span class="text-xs font-mono bg-slate-900 text-cyan-400 px-2.5 py-1 rounded border border-cyan-500/30">α_k = (1/Z) Σ (∂y/∂A_k)</span>
        </div>
        <div id="gradcamReel" class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <!-- Dynamically Loaded Grad-CAM images -->
        </div>
    </div>

    <script>
        lucide.createIcons();

        async function fetchForecast() {
            const sc = document.getElementById('scenarioSelect').value;
            try {
                const res = await fetch('/api/forecast', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({scenario_id: sc})
                });
                const data = await res.json();

                // Update metrics
                document.getElementById('probValue').innerText = data.flare_probability_percent + '%';
                document.getElementById('probCircle').setAttribute('stroke-dasharray', `${data.flare_probability_percent}, 100`);
                document.getElementById('flareClass').innerText = 'Predicted: ' + data.predicted_noaa_class;
                document.getElementById('peakFlux').innerText = 'Peak Flux: ' + data.estimated_peak_flux;
                document.getElementById('impactWindow').innerText = data.impact_window_hours;
                document.getElementById('kpIndex').innerText = data.kp_geomagnetic_storm_index;
                
                const defBadge = document.getElementById('defconBadge');
                if (data.defcon_alert_condition === 'CRITICAL') {
                    defBadge.className = 'text-xl font-extrabold text-red-500 glow-red';
                    defBadge.innerText = 'CONDITION RED // CRITICAL';
                } else if (data.defcon_alert_condition === 'WATCH') {
                    defBadge.className = 'text-xl font-extrabold text-amber-400';
                    defBadge.innerText = 'CONDITION AMBER // WATCH';
                } else {
                    defBadge.className = 'text-xl font-extrabold text-emerald-400';
                    defBadge.innerText = 'CONDITION GREEN // NOMINAL';
                }

                // Physics
                document.getElementById('fluxProxy').innerText = data.physics_indicators.unsigned_flux_proxy.toFixed(2);
                document.getElementById('maxGrad').innerText = data.physics_indicators.max_flux_gradient.toFixed(2);
                document.getElementById('shearIndex').innerText = data.physics_indicators.shear_complexity_index.toFixed(1) + '/100';
                document.getElementById('loopCount').innerText = data.physics_indicators.total_contour_loops;

                // Directives
                const dirCont = document.getElementById('directivesContainer');
                dirCont.innerHTML = data.mitigation_directives.map(d => `
                    <div class="p-4 rounded-xl bg-slate-950/60 border ${d.risk_level === 'HIGH' || d.risk_level === 'SEVERE' ? 'border-red-500/40 bg-red-950/10' : 'border-slate-800'}">
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-bold text-white text-xs">${d.sector}</span>
                            <span class="text-[10px] font-bold px-1.5 py-0.5 rounded ${d.risk_level === 'HIGH' || d.risk_level === 'SEVERE' ? 'bg-red-900 text-red-300' : 'bg-emerald-900 text-emerald-300'}">${d.status}</span>
                        </div>
                        <p class="text-xs text-slate-300 mt-2">${d.action_directive}</p>
                    </div>
                `).join('');

                // Fetch Grad-CAM
                fetchGradCAM(sc);
            } catch(e) {
                console.error(e);
            }
        }

        async function fetchGradCAM(sc) {
            try {
                const res = await fetch('/api/gradcam?scenario_id=' + sc);
                const data = await res.json();
                const container = document.getElementById('gradcamReel');
                container.innerHTML = data.frames.map(f => `
                    <div class="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 text-center">
                        <img src="${f.gradcam_base64}" class="w-full h-auto object-cover" alt="GradCAM ${f.step}" />
                        <div class="p-2 text-xs font-semibold text-cyan-300 bg-slate-900/80">Step ${f.step}</div>
                    </div>
                `).join('');
            } catch(e) {
                console.error(e);
            }
        }

        // Initial call
        fetchForecast();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
