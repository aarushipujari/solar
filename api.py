"""
🚀 Aditya-L1 Solar Flare & Space Weather Warning System
Production-Grade FastAPI Backend & Space-Ops Command Center

Features:
  - 4-Channel Spatio-Temporal Multi-Task Model (CNN + ConvLSTM)
  - Learned Multi-Class NOAA Flare Classification & Learned Log Peak Flux Regression
  - Authentic PyTorch Autograd Grad-CAM (XAI)
  - Decision Support & National Infrastructure Threat Engine (NavIC, PGCIL, Aviation, Gaganyaan)
  - Standard Space Weather Verification Metrics
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
    build_multi_channel_frame,
    apply_spectral_colormap,
    compute_magnetic_flux_gradient,
    compute_optical_flux_and_shear_proxies,
)

# -----------------------------------------------------------------------------
# FASTAPI APP SETUP & SWAGGER CONFIGURATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="☀️ ISRO Aditya-L1 Solar Flare Early Warning System API",
    description="""
    **Smart India Hackathon (SIH) 2026**
    
    Production-grade AI microservice providing spatio-temporal solar flare forecasting 
    24 to 48 hours prior to Earth impact using 4-channel multi-spectral solar representations.
    """,
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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
model = SolarFlarePredictor(in_channels=4, hidden_dim=32).to(device)

model_paths = [
    BASE_DIR / "solar_flare_model.pth",
    DATA_DIR / "solar_flare_model.pth",
    PROJECT_ROOT / "data" / "solar_flare_model.pth"
]
for p in model_paths:
    if p.exists():
        try:
            model.load_state_dict(torch.load(p, map_location=device))
            print(f"Loaded 4-channel model weights from {p}")
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
    multiclass_distribution: dict
    estimated_peak_flux: str
    kp_geomagnetic_storm_index: str
    defcon_alert_condition: str
    impact_window_hours: str
    optical_proxies: dict
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
    return {
        "status": "ONLINE",
        "service": "Aditya-L1 Space Weather Warning System",
        "model_architecture": "4-Channel Multi-Task ConvLSTM",
        "xai_engine": "PyTorch Autograd Grad-CAM",
        "time_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/scenarios", tags=["Observational Telemetry"])
def list_scenarios():
    return {
        "scenarios": [
            {"id": "AR3664_Impending_X_Flare", "name": "AR-13664 Impending X-Class Superflare", "expected_risk": "CRITICAL"},
            {"id": "AR3685_M_Class_Eruption", "name": "AR-11158 M-Class Eruptive Region", "expected_risk": "WATCH"},
            {"id": "AR3670_Quiet_Sun", "name": "AR-13100 Quiet Sun Nominal State", "expected_risk": "NOMINAL"},
            {"id": "live_feed", "name": "Live FITS Historical Stream", "expected_risk": "DYNAMIC"}
        ]
    }


@app.post("/api/forecast", response_model=ForecastResponse, tags=["AI Forecasting"])
def run_forecast(request: ForecastRequest):
    target_dir = get_scenario_dir(request.scenario_id)
    fits_files = sorted(list(target_dir.glob("*.fits")))

    if len(fits_files) < SEQ_LENGTH:
        fits_files = sorted(list(DATA_DIR.glob("*.fits")))

    if len(fits_files) < SEQ_LENGTH:
        raise HTTPException(status_code=400, detail="Insufficient FITS files to build 4-frame temporal sequence.")

    seq_files = fits_files[:SEQ_LENGTH]
    patches = []
    mch_frames = []
    headers = []
    prev_patch = None

    for fpath in seq_files:
        raw = load_and_clean_fits(fpath)
        disk = preprocess_solar_disk(raw)
        patch = extract_active_region(disk, patch_size=(256, 256))
        patches.append(patch)

        mch = build_multi_channel_frame(patch, prev_patch=prev_patch)
        prev_patch = patch
        mch_frames.append(torch.tensor(mch, dtype=torch.float32))

        meta = {"noaa_ar": "AR-13664", "date_obs": fpath.stem}
        try:
            with fits.open(fpath) as hdul:
                h = hdul[0].header
                meta["noaa_ar"] = h.get("NOAA_AR", "AR-13664")
                meta["date_obs"] = h.get("DATE-OBS", fpath.stem)
        except Exception:
            pass
        headers.append(meta)

    # Tensor [1, 4, 4, 256, 256]
    seq_tensor = torch.stack(mch_frames, dim=0).unsqueeze(0)

    # Multi-task inference
    with torch.no_grad():
        preds = model(seq_tensor, return_all_heads=True)
        bin_probs = torch.softmax(preds["binary_logits"], dim=1).numpy()[0]
        flare_prob = float(bin_probs[1]) * 100.0

        multi_probs = torch.softmax(preds["multiclass_logits"], dim=1).numpy()[0]
        pred_idx = int(np.argmax(multi_probs))
        labels = ["Quiet / B-Class", "C-Class (Minor)", "M-Class (Moderate)", "X-Class (Extreme)"]
        flare_class = labels[pred_idx]

        log_flux = float(preds["log_flux_pred"].numpy()[0])
        peak_flux = f"{10.0 ** log_flux:.2e} W/m²"

    physics = compute_optical_flux_and_shear_proxies(patches[-1])

    if flare_prob >= 50.0:
        alert_cond = "CRITICAL" if pred_idx >= 2 else "WATCH"
        kp_index = "7 - 8 (Severe G3/G4) [Empirical]" if pred_idx == 3 else "5 - 6 (Moderate G1/G2) [Empirical]"
        impact_win = "18 - 36 Hours"
        directives = [
            ImpactDirective(sector="ISRO NavIC (IRNSS)", status="CRITICAL RISK", risk_level="HIGH", action_directive="Broadcast differential ionospheric correction flags. Potential position error elevated."),
            ImpactDirective(sector="GSAT/INSAT Telecom", status="SURGE SAFE-MODE", risk_level="HIGH", action_directive="Safeguard GEO high-gain transponders against solar array surface charging."),
            ImpactDirective(sector="Gaganyaan Human Spaceflight", status="EVA NO-GO", risk_level="SEVERE", action_directive="Astronaut radiation dose elevated in LEO. Extravehicular activity prohibited."),
            ImpactDirective(sector="PGCIL Power Grid", status="GIC ALERT", risk_level="HIGH", action_directive="Engage series capacitor banks across Northern & Western 765kV transmission corridors.")
        ]
    else:
        alert_cond = "NOMINAL"
        kp_index = "1 - 2 (Quiet Space Weather) [Empirical]"
        impact_win = "No Impending Disturbance"
        directives = [
            ImpactDirective(sector="ISRO NavIC (IRNSS)", status="NOMINAL", risk_level="SAFE", action_directive="Nominal satellite atomic clock accuracy (< 2.5m)."),
            ImpactDirective(sector="GSAT/INSAT Telecom", status="NOMINAL", risk_level="SAFE", action_directive="Nominal downlink transponder operations."),
            ImpactDirective(sector="Gaganyaan Human Spaceflight", status="SAFE", risk_level="SAFE", action_directive="Safe orbital space environment. Background radiation nominal."),
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
        multiclass_distribution={
            "Quiet_B": round(float(multi_probs[0]) * 100, 2),
            "C_Class": round(float(multi_probs[1]) * 100, 2),
            "M_Class": round(float(multi_probs[2]) * 100, 2),
            "X_Class": round(float(multi_probs[3]) * 100, 2),
        },
        estimated_peak_flux=peak_flux,
        kp_geomagnetic_storm_index=kp_index,
        defcon_alert_condition=alert_cond,
        impact_window_hours=impact_win,
        optical_proxies=physics,
        mitigation_directives=directives
    )


@app.get("/api/gradcam", tags=["Explainable AI (XAI)"])
def get_gradcam_heatmaps(scenario_id: Optional[str] = "AR3664_Impending_X_Flare"):
    target_dir = get_scenario_dir(scenario_id)
    fits_files = sorted(list(target_dir.glob("*.fits")))
    if len(fits_files) < SEQ_LENGTH:
        fits_files = sorted(list(DATA_DIR.glob("*.fits")))

    patches = []
    mch_frames = []
    prev_patch = None
    for fpath in fits_files[:SEQ_LENGTH]:
        raw = load_and_clean_fits(fpath)
        disk = preprocess_solar_disk(raw)
        patch = extract_active_region(disk, patch_size=(256, 256))
        patches.append(patch)

        mch = build_multi_channel_frame(patch, prev_patch=prev_patch)
        prev_patch = patch
        mch_frames.append(torch.tensor(mch, dtype=torch.float32))

    seq_tensor = torch.stack(mch_frames, dim=0).unsqueeze(0)
    cams, preds = gradcam_engine.generate(seq_tensor, target_class=1, task="binary")

    result_frames = []
    for i in range(len(cams)):
        patch_base = patches[i]
        cam_map = cams[i]

        cam_uint8 = np.clip(cam_map * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        base_rgb = cv2.cvtColor(np.clip(patch_base * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(base_rgb, 0.4, heatmap_rgb, 0.6, 0)

        result_frames.append({
            "step": f"T-{SEQ_LENGTH - 1 - i}",
            "patch_base64": f"data:image/png;base64,{array_to_base64_png(base_rgb)}",
            "gradcam_base64": f"data:image/png;base64,{array_to_base64_png(blended)}",
            "peak_attention_score": float(np.max(cam_map))
        })

    return {
        "mathematical_formula": "L_GradCAM = ReLU(sum_k (alpha_k * A_k)) across 4-channel spatio-temporal activations",
        "target_layer": "SolarFlarePredictor.encoder[3] (Conv2d 32-channel)",
        "frames": result_frames
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
