import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Import functions from project components
from preprocess import load_and_clean_fits, preprocess_solar_disk, extract_active_region
from model import SolarFlarePredictor
from config import BASE_DIR, DATA_DIR, SEQ_LENGTH

# Set Streamlit page configuration
st.set_page_config(
    page_title="ISRO Aditya-L1 | Solar Flare Warning System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Space-Ops Dark Theme Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0b0d17;
        color: #e0e6ed;
    }
    .metric-card {
        background: rgba(23, 27, 44, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .status-safe {
        color: #00e676;
        font-weight: bold;
    }
    .status-warning {
        color: #ff3d00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Cache Model Weights Loading
@st.cache_resource
def load_trained_model():
    model = SolarFlarePredictor()
    model_path = BASE_DIR / "solar_flare_model.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


model = load_trained_model()

# Header & Subtitle
st.title("☀️ Aditya-L1 Space Weather Warning System")
st.caption("Deep Learning-based Early Detection for Solar Flares and Geomagnetic Storms")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("🕹️ Observation Controls")
data_folder = Path(DATA_DIR)
all_fits = sorted(list(data_folder.glob("*.fits")))

if len(all_fits) < SEQ_LENGTH:
    st.error(f"Insufficient FITS files in `{DATA_DIR}`. Found {len(all_fits)}, required minimum {SEQ_LENGTH}.")
    st.stop()

selected_files = st.sidebar.multiselect(
    "Select Temporal Sequence (4 Frames)",
    options=[f.name for f in all_fits],
    default=[f.name for f in all_fits[:SEQ_LENGTH]]
)

confidence_threshold = st.sidebar.slider("Alert Sensitivity Threshold (%)", 0, 100, 60)

# Dashboard Body
if len(selected_files) == SEQ_LENGTH:
    file_paths = [data_folder / f for f in selected_files]

    # Process sequence and create model tensor
    processed_patches = []
    full_disks = []

    for path in file_paths:
        raw = load_and_clean_fits(path)
        disk = preprocess_solar_disk(raw)
        patch = extract_active_region(disk, patch_size=(256, 256))

        full_disks.append(disk)
        processed_patches.append(patch)

    # Tensor conversion: [Batch=1, Time=4, Channel=1, Height=256, Width=256]
    seq_tensor = torch.tensor(np.stack(processed_patches), dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    # Model Inference
    with torch.no_grad():
        logits = model(seq_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        flare_prob = float(probabilities[1]) * 100

    # Layout Columns
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🖼️ Processed Multi-spectral Active Region Patch")
        # Display latest solar observation patch
        st.image(
            processed_patches[-1],
            caption=f"Latest Observation Frame: {selected_files[-1]}",
            use_container_width=True,
            clamp=True
        )

    with col2:
        st.subheader("📊 24-48h Flare Probability Gauge")

        # Plotly Radial Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=flare_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'color': "#ffffff", 'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#ffffff"},
                'bar': {'color': "#ff3d00" if flare_prob >= confidence_threshold else "#00e676"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(0, 230, 118, 0.2)"},
                    {'range': [40, 70], 'color': "rgba(255, 171, 0, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(255, 61, 0, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence_threshold
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"},
            height=280,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Dynamic Status Cards
        if flare_prob >= confidence_threshold:
            st.markdown(
                f"""
                <div class="metric-card" style="border-color: #ff3d00;">
                    <h3 class="status-warning">⚠️ ALERT: HIGH FLARE RISK DETECTED</h3>
                    <p>Estimated Probability: <b>{flare_prob:.1f}%</b> within 24–48 Hours.</p>
                    <p><small>Recommend initiating protective safe-mode for orbital assets and power grid monitoring.</small></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="metric-card" style="border-color: #00e676;">
                    <h3 class="status-safe">✅ SAFE OPERATIONAL CONDITIONS</h3>
                    <p>Estimated Probability: <b>{flare_prob:.1f}%</b> within 24–48 Hours.</p>
                    <p><small>Solar activity remains nominal. Low geomagnetic disturbance risk.</small></p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Frame Inspection Toolbar
    st.markdown("---")
    st.subheader("🕒 Temporal Sequence Inspection")
    cols = st.columns(SEQ_LENGTH)
    for idx, col in enumerate(cols):
        with col:
            st.image(processed_patches[idx], caption=f"T-{SEQ_LENGTH - idx}", use_container_width=True, clamp=True)

else:
    st.warning(f"Please select exactly **{SEQ_LENGTH}** FITS files in the sidebar to run time-series forecasting.")