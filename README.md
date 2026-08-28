# ☀️ Aditya-L1 Solar Flare & Space Weather Warning System

A spatio-temporal deep learning pipeline that processes multi-spectral UV solar imagery from ISRO's **Aditya-L1 SUIT** payload to forecast solar flares 24–48 hours prior to eruption.

---

## 📌 Features
* **FITS Data Processing**: Automated cleaning, solar disk registration, and dynamic intensity scaling using `Astropy`.
* **Active Region Cropping**: Focuses on high-flux magnetic regions to extract localized solar patches.
* **Spatio-Temporal Deep Learning**: Combines CNN spatial encoders with a **ConvLSTM** sequence layer to track temporal magnetic shear evolution.
* **Interactive Dashboard**: A Streamlit web UI featuring real-time patch inspection and Plotly risk dial indicators.

---

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/your-username/solar-flare-warning-system.git](https://github.com/your-username/solar-flare-warning-system.git)
   cd solar-flare-warning-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

1. **Train the Model**:
   ```bash
   python train.py
   ```

2. **Launch the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
