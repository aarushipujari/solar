"""
Generates realistic synthetic Aditya-L1 SUIT multi-spectral FITS files
with header metadata (DATE-OBS, WAVELNTH, NOAA_AR) and companion GOES X-ray flare event catalog.
Includes distinct test scenarios:
  1. AR3664 - Impending X-Class Superflare (High Risk)
  2. AR3670 - Quiet Sun Nominal State (Low Risk)
  3. AR3685 - M-Class Eruptive Active Region (Moderate Risk)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from config import BASE_DIR, DATA_DIR


def generate_solar_frame(size=512, time_step=0, scenario="X_FLARE"):
    """
    Simulates physics of solar disk (limb darkening, background corona, magnetic active regions).
    """
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2
    radius = size * 0.42

    # Solar disk mask
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    disk_mask = dist_from_center <= radius

    # Limb darkening effect
    mu = np.sqrt(np.maximum(0.0, 1.0 - (dist_from_center / radius) ** 2))
    solar_disk = (0.3 + 0.7 * mu) * disk_mask * 1000.0

    # Background space noise
    noise = np.random.normal(5.0, 2.0, (size, size))
    image = np.where(disk_mask, solar_disk + np.random.normal(0, 12.0, (size, size)), noise)

    if scenario == "X_FLARE":
        # Rapidly shearing bipole with flux emergence
        ar_cy = center_y - 45 + int(time_step * 2)
        ar_cx = center_x + 35 + int(time_step * 4)
        intensity = 3500.0 + (time_step * 1400.0)
        spread = 22 + (time_step * 3)

        # Primary delta-spot core
        spot1 = np.exp(-((x - ar_cx) ** 2 + (y - ar_cy) ** 2) / (2 * spread ** 2))
        # Opposite polarity shear filament
        spot2 = np.exp(-((x - (ar_cx - 15)) ** 2 + (y - (ar_cy + 15)) ** 2) / (2 * (spread - 4) ** 2))
        image += (spot1 * intensity) + (spot2 * intensity * 0.75)

    elif scenario == "M_FLARE":
        # Moderate active region with intermediate shear
        ar_cy = center_y + 30 + int(time_step * 1)
        ar_cx = center_x - 40 + int(time_step * 2)
        intensity = 1800.0 + (time_step * 450.0)
        spot = np.exp(-((x - ar_cx) ** 2 + (y - ar_cy) ** 2) / (2 * 16 ** 2))
        image += spot * intensity

    else:  # QUIET_SUN
        # Small stable unipolar sunspot
        ar_cy = center_y + 60
        ar_cx = center_x - 70
        spot = np.exp(-((x - ar_cx) ** 2 + (y - ar_cy) ** 2) / (2 * 10 ** 2))
        image += spot * 400.0

    image = np.clip(image, 0.0, None).astype(np.float32)
    return image


def save_fits_with_header(filepath, data, obs_time, noaa_ar, goes_class, flare_label, peak_flux):
    """Writes standard FITS HDU with Aditya-L1 and GOES metadata tags."""
    hdu = fits.PrimaryHDU(data)
    hdr = hdu.header
    hdr["TELESCOP"] = "Aditya-L1"
    hdr["INSTRUME"] = "SUIT"
    hdr["WAVELNTH"] = "279.6 nm (Mg II k)"
    hdr["DATE-OBS"] = obs_time
    hdr["NOAA_AR"] = noaa_ar
    hdr["GOES_CLASS"] = goes_class
    hdr["FLARE_LABEL"] = flare_label
    hdr["PEAK_FLUX"] = peak_flux
    hdr["EXPTIME"] = 0.250  # Seconds
    hdr["OBS_GEO"] = "Lagrange Point L1"
    hdu.writeto(filepath, overwrite=True)


def build_all_datasets_and_catalog():
    print("Building Aditya-L1 SUIT observation datasets & GOES flare catalog...")

    scenarios = {
        "AR3664_Impending_X_Flare": {
            "type": "X_FLARE",
            "ar": "AR-3664",
            "class": "X2.8",
            "label": 1,
            "flux": "2.8e-4",
            "dest": BASE_DIR / "scenarios" / "AR3664_Impending_X_Flare"
        },
        "AR3670_Quiet_Sun": {
            "type": "QUIET_SUN",
            "ar": "AR-3670",
            "class": "Quiet",
            "label": 0,
            "flux": "1.2e-7",
            "dest": BASE_DIR / "scenarios" / "AR3670_Quiet_Sun"
        },
        "AR3685_M_Class_Eruption": {
            "type": "M_FLARE",
            "ar": "AR-3685",
            "class": "M5.4",
            "label": 1,
            "flux": "5.4e-5",
            "dest": BASE_DIR / "scenarios" / "AR3685_M_Class_Eruption"
        }
    }

    catalog_records = []

    # 1. Generate named scenarios
    for sc_name, sc_info in scenarios.items():
        dest = sc_info["dest"]
        dest.mkdir(parents=True, exist_ok=True)
        for t in range(4):
            obs_time = f"2026-08-28T{10 + t:02d}:00:00.000"
            fname = f"{sc_name}_T{t:02d}.fits"
            fpath = dest / fname
            data = generate_solar_frame(time_step=t, scenario=sc_info["type"])
            save_fits_with_header(
                fpath, data, obs_time, sc_info["ar"],
                sc_info["class"], sc_info["label"], sc_info["flux"]
            )
            catalog_records.append({
                "obs_file": fname,
                "date_obs": obs_time,
                "active_region": sc_info["ar"],
                "flare_class": sc_info["class"],
                "peak_flux_wm2": sc_info["flux"],
                "forecast_label_24h": sc_info["label"]
            })

    # 2. Populate main DATA_DIR with a full 8-frame temporal progression
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for t in range(8):
        obs_time = f"2026-08-28T{0 + t:02d}:30:00.000"
        fname = f"suit_obs_2026Aug28_T{t:02d}.fits"
        fpath = DATA_DIR / fname
        flare_active = t >= 4
        data = generate_solar_frame(time_step=t, scenario="X_FLARE" if flare_active else "M_FLARE")
        goes_class = "X1.4" if t >= 6 else ("M3.2" if t >= 4 else "C2.1")
        label = 1 if flare_active else 0
        save_fits_with_header(
            fpath, data, obs_time, "AR-3664",
            goes_class, label, "1.4e-4" if flare_active else "3.2e-5"
        )
        catalog_records.append({
            "obs_file": fname,
            "date_obs": obs_time,
            "active_region": "AR-3664",
            "flare_class": goes_class,
            "peak_flux_wm2": "1.4e-4" if flare_active else "3.2e-5",
            "forecast_label_24h": label
        })

    # 3. Save GOES Flare Catalog CSV
    catalog_df = pd.DataFrame(catalog_records)
    catalog_path = BASE_DIR / "goes_flare_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    print(f"Created GOES catalog at {catalog_path} with {len(catalog_df)} entries.")
    print("Dataset and scenario generation complete!")


if __name__ == "__main__":
    build_all_datasets_and_catalog()
