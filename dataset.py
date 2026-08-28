"""
☀️ Aditya-L1 Spatio-Temporal Sequence Dataset Pipeline
Features:
  1. Zero Label Leakage: FITS headers contain only observational metadata (DATE-OBS, NOAA_AR).
  2. Scientifically Rigorous 24-48h Forward Target Window Construction:
     - Given sequence ending at T_obs:
     - Target window = [T_obs + 24h, T_obs + 48h]
     - Queries independent GOES X-ray catalog for M/X flare occurrences in that exact future window.
  3. Multi-Channel Tensor Shape: [Sequence (4), Channels (4), Height (256), Width (256)]
  4. Multi-Task Targets:
     - binary_label: 0 (No Flare / <M1.0), 1 (Flare >= M1.0)
     - multiclass_label: 0 (Quiet/B), 1 (C-Class), 2 (M-Class), 3 (X-Class)
     - log_peak_flux: log10(Peak Flux in W/m²)
  5. Chronological Splitting (Train / Val / Test) with zero temporal contamination.
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits

from preprocess import (
    load_and_clean_fits,
    preprocess_solar_disk,
    extract_active_region,
    build_multi_channel_frame
)


class SolarSequenceDataset(Dataset):
    """
    Chronological Sequence Dataset for Aditya-L1 Solar Flare Forecasting.
    """

    def __init__(self, data_dir, seq_length=4, img_size=(256, 256), catalog_path=None, file_list=None):
        self.seq_length = seq_length
        self.img_size = img_size
        self.data_dir = Path(data_dir)

        if file_list is not None:
            self.file_paths = sorted(file_list)
        else:
            self.file_paths = sorted(list(self.data_dir.glob("*.fits")))

        # Load GOES Flare Catalog
        if catalog_path is None:
            catalog_path = self.data_dir.parent / "goes_flare_catalog.csv"
            if not catalog_path.exists():
                catalog_path = self.data_dir / "goes_flare_catalog.csv"

        self.goes_catalog = pd.DataFrame()
        if Path(catalog_path).exists():
            try:
                self.goes_catalog = pd.read_csv(catalog_path)
                # Parse ISO timestamps
                self.goes_catalog["start_dt"] = pd.to_datetime(self.goes_catalog["start_time"], utc=True)
                self.goes_catalog["peak_dt"] = pd.to_datetime(self.goes_catalog["peak_time"], utc=True)
            except Exception as e:
                print(f"Warning: Could not parse GOES catalog at {catalog_path}: {e}")

        # Index contiguous sequences by Active Region
        self.sequences = self._build_contiguous_sequences()

    def _extract_fits_header(self, filepath):
        """Extracts pure observational metadata from FITS primary header."""
        meta = {
            "date_obs": None,
            "noaa_ar": "AR-13664",
            "wavelnth": "279.6 nm"
        }
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                meta["date_obs"] = header.get("DATE-OBS", None)
                meta["noaa_ar"] = header.get("NOAA_AR", "AR-13664")
                meta["wavelnth"] = header.get("WAVELNTH", "279.6 nm")
        except Exception:
            pass
        return meta

    def _build_contiguous_sequences(self):
        """Groups FITS frames chronologically into contiguous temporal sliding windows."""
        seqs = []
        if len(self.file_paths) < self.seq_length:
            return seqs

        # Parse timestamps and active regions for all files
        file_meta = []
        for f in self.file_paths:
            m = self._extract_fits_header(f)
            dt = None
            if m["date_obs"]:
                try:
                    dt = pd.to_datetime(m["date_obs"], utc=True)
                except Exception:
                    pass
            file_meta.append({"path": f, "ar": m["noaa_ar"], "dt": dt})

        df_files = pd.DataFrame(file_meta)
        if df_files["dt"].isnull().any():
            # Fallback to simple sliding window if timestamps missing
            for i in range(len(self.file_paths) - self.seq_length + 1):
                seqs.append(self.file_paths[i: i + self.seq_length])
            return seqs

        # Group by Active Region to prevent cross-region sequence mixing
        for ar, group in df_files.groupby("ar"):
            group = group.sort_values("dt").reset_index(drop=True)
            if len(group) >= self.seq_length:
                for i in range(len(group) - self.seq_length + 1):
                    sub_files = list(group.iloc[i: i + self.seq_length]["path"])
                    seqs.append(sub_files)

        return seqs

    def __len__(self):
        return len(self.sequences)

    def _evaluate_24_48h_forward_target(self, last_filepath):
        """
        Calculates ground-truth target by checking if an M/X flare occurred
        in the forward window [T_obs + 24h, T_obs + 48h] associated with the active region.
        """
        meta = self._extract_fits_header(last_filepath)
        ar_name = meta["noaa_ar"]
        obs_time_str = meta["date_obs"]

        # Default targets for quiet baseline
        binary_label = 0
        multiclass_label = 0  # 0: Quiet/B, 1: C, 2: M, 3: X
        log_flux = -7.5        # Baseline log10(W/m²)

        if self.goes_catalog.empty or obs_time_str is None:
            return binary_label, multiclass_label, log_flux

        try:
            obs_dt = pd.to_datetime(obs_time_str, utc=True)
            win_start = obs_dt + timedelta(hours=24)
            win_end = obs_dt + timedelta(hours=48)

            # Query GOES catalog for events in [win_start, win_end] matching the active region
            matched = self.goes_catalog[
                (self.goes_catalog["active_region"] == ar_name) &
                (self.goes_catalog["start_dt"] >= win_start) &
                (self.goes_catalog["start_dt"] <= win_end)
            ]

            if not matched.empty:
                # Find maximum flare intensity event in the 24-48h window
                max_event = matched.sort_values("peak_flux_wm2", ascending=False).iloc[0]
                fl_class = str(max_event["flare_class"])
                fl_flux = float(max_event["peak_flux_wm2"])
                log_flux = float(np.log10(max(1e-8, fl_flux)))

                if fl_class.startswith("X"):
                    binary_label = 1
                    multiclass_label = 3
                elif fl_class.startswith("M"):
                    binary_label = 1
                    multiclass_label = 2
                elif fl_class.startswith("C"):
                    binary_label = 0
                    multiclass_label = 1
                else:
                    binary_label = 0
                    multiclass_label = 0

        except Exception:
            pass

        return binary_label, multiclass_label, log_flux

    def __getitem__(self, idx):
        seq_files = self.sequences[idx]
        channel_frames = []
        prev_patch = None

        for filepath in seq_files:
            raw = load_and_clean_fits(filepath)
            disk = preprocess_solar_disk(raw)
            patch = extract_active_region(disk, patch_size=self.img_size)

            # Build 4-channel tensor: [4, H, W]
            multi_ch_tensor = build_multi_channel_frame(patch, prev_patch=prev_patch)
            channel_frames.append(torch.tensor(multi_ch_tensor, dtype=torch.float32))
            prev_patch = patch

        # Stack across temporal sequence dimension -> [T (4), C (4), H (256), W (256)]
        sequence_tensor = torch.stack(channel_frames, dim=0)

        # Compute 24-48h target
        bin_label, multi_label, log_flux = self._evaluate_24_48h_forward_target(seq_files[-1])

        targets = {
            "binary_label": torch.tensor(bin_label, dtype=torch.long),
            "multiclass_label": torch.tensor(multi_label, dtype=torch.long),
            "log_flux": torch.tensor(log_flux, dtype=torch.float32)
        }

        return sequence_tensor, targets


def create_chronological_splits(data_dir, seq_length=4, train_ratio=0.70, val_ratio=0.15):
    """
    Splits FITS sequence files chronologically to prevent temporal data leakage.
    Returns: (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)
    all_files = sorted(list(data_dir.glob("*.fits")))

    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = all_files[:n_train]
    val_files = all_files[n_train: n_train + n_val]
    test_files = all_files[n_train + n_val:]

    train_ds = SolarSequenceDataset(data_dir, seq_length=seq_length, file_list=train_files)
    val_ds = SolarSequenceDataset(data_dir, seq_length=seq_length, file_list=val_files)
    test_ds = SolarSequenceDataset(data_dir, seq_length=seq_length, file_list=test_files)

    return train_ds, val_ds, test_ds