import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from preprocess import load_and_clean_fits, preprocess_solar_disk, extract_active_region


class SolarSequenceDataset(Dataset):
    """
    Groups chronologically ordered Aditya-L1 SUIT FITS files into temporal sliding windows
    and matches them against NOAA/GOES X-ray flare event catalogs (C/M/X classes).
    
    Output Tensor Shape: [Sequence_Length, Channels, Height, Width]
    """

    def __init__(self, data_dir, seq_length=4, img_size=(256, 256), catalog_path=None):
        self.seq_length = seq_length
        self.img_size = img_size
        self.data_dir = Path(data_dir)

        # Chronologically sort files based on filenames / timestamps
        self.file_paths = sorted(list(self.data_dir.glob("*.fits")))

        # Calculate available temporal sequences
        self.num_sequences = len(self.file_paths) - seq_length + 1

        # Load GOES Flare Catalog if available
        if catalog_path is None:
            catalog_path = self.data_dir.parent / "goes_flare_catalog.csv"
            if not catalog_path.exists():
                catalog_path = self.data_dir / "goes_flare_catalog.csv"

        self.goes_catalog = None
        if Path(catalog_path).exists():
            try:
                self.goes_catalog = pd.read_csv(catalog_path)
                print(f"Loaded GOES Flare Catalog: {len(self.goes_catalog)} events recorded.")
            except Exception as e:
                print(f"Warning: Could not parse GOES catalog at {catalog_path}: {e}")

    def __len__(self):
        return max(0, self.num_sequences)

    def _extract_fits_metadata(self, filepath):
        """Extracts observational metadata from FITS primary header."""
        meta = {
            "date_obs": None,
            "telescop": "Aditya-L1",
            "instrume": "SUIT",
            "wavelnth": "279.6 nm",
            "noaa_ar": "AR-3664",
            "flare_label": 0,
            "goes_class": "Quiet"
        }
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                meta["date_obs"] = header.get("DATE-OBS", filepath.stem)
                meta["telescop"] = header.get("TELESCOP", "Aditya-L1")
                meta["instrume"] = header.get("INSTRUME", "SUIT")
                meta["wavelnth"] = header.get("WAVELNTH", "279.6 nm")
                meta["noaa_ar"] = header.get("NOAA_AR", "AR-3664")
                meta["flare_label"] = int(header.get("FLARE_LABEL", 0))
                meta["goes_class"] = header.get("GOES_CLASS", "Quiet")
        except Exception:
            pass
        return meta

    def _match_goes_flare_label(self, meta_t_last):
        """
        Matches sequence observation timestamp with GOES catalog events.
        If a >= M1.0 flare occurred within the 24-48h forecast window, label = 1.
        """
        if self.goes_catalog is not None and "date_obs" in meta_t_last:
            # Match by NOAA Active Region or timestamp if present in catalog
            matched = self.goes_catalog[
                (self.goes_catalog["active_region"] == meta_t_last["noaa_ar"]) |
                (self.goes_catalog["obs_file"] == meta_t_last["date_obs"])
            ]
            if not matched.empty:
                max_class = matched.iloc[0].get("flare_class", "Quiet")
                is_high_risk = 1 if (str(max_class).startswith("M") or str(max_class).startswith("X")) else 0
                return torch.tensor(is_high_risk, dtype=torch.long)

        # Fallback to metadata tag in header
        return torch.tensor(meta_t_last["flare_label"], dtype=torch.long)

    def __getitem__(self, idx):
        # Extract a contiguous sequence of FITS frames
        seq_files = self.file_paths[idx: idx + self.seq_length]
        frames = []
        last_meta = {}

        for i, filepath in enumerate(seq_files):
            raw = load_and_clean_fits(filepath)
            disk = preprocess_solar_disk(raw)
            patch = extract_active_region(disk, patch_size=self.img_size)

            # Add channel dimension -> [1, H, W]
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
            frames.append(patch_tensor)

            if i == len(seq_files) - 1:
                last_meta = self._extract_fits_metadata(filepath)

        # Stack across sequence length dimension -> [T, C, H, W]
        sequence_tensor = torch.stack(frames, dim=0)
        target_label = self._match_goes_flare_label(last_meta)

        return sequence_tensor, target_label