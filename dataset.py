from pathlib import Path
import torch
from torch.utils.data import Dataset
from preprocess import load_and_clean_fits, preprocess_solar_disk, extract_active_region


class SolarSequenceDataset(Dataset):
    """
    Groups chronologically ordered FITS files into temporal sliding windows.
    Output Tensor Shape: [Sequence_Length, Channels, Height, Width]
    """

    def __init__(self, data_dir, seq_length=4, img_size=(256, 256)):
        self.seq_length = seq_length
        self.img_size = img_size

        # Chronologically sort files based on filenames
        self.file_paths = sorted(list(Path(data_dir).glob("*.fits")))

        # Calculate available temporal sequences
        self.num_sequences = len(self.file_paths) - seq_length + 1

    def __len__(self):
        return max(0, self.num_sequences)

    def __getitem__(self, idx):
        # Extract a contiguous sequence of FITS frames
        seq_files = self.file_paths[idx: idx + self.seq_length]
        frames = []

        for filepath in seq_files:
            raw = load_and_clean_fits(filepath)
            disk = preprocess_solar_disk(raw)
            patch = extract_active_region(disk, patch_size=self.img_size)

            # Add channel dimension -> [1, H, W]
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
            frames.append(patch_tensor)

        # Stack across sequence length dimension -> [T, C, H, W]
        sequence_tensor = torch.stack(frames, dim=0)

        # Dummy ground-truth target (0 = No Flare, 1 = M/X Class Flare)
        # Replace with actual NOAA/GOES flare logs matched by timestamp
        target_label = torch.tensor(0, dtype=torch.long)

        return sequence_tensor, target_label