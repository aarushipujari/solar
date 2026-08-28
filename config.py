from pathlib import Path

# Paths
BASE_DIR = Path(r"C:\Users\Ruma Ghosh\Data\suit_2026Aug28T052143751")
DATA_DIR = BASE_DIR / "full_resolution"
OUTPUT_DIR = BASE_DIR / "processed_patches"

# Data Pipeline Params
IMG_SIZE = (512, 512)       # Full-disk resolution after processing
CROP_SIZE = (256, 256)      # Active Region patch size
SEQ_LENGTH = 4              # Number of historical time-steps per sequence

# Model Params
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20