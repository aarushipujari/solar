import os
from pathlib import Path

# Base directory for the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Paths (defaults to ./data inside the project directory, or override with SOLAR_DATA_DIR)
BASE_DIR = Path(os.getenv("SOLAR_DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR = BASE_DIR / "full_resolution"
OUTPUT_DIR = BASE_DIR / "processed_patches"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data Pipeline Params
IMG_SIZE = (512, 512)       # Full-disk resolution after processing
CROP_SIZE = (256, 256)      # Active Region patch size
SEQ_LENGTH = 4              # Number of historical time-steps per sequence

# Model Params
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20