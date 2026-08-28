"""
☀️ Configuration Manager & Directory Initialization
Parses config.yaml with fallback defaults and sets up structured storage:
  - data/raw/
  - data/processed/
  - data/catalogs/
  - data/demo_scenarios/
  - models/baseline/
  - models/latest/
"""

import os
from pathlib import Path
import yaml

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Load config.yaml if present
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"
if CONFIG_YAML_PATH.exists():
    try:
        with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not parse config.yaml, using defaults. ({e})")
        CONFIG = {}
else:
    CONFIG = {}

# Paths Configuration
BASE_DIR = Path(os.getenv("SOLAR_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = BASE_DIR / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "processed"
CATALOGS_DIR = BASE_DIR / "catalogs"
DEMO_SCENARIOS_DIR = BASE_DIR / "demo_scenarios"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_LATEST_DIR = MODELS_DIR / "latest"
MODELS_BASELINE_DIR = MODELS_DIR / "baseline"

# Backward compatibility paths
DATA_DIR = BASE_DIR / "full_resolution"
OUTPUT_DIR = BASE_DIR / "processed_patches"

# Ensure all structured directories exist
for directory in [
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CATALOGS_DIR,
    DEMO_SCENARIOS_DIR,
    MODELS_DIR,
    MODELS_LATEST_DIR,
    MODELS_BASELINE_DIR,
    DATA_DIR,
    OUTPUT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Pipeline & Model Hyperparameters
IMG_SIZE = tuple(CONFIG.get("data", {}).get("image_size", [256, 256]))
FULL_DISK_SIZE = tuple(CONFIG.get("data", {}).get("full_disk_size", [512, 512]))
IN_CHANNELS = int(CONFIG.get("data", {}).get("in_channels", 4))
SEQ_LENGTH = int(CONFIG.get("data", {}).get("sequence_length", 4))
CADENCE_HOURS = int(CONFIG.get("data", {}).get("cadence_hours", 3))

# Forecasting Parameters
HORIZON_START_HOURS = int(CONFIG.get("forecasting", {}).get("horizon_start_hours", 24))
HORIZON_END_HOURS = int(CONFIG.get("forecasting", {}).get("horizon_end_hours", 48))
MAJOR_FLARE_THRESHOLD_FLUX = float(CONFIG.get("forecasting", {}).get("major_flare_threshold_flux", 1.0e-5))

# Training Parameters
BATCH_SIZE = int(CONFIG.get("training", {}).get("batch_size", 4))
LEARNING_RATE = float(CONFIG.get("training", {}).get("learning_rate", 1e-4))
NUM_EPOCHS = int(CONFIG.get("training", {}).get("num_epochs", 20))
RANDOM_SEED = int(CONFIG.get("system", {}).get("random_seed", 42))

# Active-Region-Aware Splits
TRAIN_ACTIVE_REGIONS = CONFIG.get("training", {}).get("train_active_regions", ["AR-13664", "AR-12673", "AR-11158"])
VAL_ACTIVE_REGIONS = CONFIG.get("training", {}).get("val_active_regions", ["AR-12887"])
TEST_ACTIVE_REGIONS = CONFIG.get("training", {}).get("test_active_regions", ["AR-13000", "AR-13100"])

# Alert Risk Thresholds
ALERT_THRESHOLDS = CONFIG.get("alert_thresholds", {
    "low_risk": 0.30,
    "moderate_risk": 0.55,
    "high_risk": 0.75
})