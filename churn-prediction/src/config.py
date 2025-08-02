"""
Configuration management for the project.
Industry practice: Centralize all configs
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Model paths
MODEL_PATH = PROJECT_ROOT / "models"

# MLflow tracking URI
MLFLOW_TRACKING_URI = "file://" + str(PROJECT_ROOT / "mlruns")

# Random seed for reproducibility (ALWAYS set this!)
RANDOM_SEED = 42
