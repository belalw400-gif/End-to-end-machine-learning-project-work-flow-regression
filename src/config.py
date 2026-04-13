"""
Configuration management for the ML project.
Handles paths, environment variables, and settings.
"""

import os
from pathlib import Path
from typing import Optional
from src.logger import get_logger

logger = get_logger(__name__)

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Create all required directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== MODEL PATHS ====================
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "xgb_best_model.pkl"))
FREQ_ENCODER_PATH = os.getenv("FREQ_ENCODER_PATH", str(MODELS_DIR / "freq_encoder.pkl"))
TARGET_ENCODER_PATH = os.getenv("TARGET_ENCODER_PATH", str(MODELS_DIR / "target_encoder.pkl"))

# ==================== DATA PATHS ====================
TRAIN_FE_PATH = os.getenv("TRAIN_FE_PATH", str(PROCESSED_DATA_DIR / "train_fe.csv"))
EVAL_FE_PATH = os.getenv("EVAL_FE_PATH", str(PROCESSED_DATA_DIR / "eval_fe.csv"))
HOLDOUT_FE_PATH = os.getenv("HOLDOUT_FE_PATH", str(PROCESSED_DATA_DIR / "holdout_fe.csv"))
HOLDOUT_CLEANED_PATH = os.getenv("HOLDOUT_CLEANED_PATH", str(PROCESSED_DATA_DIR / "holdout_cleaned.csv"))
USMETROS_PATH = os.getenv("USMETROS_PATH", str(RAW_DATA_DIR / "usmetros.csv"))

# ==================== API CONFIGURATION ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

# ==================== MLFLOW CONFIGURATION ====================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "housing_regression")

# ==================== FEATURE ENGINEERING PARAMETERS ====================
PRICE_OUTLIER_THRESHOLD = float(os.getenv("PRICE_OUTLIER_THRESHOLD", "19000000"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


def validate_paths() -> bool:
    """Validate that all necessary data paths exist."""
    required_paths = [
        TRAIN_FE_PATH,
        EVAL_FE_PATH,
    ]
    
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        logger.warning(f"Missing required data files: {missing}")
        return False
    
    return True


def get_data_path(name: str) -> Path:
    """Get path for a specific data file."""
    mapping = {
        "train": TRAIN_FE_PATH,
        "eval": EVAL_FE_PATH,
        "holdout": HOLDOUT_FE_PATH,
        "raw_train": RAW_DATA_DIR / "train.csv",
        "raw_eval": RAW_DATA_DIR / "eval.csv",
        "raw_holdout": RAW_DATA_DIR / "holdout.csv",
    }
    return Path(mapping.get(name, name))


def log_config():
    """Log configuration for debugging."""
    logger.info("=== PROJECT CONFIGURATION ===")
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    logger.info(f"MODELS_DIR: {MODELS_DIR}")
    logger.info(f"MODEL_PATH: {MODEL_PATH}")
    logger.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    logger.info("============================")
