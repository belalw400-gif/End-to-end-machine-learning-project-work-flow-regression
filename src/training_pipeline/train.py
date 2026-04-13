"""
Train a baseline XGBoost model.

- Reads feature-engineered train/eval CSVs.
- Trains XGBRegressor.
- Returns metrics and saves model to `model_output`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TRAIN = Path("data/processed/train_fe.csv")
DEFAULT_EVAL = Path("data/processed/eval_fe.csv")
DEFAULT_OUT = Path("models/xgb_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def train_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    model_params: Optional[Dict] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    """Train baseline XGB and save model.

    Args:
        train_path: Path to training data
        eval_path: Path to evaluation data
        model_output: Path to save model
        model_params: Optional model parameters to override defaults
        sample_frac: Optional fraction for sampling (for testing)
        random_state: Random seed

    Returns:
        Tuple of (model, metrics dict)
        
    Raises:
        FileNotFoundError: If training or eval data not found
        ValueError: If data is empty or invalid
    """
    try:
        train_df = pd.read_csv(train_path)
        logger.info(f"✅ Loaded training data: {train_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"❌ Training data not found: {train_path}")
        raise
    
    try:
        eval_df = pd.read_csv(eval_path)
        logger.info(f"✅ Loaded evaluation data: {eval_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"❌ Evaluation data not found: {eval_path}")
        raise

    if train_df.empty or eval_df.empty:
        raise ValueError("Training or evaluation data is empty")

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    if target not in train_df.columns or target not in eval_df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if model_params:
        params.update(model_params)

    logger.info(f"Training XGBoost with params: {params}")
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    logger.info(f"✅ Model trained. Saved to {out}")
    logger.info(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    return model, metrics


if __name__ == "__main__":
    train_model()