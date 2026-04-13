"""
Evaluate a saved XGBoost model on the eval split.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVAL = Path("data/processed/eval_fe.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_EVAL,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate model on evaluation set.
    
    Args:
        model_path: Path to saved model
        eval_path: Path to evaluation data
        sample_frac: Optional fraction for sampling
        random_state: Random seed
        
    Returns:
        Dictionary with mae, rmse, r2 metrics
        
    Raises:
        FileNotFoundError: If model or data not found
        ValueError: If data is empty or invalid
    """
    try:
        eval_df = pd.read_csv(eval_path)
        logger.info(f"✅ Loaded evaluation data: {eval_df.shape}")
    except FileNotFoundError as e:
        logger.error(f"❌ Evaluation data not found: {eval_path}")
        raise
    
    if eval_df.empty:
        raise ValueError("Evaluation data is empty")
    
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    if target not in eval_df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    try:
        model = load(model_path)
        logger.info(f"✅ Loaded model from {model_path}")
    except FileNotFoundError as e:
        logger.error(f"❌ Model not found: {model_path}")
        raise
    
    y_pred = model.predict(X_eval)

    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    logger.info("📊 Evaluation:")
    logger.info(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()