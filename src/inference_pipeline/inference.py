"""
Inference pipeline for Housing Regression MLE.

- Takes RAW input data (same schema as holdout.csv).
- Applies preprocessing + feature engineering using saved encoders.
- Aligns features with training.
- Returns predictions.
"""

# Raw → preprocess → feature engineering → align schema → model.predict → predictions.

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing + feature engineering helpers
from src.feature_pipeline.preprocess import clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_date_features, drop_unused_columns
from src.logger import get_logger

logger = get_logger(__name__)

# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "train_fe.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("📂 Inference using project root:", PROJECT_ROOT)

# Load training feature columns (strict schema from training dataset)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]  # excluding price column
else:
    TRAIN_FEATURE_COLUMNS = None


def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    """Run inference on input data.
    
    Args:
        input_df: Raw input DataFrame
        model_path: Path to trained model
        freq_encoder_path: Path to frequency encoder
        target_encoder_path: Path to target encoder
        
    Returns:
        DataFrame with predictions
        
    Raises:
        FileNotFoundError: If model or encoders not found
        ValueError: If data is invalid
    """
    if input_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    logger.info(f"📥 Running inference on {len(input_df)} samples")
    
    # Step 1: Preprocess raw input
    try:
        df = clean_and_merge(input_df.copy())
        df = drop_duplicates(df)
        df = remove_outliers(df)
        logger.debug("✅ Preprocessing complete")
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise ValueError(f"Data preprocessing error: {e}")

    # Step 2: Feature engineering
    try:
        if "date" in df.columns:
            df = add_date_features(df)
        logger.debug("✅ Date features added")
    except Exception as e:
        logger.warning(f"⚠️ Date feature engineering failed: {e}")

    # Step 3: Encodings ----------------
    # Frequency encoding (zipcode)
    freq_encoder_exists = Path(freq_encoder_path).exists()
    if freq_encoder_exists and "zipcode" in df.columns:
        try:
            freq_map = load(freq_encoder_path)
            df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
            df = df.drop(columns=["zipcode"], errors="ignore")
            logger.debug("✅ Frequency encoding applied")
        except Exception as e:
            logger.warning(f"⚠️  Frequency encoding failed: {e}")
            df["zipcode_freq"] = 0
    elif "zipcode" in df.columns and not freq_encoder_exists:
        logger.warning(f"⚠️  Frequency encoder not found at {freq_encoder_path}")
        df["zipcode_freq"] = 0

    # Target encoding (city_full → city_full_te)
    target_encoder_exists = Path(target_encoder_path).exists()
    if "city_full" in df.columns and target_encoder_exists:
        try:
            target_encoder = load(target_encoder_path)
            df["city_full_te"] = target_encoder.transform(df["city_full"])
            df = df.drop(columns=["city_full"], errors="ignore")
            logger.debug("✅ Target encoding applied")
        except Exception as e:
            logger.warning(f"⚠️  Target encoding failed: {e}")
    elif "city_full" in df.columns and not target_encoder_exists:
        logger.warning(f"⚠️  Target encoder not found at {target_encoder_path}")

    # Drop leakage columns
    try:
        df, _ = drop_unused_columns(df.copy(), df.copy())
        logger.debug("✅ Leakage columns dropped")
    except Exception as e:
        logger.warning(f"⚠️  Column dropping failed: {e}")

    # Step 4: Separate actuals if present
    y_true = None
    if "price" in df.columns:
        y_true = df["price"].tolist()
        df = df.drop(columns=["price"])

    # Step 5: Load model and align columns with the model schema
    try:
        model = load(model_path)
        logger.debug(f"✅ Model loaded from {model_path}")
    except FileNotFoundError:
        logger.error(f"❌ Model not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise ValueError(f"Model loading error: {e}")

    model_feature_names = None
    try:
        model_feature_names = model.get_booster().feature_names
    except Exception:
        model_feature_names = getattr(model, "feature_names", None)

    if model_feature_names is not None:
        df = df.reindex(columns=model_feature_names, fill_value=0)
        logger.debug(f"✅ Aligned {len(model_feature_names)} features from model")
    elif TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)
        logger.debug(f"✅ Aligned {len(TRAIN_FEATURE_COLUMNS)} features from training data")

    # Step 6: Predict
    try:
        preds = model.predict(df)
        logger.info(f"✅ Generated {len(preds)} predictions")
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise ValueError(f"Model prediction error: {e}")
    
    # Step 7: Build output
    out = df.copy()
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true

    return out
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true

    return out


# ----------------------------
# CLI entrypoint
# ----------------------------
# Allows running inference directly from terminal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new housing data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")
    parser.add_argument("--freq_encoder", type=str, default=str(DEFAULT_FREQ_ENCODER), help="Path to frequency encoder pickle")
    parser.add_argument("--target_encoder", type=str, default=str(DEFAULT_TARGET_ENCODER), help="Path to target encoder pickle")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")