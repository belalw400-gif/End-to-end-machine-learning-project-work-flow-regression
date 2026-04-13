from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from src.inference_pipeline.inference import predict
from src.logger import get_logger

logger = get_logger(__name__)


# ----------------------------
# Config / Paths
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "train_fe.csv"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"


# ----------------------------
# Load training feature columns (once)
# ----------------------------

def load_train_columns():
    if TRAIN_FE_PATH.exists():
        df = pd.read_csv(TRAIN_FE_PATH, nrows=1)
        return [c for c in df.columns if c != "price"]
    return None


TRAIN_FEATURE_COLUMNS = load_train_columns()


# ----------------------------
# App
# ----------------------------

app = FastAPI(title="Housing Regression API")


@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"message": "Housing Regression API is running 🚀", "version": "1.0.0"}


# ----------------------------
# Health Check
# ----------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    logger.debug("Health check requested")
    status: Dict[str, Any] = {
        "status": "unhealthy",
        "model_path": str(MODEL_PATH),
        "train_features_path": str(TRAIN_FE_PATH),
    }

    if not MODEL_PATH.exists():
        logger.warning(f"Model not found at {MODEL_PATH}")
        status.update({
            "error": "Model not found"
        })
        return status

    status["status"] = "healthy"

    if TRAIN_FEATURE_COLUMNS:
        status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
        logger.info(f"Health: OK. Model ready with {len(TRAIN_FEATURE_COLUMNS)} features")

    return status


# ----------------------------
# Latest Predictions
# ----------------------------

@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    """Get latest predictions from cache."""
    logger.info(f"Fetching latest {limit} predictions")
    
    # 1. check folder exists
    if not PRED_DIR.exists():
        logger.warning(f"Predictions folder not found: {PRED_DIR}")
        raise HTTPException(status_code=404, detail="Predictions folder not found")

    # 2. get files sorted by last modified time (correct way)
    files = list(PRED_DIR.glob("preds_*.csv"))

    if not files:
        logger.warning("No prediction files found")
        raise HTTPException(status_code=404, detail="No predictions found")

    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Reading latest predictions from {latest_file.name}")

    # 3. read only needed rows (performance optimization)
    try:
        df = pd.read_csv(latest_file, nrows=limit)
        logger.info(f"Successfully read {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading predictions file: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    return {
        "file": latest_file.name,
        "rows_previewed": int(len(df)),
        "preview": df.to_dict(orient="records")
    }


# ----------------------------
# Predict Endpoint
# ----------------------------

@app.post("/predict")
def predict_prices(data: List[Dict[str, Any]]):
    """Generate predictions on input data.
    
    Args:
        data: List of dictionaries with feature data
        
    Returns:
        Dictionary with predictions and optional actuals
        
    Raises:
        HTTPException: If data is invalid or prediction fails
    """
    logger.info(f"Prediction request received for {len(data)} samples")
    
    if not data:
        logger.warning("Empty data received in prediction request")
        raise HTTPException(status_code=400, detail="No data provided")

    try:
        df = pd.DataFrame(data)
        logger.debug(f"Converted data to DataFrame: {df.shape}")
    except Exception as e:
        logger.error(f"Error converting data to DataFrame: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")

    if df.empty:
        logger.warning("Converted DataFrame is empty")
        raise HTTPException(status_code=400, detail="No valid data provided")

    # Since data is already feature engineered, predict directly
    from joblib import load
    
    try:
        model = load(MODEL_PATH)
        logger.debug(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"Model not found: {MODEL_PATH}")
        raise HTTPException(status_code=500, detail="Model not found. Please retrain the model.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # If actual prices are provided
    actuals = None
    if "price" in df.columns:
        actuals = df["price"].tolist()
    
    # Align columns to training features
    if TRAIN_FEATURE_COLUMNS:
        try:
            df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)
            logger.debug(f"Aligned features to training schema")
        except Exception as e:
            logger.error(f"Error aligning features: {e}")
            raise HTTPException(status_code=400, detail=f"Feature alignment error: {str(e)}")

    try:
        preds = model.predict(df)
        logger.info(f"Successfully generated {len(preds)} predictions")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    result = {"predictions": preds.tolist()}

    if actuals is not None:
        result["actuals"] = actuals
    
    return result


