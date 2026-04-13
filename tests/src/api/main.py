from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from src.inference_pipeline.inference import predict


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
    return {"message": "Housing Regression API is running 🚀"}


# ----------------------------
# Health Check
# ----------------------------

@app.get("/health")
def health():
    status: Dict[str, Any] = {
        "model_path": str(MODEL_PATH),
        "train_features_path": str(TRAIN_FE_PATH),
    }

    if not MODEL_PATH.exists():
        status.update({
            "status": "unhealthy",
            "error": "Model not found"
        })
        return status

    status["status"] = "healthy"

    if TRAIN_FEATURE_COLUMNS:
        status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)

    return status


# ----------------------------
# Latest Predictions
# ----------------------------

@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    
    # 1. check folder exists
    if not PRED_DIR.exists():
        raise HTTPException(status_code=404, detail="Predictions folder not found")

    # 2. get files sorted by last modified time (correct way)
    files = list(PRED_DIR.glob("preds_*.csv"))

    if not files:
        raise HTTPException(status_code=404, detail="No predictions found")

    latest_file = max(files, key=lambda x: x.stat().st_mtime)

    # 3. read only needed rows (performance optimization)
    try:
        df = pd.read_csv(latest_file, nrows=limit)
    except Exception as e:
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
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    df = pd.DataFrame(data)

    # Since data is already feature engineered, predict directly
    from joblib import load
    model = load(MODEL_PATH)

    # Align columns to training features
    if TRAIN_FEATURE_COLUMNS:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    preds = model.predict(df)
    result = {"predictions": preds.tolist()}

    # If actual prices are provided
    if "price" in df.columns:
        actuals = df["price"].tolist()
        result["actuals"] = actuals

    return result