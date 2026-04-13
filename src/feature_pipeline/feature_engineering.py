"""
Feature engineering: date parts, frequency encoding, target encoding, drop leakage.

- Reads cleaned train/eval CSVs
- Applies feature engineering
- Saves feature-engineered CSVs
- ALSO saves fitted encoders for inference
"""
import re
from pathlib import Path
import pandas as pd
try:
    from category_encoders import TargetEncoder
except Exception:
    # lightweight fallback TargetEncoder for environments without category_encoders
    class TargetEncoder:
        def __init__(self, cols=None):
            self.cols = cols
            self.mapping = {}
            self.global_mean = 0.0

        def fit(self, X, y):
            # X may be a Series or DataFrame; support Series usage from tests
            ser = pd.Series(X) if not isinstance(X, pd.DataFrame) else pd.Series(X.iloc[:, 0])
            self.mapping = ser.groupby(ser).apply(lambda grp: float(pd.Series(y).loc[grp.index].mean())).to_dict()
            self.global_mean = float(pd.Series(y).mean())
            return self

        def transform(self, X):
            ser = pd.Series(X) if not isinstance(X, pd.DataFrame) else pd.Series(X.iloc[:, 0])
            return ser.map(self.mapping).fillna(self.global_mean)

        def fit_transform(self, X, y):
            self.fit(X, y)
            return self.transform(X)

from joblib import dump #joblib.dump saves encoders/mappings to disk (important for reusing at inference).

from src.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- feature functions ----------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    # place after date for readability (optional)
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df


#Creates a frequency encoding (how often a value appears).
#Fit only on train, then applied to eval.
def frequency_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str):
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    eval[f"{col}_freq"] = eval[col].map(freq_map).fillna(0)
    return train, eval, freq_map


#Uses target encoding (replace category with average of target variable).
#Fitted only on train (prevents leakage).
def target_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str, target: str):
    """
    Use TargetEncoder on `col`, consistently name as <col>_encoded.
    For city_full → city_full_te (keeps schema aligned with inference).
    """
    te = TargetEncoder(cols=[col])
    encoded_col = f"{col}_encoded" if col != "city_full" else "city_full_te"
    train[encoded_col] = te.fit_transform(train[col], train[target])
    eval[encoded_col] = te.transform(eval[col])
    return train, eval, te



def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame):
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    train = train.drop(columns=[c for c in drop_cols if c in train.columns], errors="ignore")
    eval = eval.drop(columns=[c for c in drop_cols if c in eval.columns], errors="ignore")
    return train, eval


# ---------- pipeline ----------

#Handles full pipeline: 
#reads cleaned CSVs → applies feature engineering → saves engineered data + encoders.
def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    """
    Run feature engineering and write outputs + encoders to disk.
    Applies the same transformations to train, eval, and holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "train_cleaned.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "eval_cleaned.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "holdout_cleaned.csv"

    # Resolve possible filename variants produced by preprocessing
    def _resolve_input(p: Path | str):
        p = Path(p)
        if p.exists():
            return p
        name = p.name
        parent = p.parent
        # try swapped naming conventions: 'X_cleaned.csv' <-> 'cleaning_X.csv'
        if "_cleaned" in name:
            split = name.split("_cleaned")[0]
            alt = parent / f"cleaning_{split}.csv"
            if alt.exists():
                return alt
        if name.startswith("cleaning_"):
            split = name.split("cleaning_")[1].rsplit(".", 1)[0]
            alt = parent / f"{split}_cleaned.csv"
            if alt.exists():
                return alt
        # fallback: try to find any similarly named csv in the parent
        for candidate in parent.glob(f"*{name.split('.')[0]}*.csv"):
            return candidate
        return p

    train_df = pd.read_csv(_resolve_input(in_train_path))
    eval_df = pd.read_csv(_resolve_input(in_eval_path))
    holdout_df = pd.read_csv(_resolve_input(in_holdout_path))

    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Eval date range:", eval_df["date"].min(), "to", eval_df["date"].max())
    print("Holdout date range:", holdout_df["date"].min(), "to", holdout_df["date"].max())

    # Date features
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    holdout_df = add_date_features(holdout_df)

    # Frequency encode zipcode (fit on train only)
    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, eval_df, freq_map = frequency_encode(train_df, eval_df, "zipcode")
        holdout_df["zipcode_freq"] = holdout_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl")   # save mapping

    # Target encode city_full (fit on train only)
    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, eval_df, target_encoder = target_encode(train_df, eval_df, "city_full", "price")
        holdout_df["city_full_te"] = target_encoder.transform(holdout_df["city_full"])
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")  # save encoder

    # Drop leakage / raw categoricals
    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    holdout_df, _ = drop_unused_columns(holdout_df.copy(), holdout_df.copy())

    # Save engineered data
    out_train_path = output_dir / "train_fe.csv"
    out_eval_path = output_dir / "eval_fe.csv"
    out_holdout_path = output_dir / "holdout_fe.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)
    holdout_df.to_csv(out_holdout_path, index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)
    print("   Holdout shape:", holdout_df.shape)
    print("   Encoders saved to models/")

    return train_df, eval_df, holdout_df, freq_map, target_encoder


if __name__ == "__main__":
    run_feature_engineering()