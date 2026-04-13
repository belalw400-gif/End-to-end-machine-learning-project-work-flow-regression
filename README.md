# End-to-End Machine Learning Project - Housing Price Prediction

A complete machine learning pipeline for predicting housing prices using XGBoost, featuring data preprocessing, feature engineering, hyperparameter tuning with Optuna, MLflow experiment tracking, and a REST API with Streamlit UI.

## ✨ Features

- **Data Pipeline**: Automated data splitting, cleaning, and preprocessing
- **Feature Engineering**: Automated feature extraction, encoding, and leakage prevention
- **Model Training**: Baseline XGBoost model with configurable parameters
- **Hyperparameter Tuning**: Optuna-based optimization with MLflow tracking
- **Model Evaluation**: Comprehensive metrics (MAE, RMSE, R²)
- **REST API**: FastAPI endpoint for real-time predictions
- **Web UI**: Streamlit dashboard for interactive exploration
- **Logging**: Centralized logging for debugging and monitoring
- **Data Quality**: Built-in validation and quality checks
- **Testing**: Comprehensive unit tests for all modules

## 🏗️ Project Structure

```
├── src/
│   ├── api/                      # FastAPI application
│   │   └── main.py
│   ├── config.py                 # Configuration management
│   ├── logger.py                 # Centralized logging
│   ├── feature_pipeline/         # Data processing
│   │   ├── load.py              # Data loading & splitting
│   │   ├── preprocess.py        # Cleaning & normalization
│   │   └── feature_engineering.py # Feature extraction
│   ├── training_pipeline/        # Model training
│   │   ├── train.py             # Baseline training
│   │   ├── eval.py              # Model evaluation
│   │   └── tune.py              # Hyperparameter tuning
│   └── inference_pipeline/       # Prediction engine
│       └── inference.py          # Inference logic
├── data/
│   ├── raw/                     # Original data
│   ├── processed/               # Cleaned & engineered data
│   └── predictions/             # Cached predictions
├── models/                       # Trained models & encoders
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── app.py                        # Streamlit web interface
├── params.yaml                   # Configuration parameters
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/download the project
cd End-to-end-machine-learning-project-work-flow

# Install uv (if not already installed)
pip install uv

# Create virtual environment with uv (optional but recommended)
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies using uv
uv pip install -r requirements.txt

# Install the project in development mode
uv pip install -e .
```

### 2. Data Pipeline

```bash
# 1. Load and split raw data
python -c "from src.feature_pipeline.load import load_and_split_data; load_and_split_data()"

# 2. Preprocess data (clean, normalize, handle outliers)
python -c "from src.feature_pipeline.preprocess import run_preprocess; run_preprocess()"

# 3. Feature engineering (encodings, date features, leakage prevention)
python -c "from src.feature_pipeline.feature_engineering import run_feature_engineering; run_feature_engineering()"
```

### 3. Train Model

```bash
# Train baseline model
python -c "from src.training_pipeline.train import train_model; train_model()"

# Evaluate model
python -c "from src.training_pipeline.eval import evaluate_model; evaluate_model()"

# Hyperparameter tuning (Optuna + MLflow)
python -c "from src.training_pipeline.tune import tune_model; tune_model()"
```

### 4. Run API

```bash
# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Test health endpoint
curl http://localhost:8000/health

# Make predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"feature1": 1.0, "feature2": 2.0}]'
```

### 5. Launch Web UI

```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_features.py -v

# Run with coverage
uv run pytest tests/ --cov=src
```

## ⚙️ Configuration

Configuration is managed through:

1. **Environment Variables** (`.env` file):
   ```
   MODEL_PATH=models/xgb_best_model.pkl
   FREQ_ENCODER_PATH=models/freq_encoder.pkl
   TARGET_ENCODER_PATH=models/target_encoder.pkl
   MLFLOW_TRACKING_URI=sqlite:///mlruns.db
   MLFLOW_EXPERIMENT_NAME=housing_regression
   API_HOST=0.0.0.0
   API_PORT=8000
   ```

2. **YAML Parameters** (`params.yaml`):
   - Data split dates
   - Model hyperparameters
   - Tuning ranges for Optuna
   - Feature engineering settings

3. **Python Config** (`src/config.py`):
   - Centralized path management
   - Automatic directory creation
   - Configuration validation

## 📊 Monitoring & Tracking

### Logging

- Console output for real-time information
- Debug logs saved to `logs/` directory
- Structured logging with timestamps

### MLflow

MLflow tracks all experiments and hyperparameter tuning:

```bash
# View MLflow dashboard
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Then navigate to `http://localhost:5000`

## 🔍 Data Quality Checks

Built-in data quality validation in `tests/data_quality.py`:

```python
from tests.data_quality import DataQualityChecker

checker = DataQualityChecker(df, "Dataset")
summary = checker.run_all_checks()
```

Checks include:
- Missing values
- Data type validation
- Outlier detection
- Duplicate row detection
- Price column validation
- Date column validation
- Feature completeness

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health status & model info |
| `/latest_predictions` | GET | Get cached predictions |
| `/predict` | POST | Generate new predictions |

## 🐛 Troubleshooting

### Model Not Found
```
Error: Model not found at models/xgb_best_model.pkl
Solution: Train the model first using src.training_pipeline.train module
```

### Encoders Missing
```
Warning: Frequency encoder not found
Solution: Run feature engineering pipeline to generate encoders
```

### Data Files Missing
```
Error: Raw data not found
Solution: Place data in data/raw/ with files: train.csv, eval.csv, holdout.csv
```

### Python Version Issues
```
Error: Type hint syntax not supported
Solution: Ensure Python >= 3.10 (for PEP 604 union syntax)
```

## 📝 Workflows

### Complete Workflow

1. **Load & Split Data**
   ```python
   from src.feature_pipeline.load import load_and_split_data
   load_and_split_data()
   ```

2. **Preprocess**
   ```python
   from src.feature_pipeline.preprocess import run_preprocess
   run_preprocess()
   ```

3. **Feature Engineering**
   ```python
   from src.feature_pipeline.feature_engineering import run_feature_engineering
   run_feature_engineering()
   ```

4. **Train & Evaluate**
   ```python
   from src.training_pipeline.train import train_model
   from src.training_pipeline.eval import evaluate_model
   model, metrics = train_model()
   metrics = evaluate_model()
   ```

5. **Hyperparameter Tuning**
   ```python
   from src.training_pipeline.tune import tune_model
   best_params, best_metrics = tune_model(n_trials=15)
   ```

6. **Inference**
   ```python
   from src.inference_pipeline.inference import predict
   predictions = predict(new_data_df)
   ```

## 🤝 Contributing

Contributions are welcome! Please:
1. Follow PEP 8 style guide
2. Add tests for new features
3. Update documentation
4. Use type hints in function signatures

## 📜 License

This project is provided as-is for educational and research purposes.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files in `logs/` directory
3. Check test files for usage examples
4. Review docstrings in source code

---

**Last Updated**: 2026-04-11
**Python Version**: >= 3.10
**Status**: Production-ready with comprehensive error handling and logging
