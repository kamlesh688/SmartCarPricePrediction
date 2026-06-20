import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("SECRET_KEY", "replace-with-a-long-random-value")
FLASK_ENV = os.getenv("FLASK_ENV", "production")
PRICE_CURRENCY = os.getenv("PRICE_CURRENCY", "INR")
PRICE_SYMBOL = os.getenv("PRICE_SYMBOL", "₹")
PRICE_CONVERSION_FACTOR = float(os.getenv("PRICE_CONVERSION_FACTOR", "85.0"))

RAW_DATASET_PATH = BASE_DIR / "dataset" / "raw" / "car_price_dataset.csv"
PROCESSED_DATASET_PATH = BASE_DIR / "dataset" / "processed" / "car_price_clean.csv"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.json"
MODEL_METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
DATASET_INSIGHTS_PATH = ARTIFACTS_DIR / "dataset_insights.json"
FEATURE_METADATA_PATH = ARTIFACTS_DIR / "feature_metadata.json"
PRICE_CATEGORIES_PATH = ARTIFACTS_DIR / "price_categories.json"
PREDICTION_INTERVAL_PATH = ARTIFACTS_DIR / "prediction_interval_metadata.json"

DATA_CLEANING_SUMMARY_PATH = ARTIFACTS_DIR / "data_cleaning_summary.json"
TRAINING_METADATA_PATH = ARTIFACTS_DIR / "training_metadata.json"
OUTLIER_ANALYSIS_PATH = ARTIFACTS_DIR / "outlier_analysis.json"
CHART_DATA_PATH = ARTIFACTS_DIR / "chart_data.json"
