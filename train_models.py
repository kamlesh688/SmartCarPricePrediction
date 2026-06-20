import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from config import (
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    DATASET_INSIGHTS_PATH,
    FEATURE_METADATA_PATH,
    MODEL_REGISTRY_PATH,
    MODELS_DIR,
    PREDICTION_INTERVAL_PATH,
    PRICE_CURRENCY,
    PRICE_CONVERSION_FACTOR,
    PRICE_SYMBOL,
    RAW_DATASET_PATH,
    TRAINING_METADATA_PATH,
    DATA_CLEANING_SUMMARY_PATH,
    PROCESSED_DATASET_PATH,
)
from services.data_service import prepare_dataset


RANDOM_STATE = 42
MODEL_CONFIGS = {
    "linear_regression": {
        "display_name": "Linear Regression",
        "artifact_filename": "linear_regression.joblib",
    },
    "random_forest": {
        "display_name": "Random Forest Regressor",
        "artifact_filename": "random_forest.joblib",
    },
}

NUMERICAL_FEATURES = ["year", "car_age", "mileage_km", "mileage_per_year", "engine_size_cc", "engine_size_liters", "owner_count"]
CATEGORICAL_FEATURES = ["brand", "model", "fuel_type", "transmission", "owner_category"]


def load_clean_dataset() -> pd.DataFrame:
    if not PROCESSED_DATASET_PATH.exists():
        prepare_dataset()
    return pd.read_csv(PROCESSED_DATASET_PATH)


def build_pipelines() -> dict[str, Pipeline]:
    num_transform = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_transform = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", fill_value="Unknown")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([
        ("num", num_transform, NUMERICAL_FEATURES),
        ("cat", cat_transform, CATEGORICAL_FEATURES),
    ])

    linear_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    forest_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    return {"linear_regression": linear_pipeline, "random_forest": forest_pipeline}


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "r2_score": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mape": np.mean(np.abs((y_test - predictions) / np.where(y_test == 0, 1, y_test))) * 100,
        "median_ae": median_absolute_error(y_test, predictions),
    }


def save_report_chart(fig, name: str):
    path = Path("reports") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_price_categories(y_train: pd.Series) -> dict[str, Any]:
    quantiles = y_train.quantile([0.2, 0.4, 0.6, 0.8]).to_list()
    labels = {
        "label_1": "Budget Range",
        "label_2": "Economy Range",
        "label_3": "Mid Range",
        "label_4": "Premium Range",
        "label_5": "Luxury Range",
    }
    thresholds = {
        "threshold_1": 0,
        "threshold_2": float(quantiles[0]),
        "threshold_3": float(quantiles[1]),
        "threshold_4": float(quantiles[2]),
        "threshold_5": float(quantiles[3]),
    }
    return {"currency": PRICE_CURRENCY, "thresholds": thresholds, "labels": labels}


def select_best_model(metrics: dict[str, Any]) -> dict[str, Any]:
    best_key = None
    best_metrics = None
    for model_key, values in metrics.items():
        if best_key is None:
            best_key = model_key
            best_metrics = values
            continue
        if values["rmse"] < best_metrics["rmse"]:
            best_key = model_key
            best_metrics = values
        elif values["rmse"] == best_metrics["rmse"]:
            if values["mae"] < best_metrics["mae"]:
                best_key = model_key
                best_metrics = values
            elif values["mae"] == best_metrics["mae"] and values["r2_score"] > best_metrics["r2_score"]:
                best_key = model_key
                best_metrics = values
    return {"model_key": best_key, **best_metrics} if best_key else {}


def generate_plot_data(df: pd.DataFrame, y_test: pd.Series, y_pred_lr: np.ndarray, y_pred_rf: np.ndarray):
    insights = {
        "price_distribution": df["selling_price"].describe().to_dict(),
        "brand_price": df.groupby("brand")["selling_price"].mean().sort_values(ascending=False).head(10).to_dict(),
    }
    with Path(ARTIFACTS_DIR / "dataset_insights.json").open("w", encoding="utf-8") as handle:
        json.dump(insights, handle, indent=2)


if __name__ == "__main__":
    try:
        df = load_clean_dataset()
        y = df["selling_price"].astype(float)
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()

        y_binned = pd.qcut(y, q=10, duplicates="drop")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binned
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )

        pipelines = build_pipelines()
        model_metrics = {}
        prediction_intervals = {}
        models_metadata = []

        for model_key, pipeline in pipelines.items():
            start_train = time.time()
            pipeline.fit(X_train, y_train)
            train_duration = time.time() - start_train
            start_pred = time.time()
            preds = pipeline.predict(X_test)
            inference_duration = (time.time() - start_pred) / len(X_test)
            metrics = evaluate_model(pipeline, X_test, y_test)
            artifact_path = MODELS_DIR / MODEL_CONFIGS[model_key]["artifact_filename"]
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, artifact_path, compress=3)

            residuals = np.abs(y_train - pipeline.predict(X_train))
            residual_quantile = float(np.quantile(residuals, 0.9))

            model_metrics[model_key] = {
                "display_name": MODEL_CONFIGS[model_key]["display_name"],
                **metrics,
                "training_time_seconds": train_duration,
                "inference_time_seconds": inference_duration,
                "artifact_size_bytes": artifact_path.stat().st_size,
                "residual_quantile": residual_quantile,
            }
            prediction_intervals[model_key] = {"residual_quantile": residual_quantile}
            models_metadata.append(
                {
                    "model_key": model_key,
                    "display_name": MODEL_CONFIGS[model_key]["display_name"],
                    "artifact_filename": MODEL_CONFIGS[model_key]["artifact_filename"],
                    "feature_order": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
                    "training_date": datetime.utcnow().isoformat(),
                    "sklearn_version": joblib.__version__,
                    "python_version": sys.version,
                    "r2_score": metrics["r2_score"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "artifact_size_bytes": artifact_path.stat().st_size,
                    "target_transformed": False,
                    "prediction_interval_residual_quantile": residual_quantile,
                    "currency": PRICE_CURRENCY,
                }
            )

        best_model = select_best_model(model_metrics)
        price_categories = build_price_categories(y_train)

        registry = {
            "models": models_metadata,
            "best_model": {
                "model_key": best_model["model_key"],
                "display_name": MODEL_CONFIGS[best_model["model_key"]]["display_name"],
                "rmse": best_model["rmse"],
                "mae": best_model["mae"],
                "r2_score": best_model["r2_score"],
                "selected_at": datetime.utcnow().isoformat(),
            },
            "prediction_interval": prediction_intervals,
        }
        with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(registry, handle, indent=2)
        with Path(ARTIFACTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(model_metrics, handle, indent=2)
        with Path(ARTIFACTS_DIR / "best_model.json").open("w", encoding="utf-8") as handle:
            json.dump(best_model, handle, indent=2)
        with Path(ARTIFACTS_DIR / "training_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump({"trained_at": datetime.utcnow().isoformat(), "row_count": len(y_train) + len(y_test)}, handle, indent=2)
        with Path(ARTIFACTS_DIR / "prediction_interval_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(prediction_intervals, handle, indent=2)
        with Path(ARTIFACTS_DIR / "price_categories.json").open("w", encoding="utf-8") as handle:
            json.dump(price_categories, handle, indent=2)

        print("Training completed.")
        print(json.dumps(best_model, indent=2))
    except Exception as exc:
        print(f"Training failed: {exc}")
        sys.exit(1)
