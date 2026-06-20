import json
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

from config import ARTIFACTS_DIR, MODELS_DIR, MODEL_REGISTRY_PATH, PRICE_CONVERSION_FACTOR
from services.model_service import load_model_registry, load_model_artifact
from services.data_service import normalize_owner_category


def _select_model_key(selected_model: str, registry: dict[str, Any]) -> str:
    if selected_model == "best":
        return registry.get("best_model", {}).get("model_key", "random_forest")
    return selected_model


def _construct_input_dataframe(data: dict[str, Any], feature_order: list[str]) -> pd.DataFrame:
    year = int(data["year"])
    mileage_km = int(data["mileage_km"])
    engine_size_cc = int(data["engine_size_cc"])
    owner_count = int(data["owner_count"])
    car_age = max(datetime.now().year - year, 0)

    features = {
        **data,
        "year": year,
        "car_age": car_age,
        "mileage_km": mileage_km,
        "mileage_per_year": mileage_km / max(car_age, 1),
        "engine_size_cc": engine_size_cc,
        "engine_size_liters": engine_size_cc / 1000.0,
        "owner_count": owner_count,
        "owner_category": normalize_owner_category(owner_count),
    }
    missing = [key for key in feature_order if key not in features]
    if missing:
        raise ValueError(f"Cannot construct model features: {', '.join(missing)}")
    return pd.DataFrame([{key: features[key] for key in feature_order}])


def run_prediction(raw_data: dict[str, Any], selected_model: str) -> dict[str, Any]:
    registry = load_model_registry()
    model_key = _select_model_key(selected_model, registry)
    model_metadata = next((m for m in registry.get("models", []) if m.get("model_key") == model_key), None)
    if not model_metadata:
        raise ValueError("Selected model is not available.")

    model = load_model_artifact(model_key)
    feature_order = model_metadata.get("feature_order", [])
    if not feature_order:
        raise ValueError("Model feature order metadata is missing.")

    input_df = _construct_input_dataframe(raw_data, feature_order)
    base_prediction = float(model.predict(input_df)[0])
    prediction = base_prediction * PRICE_CONVERSION_FACTOR

    interval_metadata = registry.get("prediction_interval", {}).get(model_key, {})
    residual_quantile = float(interval_metadata.get("residual_quantile", 0.0)) * PRICE_CONVERSION_FACTOR
    lower_price = max(0.0, prediction - residual_quantile)
    upper_price = prediction + residual_quantile

    price_categories = load_price_categories()
    category = categorize_price(base_prediction, price_categories)

    return {
        "predicted_price": prediction,
        "lower_price": lower_price,
        "upper_price": upper_price,
        "model_key": model_key,
        "model_name": model_metadata.get("display_name", "Unknown Model"),
        "residual_quantile": residual_quantile,
        "price_category": category,
        "valuation_date": datetime.now().strftime("%B %Y"),
        "vehicle_summary": {
            "brand": raw_data["brand"],
            "model": raw_data["model"],
            "year": raw_data["year"],
            "fuel_type": raw_data["fuel_type"],
            "transmission": raw_data["transmission"],
            "mileage_km": raw_data["mileage_km"],
            "engine_size_cc": raw_data["engine_size_cc"],
            "owner_count": raw_data["owner_count"],
        },
    }


def load_price_categories() -> dict[str, Any]:
    path = Path(MODEL_REGISTRY_PATH).resolve().parent.parent / "artifacts" / "price_categories.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def categorize_price(price: float, categories: dict[str, Any]) -> str:
    thresholds = categories.get("thresholds", {})
    if not thresholds:
        return "Uncategorized"
    ordered = sorted(thresholds.items(), key=lambda item: int(item[0].split("_")[-1]))
    category = "Uncategorized"
    for threshold_key, threshold in ordered:
        if price >= threshold:
            label_key = threshold_key.replace("threshold_", "label_")
            category = categories.get("labels", {}).get(label_key, "Uncategorized")
    return category
