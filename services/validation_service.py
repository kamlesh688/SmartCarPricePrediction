from datetime import datetime
from typing import Any

from services.formatting_service import normalize_text


class PredictionFormValidator:
    def __init__(self, feature_metadata: dict[str, Any]) -> None:
        self.feature_metadata = feature_metadata

    def validate(self, raw_data: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        brand = raw_data.get("brand", "").strip()
        model = raw_data.get("model", "").strip()
        if not brand:
            errors.append("Brand is required.")
        if not model:
            errors.append("Model is required.")

        valid_brands = set(self.feature_metadata.get("brands", []))
        if brand and brand not in valid_brands:
            errors.append("Selected brand is not recognized by the trained dataset.")

        valid_models = set(self.feature_metadata.get("models", []))
        if model and model not in valid_models:
            errors.append("Selected model is not recognized for the selected brand.")

        try:
            year = int(raw_data.get("year", 0))
            if year < 1980 or year > datetime.now().year:
                errors.append("Manufacturing year must be between 1980 and the current year.")
        except (ValueError, TypeError):
            errors.append("Manufacturing year must be a valid number.")

        try:
            mileage = int(raw_data.get("mileage_km", -1))
            if mileage < 0 or mileage > 1000000:
                errors.append("Mileage must be between 0 and 1,000,000 km.")
        except (ValueError, TypeError):
            errors.append("Mileage must be a valid integer.")

        try:
            engine_size = int(raw_data.get("engine_size_cc", -1))
            if engine_size < 500 or engine_size > 10000:
                errors.append("Engine size must be between 500 cc and 10,000 cc.")
        except (ValueError, TypeError):
            errors.append("Engine size must be a valid integer.")

        try:
            owners = int(raw_data.get("owner_count", -1))
            if owners < 0 or owners > 10:
                errors.append("Previous owners must be between 0 and 10.")
        except (ValueError, TypeError):
            errors.append("Previous owners must be a valid integer.")

        return (len(errors) == 0, errors)
