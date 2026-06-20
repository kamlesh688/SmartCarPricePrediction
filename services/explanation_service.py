from typing import Any

from services.formatting_service import normalize_text


def generate_value_observations(raw_data: dict[str, Any], analytics: dict[str, Any]) -> list[str]:
    observations: list[str] = []
    year = int(raw_data.get("year", 0))
    mileage = int(raw_data.get("mileage_km", 0))
    engine_size = int(raw_data.get("engine_size_cc", 0))
    owner_count = int(raw_data.get("owner_count", 0))
    fuel_type = raw_data.get("fuel_type", "").strip().title()
    transmission = raw_data.get("transmission", "").strip().title()
    brand = raw_data.get("brand", "").strip()

    average_year = analytics.get("summary", {}).get("average_year")
    average_mileage = analytics.get("summary", {}).get("average_mileage_km")
    average_engine = analytics.get("summary", {}).get("average_engine_size_cc")
    brand_price_map = analytics.get("average_price_by_brand", {})
    brand_avg_price = brand_price_map.get(brand)

    if average_year and year >= int(average_year):
        observations.append(
            "The vehicle is relatively recent compared with older records in the dataset."
        )
    if mileage and average_mileage and mileage > float(average_mileage):
        observations.append(
            "Higher accumulated mileage may reduce estimated resale value based on the training data."
        )
    if owner_count >= 2:
        observations.append(
            "A larger number of previous owners may reduce resale value in the training data."
        )
    if engine_size and engine_size > float(average_engine or 0):
        observations.append(
            "Larger engine size can increase estimated value in some categories, but may affect demand and running costs."
        )
    if transmission.lower() == "automatic":
        observations.append(
            "Automatic transmission influenced resale prices in the training dataset."
        )
    if fuel_type:
        observations.append(
            f"Fuel type {fuel_type} was an important feature in the dataset."
        )
    if brand_avg_price is not None:
        observations.append(
            "This brand had a comparatively high average price in the training data."
        )

    if not observations:
        observations.append(
            "These observations summarize general patterns and entered vehicle characteristics. They are not causal explanations."
        )

    return observations
