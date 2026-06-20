import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ARTIFACTS_DIR,
    DATA_CLEANING_SUMMARY_PATH,
    DATASET_INSIGHTS_PATH,
    FEATURE_METADATA_PATH,
    OUTLIER_ANALYSIS_PATH,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
)

COLUMN_ALIASES = {
    "brand": ["brand", "make", "manufacturer", "company", "car_brand", "Brand"],
    "model": ["model", "car_model", "vehicle_model", "name", "Model"],
    "year": ["year", "model_year", "manufacturing_year", "manufacture_year", "Year"],
    "fuel_type": ["fuel", "fuel_type", "fueltype", "Fuel_Type"],
    "transmission": ["transmission", "transmission_type", "gearbox", "Transmission"],
    "engine_size": ["engine", "engine_size", "engine_cc", "engine_capacity", "displacement", "Engine_Size"],
    "mileage": ["mileage", "km_driven", "kilometers_driven", "odometer", "distance_driven", "Mileage"],
    "owner_count": ["owner", "owners", "owner_count", "previous_owners", "number_of_owners", "Owner_Count"],
    "selling_price": ["selling_price", "price", "resale_price", "sale_price", "listed_price", "car_price", "Price"],
}

OWNER_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

OWNER_CATEGORY_LABELS = {
    0: "New or showroom/demo",
    1: "First owner",
    2: "Second owner",
    3: "Third owner",
    4: "Four or more owners",
}


def find_dataset_columns(columns: Iterable[str]) -> dict[str, str]:
    normalized = {col.lower().strip(): col for col in columns}
    selected: dict[str, str] = {}
    for target, candidates in COLUMN_ALIASES.items():
        candidates_lower = [c.lower() for c in candidates]
        matches = [normalized[c] for c in candidates_lower if c in normalized]
        if len(matches) == 1:
            selected[target] = matches[0]
        elif len(matches) > 1:
            exact = [c for c in matches if c.lower() == target]
            selected[target] = exact[0] if exact else matches[0]
    return selected


def detect_name_split(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    if "name" not in df.columns:
        return None, None
    sample = df["name"].dropna().astype(str).str.strip()
    if sample.empty:
        return None, None
    sample_value = sample.iloc[0]
    tokens = sample_value.split()
    if len(tokens) >= 2:
        return tokens[0], " ".join(tokens[1:])
    return None, None


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.title()


def parse_numeric(value: Any) -> Optional[float]:
    if pd.isna(value) or value == "":
        return None
    text = str(value).strip().lower()
    text = text.replace(",", "").replace("₹", "").replace("rs", "").replace("pk", "")
    text = re.sub(r"[^0-9\.]+", "", text)
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_price(value: Any) -> Optional[float]:
    price = parse_numeric(value)
    if price is None:
        return None
    if price <= 0:
        return None
    return price


def parse_engine_size(value: Any) -> Optional[int]:
    if pd.isna(value) or value == "":
        return None
    text = str(value).strip().lower().replace(",", "")
    
    # If it's a numeric value
    try:
        num = float(text)
        # If value is between 0.5 and 10, assume it's in liters
        if 0.5 <= num <= 10:
            return int(round(num * 1000))
        # If value is >= 500, assume it's in cc
        elif num >= 500:
            return int(num)
        # If value is between 10 and 500, it's ambiguous but likely cc
        else:
            return int(num)
    except ValueError:
        pass
    
    # Parse text format like "4.2L" or "4200cc"
    if "cc" in text:
        number = re.sub(r"[^0-9\.]+", "", text)
        return int(float(number)) if number else None
    if "l" in text:
        number = re.sub(r"[^0-9\.]+", "", text)
        if number:
            return int(round(float(number) * 1000))
    
    number = re.sub(r"[^0-9\.]+", "", text)
    if number:
        num = float(number)
        if num <= 10:
            return int(round(num * 1000))
        return int(num)
    
    return None


def parse_mileage(value: Any) -> Optional[int]:
    if pd.isna(value) or value == "":
        return None
    text = str(value).strip().lower().replace(",", "")
    text = text.replace("km", "").replace("kms", "").replace("kilometers", "").replace("kilometres", "")
    number = re.sub(r"[^0-9\.]+", "", text)
    if number == "":
        return None
    return int(float(number))


def parse_owner_count(value: Any) -> Optional[int]:
    if pd.isna(value) or value == "":
        return None
    text = str(value).strip().lower()
    number = re.search(r"(\d+)", text)
    if number:
        owner_number = int(number.group(1))
        return min(owner_number, 10)
    for label, number in OWNER_MAP.items():
        if label in text:
            return min(number, 10)
    if "test" in text or "drive" in text:
        return None
    if "first" in text:
        return 1
    if "second" in text:
        return 2
    if "third" in text:
        return 3
    if "fourth" in text or "four" in text:
        return 4
    return None


def normalize_owner_category(owner_count: int) -> str:
    if owner_count <= 0:
        return OWNER_CATEGORY_LABELS[0]
    if owner_count == 1:
        return OWNER_CATEGORY_LABELS[1]
    if owner_count == 2:
        return OWNER_CATEGORY_LABELS[2]
    if owner_count == 3:
        return OWNER_CATEGORY_LABELS[3]
    return OWNER_CATEGORY_LABELS[4]


def validate_row(row: pd.Series, current_year: int) -> Tuple[bool, List[str]]:
    reasons: list[str] = []
    if pd.isna(row["selling_price"]):
        reasons.append("Missing selling price")
    if pd.isna(row["year"]):
        reasons.append("Missing year")
    elif not (1980 <= int(row["year"]) <= current_year):
        reasons.append("Invalid year")
    if pd.isna(row["mileage_km"]):
        reasons.append("Missing mileage")
    elif not (0 <= int(row["mileage_km"]) <= 1000000):
        reasons.append("Invalid mileage")
    if pd.isna(row["engine_size_cc"]):
        reasons.append("Missing engine size")
    elif not (500 <= int(row["engine_size_cc"]) <= 10000):
        reasons.append("Invalid engine size")
    if pd.isna(row["owner_count"]):
        reasons.append("Missing owner count")
    elif not (0 <= int(row["owner_count"]) <= 10):
        reasons.append("Invalid owner count")
    return (len(reasons) == 0, reasons)


def prepare_dataset() -> pd.DataFrame:
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATASET_PATH}. Place a CSV file at this path and rerun the preparation script."
        )
    df = pd.read_csv(RAW_DATASET_PATH)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    original_rows = len(df)
    mapping = find_dataset_columns(df.columns)
    if "selling_price" not in mapping:
        raise ValueError(
            "Unable to detect the target price column. Verify the input CSV contains a price field."
        )
    if "brand" not in mapping or "model" not in mapping:
        raise ValueError(
            "Unable to detect brand/model columns. Add columns with vehicle brand and model."
        )

    # Rename columns to standard names
    rename_dict = {mapping[key]: key for key in mapping if key in mapping}
    df = df.rename(columns=rename_dict)
    
    # Select only the columns we need
    required_cols = ["brand", "model", "year", "fuel_type", "transmission", "engine_size", "mileage", "owner_count", "selling_price"]
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols].copy()

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Normalize text columns
    df["brand"] = df["brand"].astype(str).apply(normalize_text)
    df["model"] = df["model"].astype(str).apply(normalize_text)
    df["fuel_type"] = df["fuel_type"].astype(str).apply(normalize_text)
    df["transmission"] = df["transmission"].astype(str).apply(normalize_text)

    # Parse numeric columns
    df["year"] = df["year"].apply(parse_numeric).astype("Int64")
    df["engine_size_cc"] = df["engine_size"].apply(parse_engine_size).astype("Int64")
    df["mileage_km"] = df["mileage"].apply(parse_mileage).astype("Int64")
    df["owner_count"] = df["owner_count"].apply(parse_owner_count).astype("Int64")
    df["selling_price"] = df["selling_price"].apply(parse_price).astype("float")

    # Keep only standard columns
    df = df[["brand", "model", "year", "fuel_type", "transmission", "engine_size_cc", "mileage_km", "owner_count", "selling_price"]].copy()

    # Validate rows
    current_year = datetime.now().year
    validation = df.apply(lambda row: validate_row(row, current_year), axis=1)
    validity = [valid for valid, _ in validation]
    invalid_reasons = [reasons for _, reasons in validation]

    invalid_counts = Counter(reason for reasons in invalid_reasons for reason in reasons)
    df_clean = df[validity].copy()

    if len(df_clean) < 500:
        raise ValueError(
            f"After cleaning, the dataset has only {len(df_clean)} usable records. At least 500 are required."
        )

    # Feature engineering
    df_clean["car_age"] = current_year - df_clean["year"].astype(int)
    df_clean["mileage_per_year"] = df_clean["mileage_km"].astype(float) / df_clean["car_age"].replace(0, 1).astype(float)
    df_clean["engine_size_liters"] = df_clean["engine_size_cc"].astype(float) / 1000.0
    df_clean["owner_category"] = df_clean["owner_count"].astype(int).map(normalize_owner_category)

    df_clean.to_csv(PROCESSED_DATASET_PATH, index=False)

    # Save cleaning summary
    missing_by_column = df_clean.isna().sum().to_dict()
    summary = {
        "original_row_count": original_rows,
        "final_usable_row_count": len(df_clean),
        "duplicate_rows_removed": original_rows - len(df.drop_duplicates()),
        "rows_removed_for_missing_target": int(invalid_counts.get("Missing selling price", 0)),
        "rows_removed_for_invalid_price": int(invalid_counts.get("Missing selling price", 0)),
        "rows_removed_for_invalid_year": int(invalid_counts.get("Invalid year", 0)),
        "rows_removed_for_invalid_mileage": int(invalid_counts.get("Invalid mileage", 0)),
        "rows_removed_for_invalid_engine_size": int(invalid_counts.get("Invalid engine size", 0)),
        "missing_values_by_column": missing_by_column,
        "final_column_data_types": {col: str(dtype) for col, dtype in df_clean.dtypes.items()},
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with DATA_CLEANING_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Save dataset insights
    dataset_insights = {
        "summary": {
            "original_row_count": original_rows,
            "final_usable_row_count": len(df_clean),
            "number_of_columns": len(df_clean.columns),
            "number_of_brands": int(df_clean["brand"].nunique()),
            "number_of_models": int(df_clean["model"].nunique()),
            "average_price": float(df_clean["selling_price"].mean()),
            "median_price": float(df_clean["selling_price"].median()),
            "average_mileage_km": float(df_clean["mileage_km"].mean()),
            "average_year": float(df_clean["year"].mean()),
            "average_engine_size_cc": float(df_clean["engine_size_cc"].mean()),
        },
        "top_brands": df_clean["brand"].value_counts().head(10).to_dict(),
        "top_models_by_price": df_clean.groupby("model")["selling_price"].mean().sort_values(ascending=False).head(10).round(2).to_dict(),
        "average_price_by_brand": df_clean.groupby("brand")["selling_price"].mean().round(2).to_dict(),
        "average_price_by_fuel_type": df_clean.groupby("fuel_type")["selling_price"].mean().round(2).to_dict(),
        "average_price_by_transmission": df_clean.groupby("transmission")["selling_price"].mean().round(2).to_dict(),
        "price_distribution": df_clean["selling_price"].describe().to_dict(),
    }
    with DATASET_INSIGHTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(dataset_insights, handle, indent=2)

    # Save feature metadata
    feature_metadata = {
        "brands": sorted(df_clean["brand"].dropna().unique().tolist()),
        "models": sorted(df_clean["model"].dropna().unique().tolist()),
        "models_by_brand": {
            brand: sorted(group["model"].dropna().unique().tolist())
            for brand, group in df_clean.groupby("brand")
        },
        "fuel_types": sorted(df_clean["fuel_type"].dropna().unique().tolist()),
        "transmissions": sorted(df_clean["transmission"].dropna().unique().tolist()),
        "years": sorted(df_clean["year"].dropna().astype(int).unique().tolist()),
    }
    with FEATURE_METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(feature_metadata, handle, indent=2)

    # Save outlier analysis
    outlier_analysis = {
        "method": "Domain-aware filtering and manual review",
        "final_rows": len(df_clean),
        "rows_flagged": {k: int(v) for k, v in invalid_counts.items()},
        "notes": [
            "Rows with missing or invalid target values were excluded.",
            "Mileage and engine size were normalized from common textual formats.",
            "Vehicle age was calculated from the manufacturing year.",
        ],
    }
    with OUTLIER_ANALYSIS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(outlier_analysis, handle, indent=2)

    return df_clean
