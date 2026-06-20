import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR, MODEL_REGISTRY_PATH, MODELS_DIR, PRICE_CURRENCY


def verify_artifacts() -> None:
    if not MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError("Model registry is missing.")
    with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    required_artifacts = [
        ARTIFACTS_DIR / "dataset_insights.json",
        ARTIFACTS_DIR / "feature_metadata.json",
        ARTIFACTS_DIR / "price_categories.json",
        ARTIFACTS_DIR / "prediction_interval_metadata.json",
        ARTIFACTS_DIR / "model_metrics.json",
        ARTIFACTS_DIR / "best_model.json",
    ]
    for artifact in required_artifacts:
        if not artifact.exists():
            raise FileNotFoundError(f"Artifact missing: {artifact}")

    for model in registry.get("models", []):
        artifact_path = MODELS_DIR / model["artifact_filename"]
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact missing: {artifact_path}")

    print("Model registry and artifacts are present.")
    price_categories = ARTIFACTS_DIR / "price_categories.json"
    with price_categories.open("r", encoding="utf-8") as handle:
        categories = json.load(handle)
    if categories.get("currency") != PRICE_CURRENCY:
        raise ValueError("Price category currency mismatch.")

    sample_model = registry["models"][0]
    artifact_path = MODELS_DIR / sample_model["artifact_filename"]
    import joblib

    model = joblib.load(artifact_path)
    print(f"Loaded {sample_model['display_name']} from {artifact_path.name}.")

    print("All required artifact checks passed.")


if __name__ == "__main__":
    try:
        verify_artifacts()
    except Exception as exc:
        print(f"Artifact verification failed: {exc}")
        sys.exit(1)
