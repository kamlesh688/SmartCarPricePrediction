import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import MODEL_REGISTRY_PATH, MODELS_DIR


class ModelLoadingError(Exception):
    pass


@lru_cache(maxsize=1)
def load_model_registry() -> dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        raise ModelLoadingError(
            "Model registry not found. Train the models locally using python train_models.py and commit the generated artifacts."
        )
    with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_best_model() -> dict[str, Any]:
    registry = load_model_registry()
    best = registry.get("best_model")
    if not best:
        raise ModelLoadingError("Best model metadata is missing from the registry.")
    return best


@lru_cache(maxsize=2)
def load_model_artifact(model_key: str):
    registry = load_model_registry()
    for model in registry.get("models", []):
        if model.get("model_key") == model_key:
            artifact_path = MODELS_DIR / model.get("artifact_filename", "")
            if not artifact_path.exists():
                raise ModelLoadingError(
                    f"Model artifact {artifact_path} is missing. Train models locally and commit the artifact."
                )
            import joblib

            return joblib.load(artifact_path)
    raise ModelLoadingError(f"Model key {model_key} not found in model registry.")
