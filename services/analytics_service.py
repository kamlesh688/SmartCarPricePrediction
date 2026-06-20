import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import DATASET_INSIGHTS_PATH, FEATURE_METADATA_PATH


@lru_cache(maxsize=1)
def load_analytics_artifacts() -> dict[str, Any]:
    artifact_paths = [DATASET_INSIGHTS_PATH, FEATURE_METADATA_PATH]
    data: dict[str, Any] = {}
    for path in artifact_paths:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Required analytics artifact {path} is missing. Generate artifacts by running python train_models.py."
            )
    with DATASET_INSIGHTS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    with FEATURE_METADATA_PATH.open("r", encoding="utf-8") as handle:
        data["feature_metadata"] = json.load(handle)
    return data
