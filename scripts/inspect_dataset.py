import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RAW_DATASET_PATH


def inspect_dataset() -> None:
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATASET_PATH}. Place the CSV file in dataset/raw/ and rerun."
        )
    df = pd.read_csv(RAW_DATASET_PATH)
    print(f"Dataset path: {RAW_DATASET_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Columns:")
    for column in df.columns:
        print(f" - {column}")
    print("\nSample data:")
    print(df.head(5).to_string(index=False))
    print("\nMissing values by column:")
    print(df.isna().sum())
    print("\nColumn types:")
    print(df.dtypes)


if __name__ == "__main__":
    inspect_dataset()
