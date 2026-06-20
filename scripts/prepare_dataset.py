import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.data_service import prepare_dataset


if __name__ == "__main__":
    try:
        df_clean = prepare_dataset()
        print(f"Processed dataset saved successfully with {len(df_clean)} usable records.")
    except Exception as exc:
        print(f"Dataset preparation failed: {exc}")
        sys.exit(1)
