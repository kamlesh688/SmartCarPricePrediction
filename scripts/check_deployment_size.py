import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE_PATHS = [
    ROOT / "dataset" / "raw",
    ROOT / "dataset" / "processed",
    ROOT / "reports",
    ROOT / "tests",
    ROOT / ".env",
]


def get_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def should_include(path: Path) -> bool:
    return not any(str(path).startswith(str(ignore)) for ignore in IGNORE_PATHS)


def estimate_size() -> None:
    total = 0
    print("Deployment size summary:")
    for root, dirs, files in os.walk(ROOT):
        root_path = Path(root)
        if any(root_path == ignore or root_path.is_relative_to(ignore) for ignore in IGNORE_PATHS):
            continue
        for file in files:
            path = root_path / file
            if path.name in [".gitignore", ".vercelignore"]:
                continue
            size = get_size(path)
            total += size
            if path.suffix in [".joblib", ".json"] or path.parent.name in ["css", "js", "images", "templates"]:
                print(f" - {path.relative_to(ROOT)}: {size / 1024:.1f} KB")
    print(f"Total estimated included size: {total / 1024:.1f} KB")


if __name__ == "__main__":
    estimate_size()
