from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_FILE = MODEL_DIR / "model.pkl"
LABEL_FILE = MODEL_DIR / "label_encoder.pkl"


def ensure_directories() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_paths() -> dict:
    return {
        "dataset_dir": DATASET_DIR,
        "model_file": MODEL_FILE,
        "label_file": LABEL_FILE,
    }
