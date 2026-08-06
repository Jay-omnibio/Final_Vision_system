"""Path helpers for importing local model modules."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "Models"

MODEL_IMPORT_PATHS = [
    MODELS_ROOT / "YOLO_Detector",
    MODELS_ROOT / "Subtract_Detector",
    MODELS_ROOT / "Object_Tracker",
]


def ensure_model_import_paths() -> None:
    for path in MODEL_IMPORT_PATHS:
        path_text = str(path)
        if path.is_dir() and path_text not in sys.path:
            sys.path.insert(0, path_text)

