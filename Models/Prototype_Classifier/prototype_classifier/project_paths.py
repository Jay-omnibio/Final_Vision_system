"""Path helpers for importing local model modules."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DINO_ROOT = PROJECT_ROOT / "Models" / "DINO"


def ensure_dino_import_path() -> None:
    path_text = str(DINO_ROOT)
    if DINO_ROOT.is_dir() and path_text not in sys.path:
        sys.path.insert(0, path_text)
