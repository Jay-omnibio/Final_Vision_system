"""Runtime image classifier using DINO embeddings and a prototype gallery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .gallery import load_gallery
from .project_paths import ensure_dino_import_path
from .types import ClassificationResult

ensure_dino_import_path()

from dino_embedder import create_dino_embedder  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GALLERY_PATH = PACKAGE_ROOT / "galleries" / "default_gallery.npz"


class PrototypeClassifier:
    """Classify images or embeddings with an existing `.npz` prototype gallery."""

    def __init__(
        self,
        gallery_path: str | Path | None = None,
        *,
        dino_model: str = "dinov2-small",
        device: str = "auto",
        dino_model_id: str | None = None,
        dino_backend: str = "torch",
        dino_onnx_path: str | Path | None = None,
    ) -> None:
        self.gallery_path = Path(gallery_path) if gallery_path is not None else DEFAULT_GALLERY_PATH
        if not self.gallery_path.is_file():
            raise FileNotFoundError(f"Prototype gallery not found: {self.gallery_path}")

        self.gallery = load_gallery(self.gallery_path)
        self.embedder = create_dino_embedder(
            backend=dino_backend,
            model_name=dino_model,
            device=device,
            onnx_path=dino_onnx_path,
            model_id=dino_model_id,
        )

    def classify_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        array_format: str = "rgb",
    ) -> ClassificationResult:
        embedding = self.embedder.embed_image(image, array_format=array_format)
        return self.classify_embedding(embedding)

    def classify_embedding(self, embedding: np.ndarray) -> ClassificationResult:
        return self.gallery.classify(embedding)
