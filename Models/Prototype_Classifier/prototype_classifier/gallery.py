"""Load and query existing DINO prototype-gallery `.npz` files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import ClassificationResult, Prediction


def _as_string_list(array: np.ndarray) -> list[str]:
    return [str(item) for item in array.tolist()]


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def gallery_type_from_path(path: str | Path) -> str:
    """Return `class_only` or `hierarchical` from sidecar JSON or NPZ keys."""
    gallery_path = Path(path)
    sidecar = gallery_path.with_suffix(".json")
    if sidecar.is_file():
        with sidecar.open(encoding="utf-8") as handle:
            gallery_type = str(json.load(handle).get("gallery_type", "class_only"))
        return "class_only" if gallery_type in {"class", "class_only"} else gallery_type

    data = np.load(gallery_path, allow_pickle=True)
    if "subclass_vectors" in data.files:
        return "hierarchical"
    return "class_only"


class FlatPrototypeGallery:
    """One normalized prototype vector per class."""

    gallery_type = "class_only"

    def __init__(self, vectors: np.ndarray, class_names: list[str]) -> None:
        if vectors.shape[0] != len(class_names):
            raise ValueError("Vector count and class name count do not match")
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.class_names = list(class_names)

    @classmethod
    def load(cls, path: str | Path) -> "FlatPrototypeGallery":
        data = np.load(Path(path), allow_pickle=True)
        names = _as_string_list(data["class_names"])
        vectors_key = "vectors" if "vectors" in data.files else "class_vectors"
        return cls(data[vectors_key], names)

    def predict(self, embedding: np.ndarray) -> Prediction:
        vector = _normalize_embedding(embedding)
        scores_arr = self.vectors @ vector
        best = int(np.argmax(scores_arr))
        scores = {name: float(scores_arr[index]) for index, name in enumerate(self.class_names)}
        return Prediction(self.class_names[best], float(scores_arr[best]), scores)

    def classify(self, embedding: np.ndarray) -> ClassificationResult:
        prediction = self.predict(embedding)
        return ClassificationResult(
            label=prediction.label,
            score=prediction.score,
            gallery_type=self.gallery_type,
            class_name=prediction.label,
            class_score=prediction.score,
            class_scores=prediction.scores,
        )


class HierarchicalPrototypeGallery:
    """Class gallery plus subclass prototypes searched inside predicted class."""

    gallery_type = "hierarchical"

    def __init__(
        self,
        class_gallery: FlatPrototypeGallery,
        subclass_vectors: np.ndarray,
        subclass_names: list[str],
        subclass_class_names: list[str],
    ) -> None:
        if len(subclass_names) != len(subclass_class_names):
            raise ValueError("Subclass names and parent class names do not match")
        if subclass_vectors.shape[0] != len(subclass_names):
            raise ValueError("Subclass vector count and subclass name count do not match")

        self.class_gallery = class_gallery
        self.subclass_vectors = np.asarray(subclass_vectors, dtype=np.float32)
        self.subclass_names = list(subclass_names)
        self.subclass_class_names = list(subclass_class_names)
        self._subclass_indexes_by_class = self._build_subclass_index()

    @property
    def class_names(self) -> list[str]:
        return self.class_gallery.class_names

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalPrototypeGallery":
        data = np.load(Path(path), allow_pickle=True)
        class_gallery = FlatPrototypeGallery(
            data["class_vectors"],
            _as_string_list(data["class_names"]),
        )
        return cls(
            class_gallery,
            data["subclass_vectors"],
            _as_string_list(data["subclass_names"]),
            _as_string_list(data["subclass_class_names"]),
        )

    def _build_subclass_index(self) -> dict[str, list[int]]:
        indexes: dict[str, list[int]] = {}
        for index, class_name in enumerate(self.subclass_class_names):
            indexes.setdefault(class_name, []).append(index)
        return indexes

    def predict_subclass_in_class(self, embedding: np.ndarray, class_name: str) -> Prediction:
        vector = _normalize_embedding(embedding)
        indexes = self._subclass_indexes_by_class.get(class_name, [])
        if not indexes:
            return Prediction("unknown", float("-inf"), {})

        vectors = self.subclass_vectors[indexes]
        scores_arr = vectors @ vector
        best_local = int(np.argmax(scores_arr))
        scores = {
            self.subclass_names[indexes[index]]: float(scores_arr[index])
            for index in range(len(indexes))
        }
        return Prediction(
            self.subclass_names[indexes[best_local]],
            float(scores_arr[best_local]),
            scores,
        )

    def classify(self, embedding: np.ndarray) -> ClassificationResult:
        class_prediction = self.class_gallery.predict(embedding)
        subclass_prediction = self.predict_subclass_in_class(embedding, class_prediction.label)
        label = f"{class_prediction.label}/{subclass_prediction.label}"
        score = subclass_prediction.score
        return ClassificationResult(
            label=label,
            score=score,
            gallery_type=self.gallery_type,
            class_name=class_prediction.label,
            subclass_name=subclass_prediction.label,
            class_score=class_prediction.score,
            subclass_score=subclass_prediction.score,
            class_scores=class_prediction.scores,
            subclass_scores=subclass_prediction.scores,
        )


def load_gallery(path: str | Path) -> FlatPrototypeGallery | HierarchicalPrototypeGallery:
    gallery_type = gallery_type_from_path(path)
    if gallery_type == "hierarchical":
        return HierarchicalPrototypeGallery.load(path)
    return FlatPrototypeGallery.load(path)

