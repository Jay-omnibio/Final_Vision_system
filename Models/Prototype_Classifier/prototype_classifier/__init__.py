"""Standalone DINO prototype classifier package."""

from .classifier import PrototypeClassifier
from .gallery import FlatPrototypeGallery, HierarchicalPrototypeGallery, gallery_type_from_path
from .types import ClassificationResult, Prediction

__all__ = [
    "ClassificationResult",
    "FlatPrototypeGallery",
    "HierarchicalPrototypeGallery",
    "Prediction",
    "PrototypeClassifier",
    "gallery_type_from_path",
]

