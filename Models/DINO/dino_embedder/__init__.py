"""Standalone DINO image embedding package."""

from .embedder import DinoEmbedder
from .model_registry import DINO_MODEL_IDS, SUPPORTED_MODELS

__all__ = ["DinoEmbedder", "DINO_MODEL_IDS", "SUPPORTED_MODELS"]

