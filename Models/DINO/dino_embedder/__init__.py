"""Standalone DINO image embedding package."""

from .embedder import DinoEmbedder
from .factory import create_dino_embedder
from .model_registry import DINO_MODEL_IDS, SUPPORTED_MODELS
from .onnx_embedder import OnnxDinoEmbedder

__all__ = ["DinoEmbedder", "DINO_MODEL_IDS", "OnnxDinoEmbedder", "SUPPORTED_MODELS", "create_dino_embedder"]
