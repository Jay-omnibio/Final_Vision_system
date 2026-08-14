"""Factory helpers for selecting a DINO embedding backend."""

from __future__ import annotations

from pathlib import Path

from .embedder import DinoEmbedder
from .onnx_embedder import OnnxDinoEmbedder


def create_dino_embedder(
    *,
    backend: str = "torch",
    model_name: str = "dinov2-small",
    device: str = "auto",
    onnx_path: str | Path | None = None,
    model_id: str | None = None,
):
    """Create a DINO embedder from runtime/config settings."""
    backend_name = str(backend or "torch").lower()
    if backend_name in {"torch", "pytorch"}:
        return DinoEmbedder(model_name=model_name, device=device, model_id=model_id)
    if backend_name == "onnx":
        if model_name != "dinov2-small":
            raise ValueError("ONNX DINO backend currently supports only dinov2-small")
        if onnx_path in (None, ""):
            raise ValueError("onnx_path is required when DINO backend is 'onnx'")
        return OnnxDinoEmbedder(onnx_path, device=device)
    raise ValueError("DINO backend must be 'torch' or 'onnx'")
