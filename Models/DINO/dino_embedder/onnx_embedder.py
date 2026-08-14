"""ONNX Runtime backend for exported DINOv2-small embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .embedder import _IMAGENET_MEAN, _IMAGENET_STD, _l2_normalize
from .image_utils import load_image


def _preprocess_image(image: Image.Image, image_size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    for channel in range(3):
        array[:, :, channel] = (array[:, :, channel] - _IMAGENET_MEAN[channel]) / _IMAGENET_STD[channel]
    return array.transpose(2, 0, 1).astype(np.float32)


def _providers(device: str) -> list[str]:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if device in {"auto", "cuda"} and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxDinoEmbedder:
    """Create L2-normalized DINOv2-small embeddings with ONNX Runtime."""

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        device: str = "auto",
        image_size: int = 224,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX DINO model not found: {self.onnx_path}")

        self.model_name = "dinov2-small"
        self.model_id = "facebook/dinov2-small"
        self.device = device
        self.image_size = image_size
        self.session = ort.InferenceSession(str(self.onnx_path), providers=_providers(device))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        array_format: str = "rgb",
    ) -> np.ndarray:
        pil_image = load_image(image, array_format=array_format)
        return self.embed_pil(pil_image)

    def embed_pil(self, image: Image.Image) -> np.ndarray:
        return self.embed_batch_pil([image])[0]

    def embed_images(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        *,
        array_format: str = "rgb",
    ) -> np.ndarray:
        pil_images = [load_image(image, array_format=array_format) for image in images]
        return self.embed_batch_pil(pil_images)

    def embed_batch_pil(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("images must contain at least one image")

        batch = np.stack([_preprocess_image(image, self.image_size) for image in images], axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: batch})[0]
        vectors = np.asarray(outputs, dtype=np.float32)
        return np.stack([_l2_normalize(vector) for vector in vectors], axis=0)
