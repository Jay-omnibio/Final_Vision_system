"""DINOv2-small and DINOv3 embedding backends."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .image_utils import load_image
from .model_registry import model_family, resolve_model_id

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def _resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _preprocess_dinov2(image: Image.Image, image_size: int):
    import torch

    resized = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    for channel in range(3):
        array[:, :, channel] = (array[:, :, channel] - _IMAGENET_MEAN[channel]) / _IMAGENET_STD[channel]
    return torch.from_numpy(array.transpose(2, 0, 1))


class DinoEmbedder:
    """Create L2-normalized DINO image embeddings.

    Parameters
    ----------
    model_name:
        Either ``dinov2-small`` or ``dinov3``.
    device:
        ``auto``, ``cpu``, or ``cuda``. ``auto`` selects CUDA when available.
    model_id:
        Optional Hugging Face model ID override, useful for local mirrors.
    image_size:
        Manual preprocessing size for DINOv2-small. DINOv3 uses its processor.
    """

    def __init__(
        self,
        model_name: str = "dinov2-small",
        *,
        device: str = "auto",
        model_id: str | None = None,
        image_size: int = 224,
    ) -> None:
        import torch
        from transformers import AutoModel

        self.model_name = model_name
        self.model_id = resolve_model_id(model_name, model_id)
        self.family = model_family(model_name)
        self.device = _resolve_device(device)
        self.image_size = image_size
        self._torch = torch
        self._processor = None

        if self.family == "dinov3":
            from transformers import AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(self.model_id)

        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.eval()
        self._model.to(self.device)

    def embed_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        array_format: str = "rgb",
    ) -> np.ndarray:
        """Embed one image path/PIL/NumPy input into a 1D float32 vector."""
        pil_image = load_image(image, array_format=array_format)
        return self.embed_pil(pil_image)

    def embed_pil(self, image: Image.Image) -> np.ndarray:
        """Embed one PIL image into a 1D float32 vector."""
        return self.embed_batch_pil([image])[0]

    def embed_images(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        *,
        array_format: str = "rgb",
    ) -> np.ndarray:
        """Embed many image inputs into an NxD float32 matrix."""
        pil_images = [load_image(image, array_format=array_format) for image in images]
        return self.embed_batch_pil(pil_images)

    def embed_batch_pil(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed many PIL images into an NxD float32 matrix."""
        if not images:
            raise ValueError("images must contain at least one image")

        if self.family == "dinov3":
            return self._embed_dinov3(images)
        return self._embed_dinov2(images)

    def _embed_dinov2(self, images: Sequence[Image.Image]) -> np.ndarray:
        tensors = [_preprocess_dinov2(image, self.image_size) for image in images]
        pixel_values = self._torch.stack(tensors, dim=0).to(self.device)
        with self._torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)
        vectors = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        return np.stack([_l2_normalize(vector) for vector in vectors], axis=0)

    def _embed_dinov3(self, images: Sequence[Image.Image]) -> np.ndarray:
        if self._processor is None:
            raise RuntimeError("DINOv3 processor was not initialized")
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self._processor(images=rgb_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        if getattr(outputs, "pooler_output", None) is not None:
            vectors = outputs.pooler_output.float().cpu().numpy()
        else:
            vectors = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
        return np.stack([_l2_normalize(vector) for vector in vectors], axis=0)

