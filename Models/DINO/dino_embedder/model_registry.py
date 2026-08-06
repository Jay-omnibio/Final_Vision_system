"""Supported DINO model names."""

from __future__ import annotations

DINO_MODEL_IDS = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
}

SUPPORTED_MODELS = tuple(DINO_MODEL_IDS)


def resolve_model_id(model_name: str, override_model_id: str | None = None) -> str:
    """Resolve a supported public name to its Hugging Face model ID."""
    if override_model_id:
        return override_model_id
    if model_name not in DINO_MODEL_IDS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unknown DINO model '{model_name}'. Supported models: {supported}")
    return DINO_MODEL_IDS[model_name]


def model_family(model_name: str) -> str:
    """Return the implementation family for a supported model name."""
    if model_name == "dinov2-small":
        return "dinov2"
    if model_name == "dinov3":
        return "dinov3"
    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unknown DINO model '{model_name}'. Supported models: {supported}")

