#!/usr/bin/env python
"""Compare PyTorch and ONNX DINOv2-small embeddings on the same images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_embedder import DinoEmbedder, OnnxDinoEmbedder
from dino_embedder.image_utils import iter_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Image path or folder")
    parser.add_argument("--onnx", default="Models/DINO/onnx/dinov2-small.onnx")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--torch-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--onnx-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--details", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def collect_images(path: str | Path, *, recursive: bool, limit: int) -> list[Path]:
    value = Path(path)
    if value.is_file():
        return [value]
    images = list(iter_image_paths(value, recursive=recursive))
    if limit > 0:
        images = images[:limit]
    return images


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    images = collect_images(input_path, recursive=not args.no_recursive, limit=args.limit)
    if not images:
        raise SystemExit(f"No images found: {input_path}")

    onnx_path = resolve_project_path(args.onnx)
    torch_embedder = DinoEmbedder(model_name="dinov2-small", device=args.torch_device)
    onnx_embedder = OnnxDinoEmbedder(onnx_path, device=args.onnx_device)

    torch_embeddings = torch_embedder.embed_images(images)
    onnx_embeddings = onnx_embedder.embed_images(images)
    cosine = np.sum(torch_embeddings * onnx_embeddings, axis=1)

    if args.details:
        for path, score in zip(images, cosine):
            print(f"{score:.6f}  {path}")

    print(
        "cosine_similarity "
        f"count={len(images)} min={float(np.min(cosine)):.6f} "
        f"mean={float(np.mean(cosine)):.6f} max={float(np.max(cosine)):.6f}"
    )


if __name__ == "__main__":
    main()
