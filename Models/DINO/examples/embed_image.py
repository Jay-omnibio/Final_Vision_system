#!/usr/bin/env python
"""Embed one image and save the vector as .npy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_embedder import DinoEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image path to embed")
    parser.add_argument("--model", default="dinov2-small", choices=["dinov2-small", "dinov3"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="outputs/image_embedding.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedder = DinoEmbedder(model_name=args.model, device=args.device)
    vector = embedder.embed_image(args.image)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, vector)
    print(f"saved {output} shape={vector.shape} dtype={vector.dtype} model={args.model}")


if __name__ == "__main__":
    main()

