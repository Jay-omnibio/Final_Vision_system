#!/usr/bin/env python
"""Embed a folder of images and save paths plus vectors as .npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_embedder import DinoEmbedder
from dino_embedder.image_utils import iter_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument("--model", default="dinov2-small", choices=["dinov2-small", "dinov3"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="outputs/folder_embeddings.npz")
    parser.add_argument("--no-recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(iter_image_paths(args.folder, recursive=not args.no_recursive))
    if not paths:
        raise SystemExit(f"No images found in {args.folder}")

    embedder = DinoEmbedder(model_name=args.model, device=args.device)
    embeddings = embedder.embed_images(paths)
    relpaths = np.array([str(path.relative_to(args.folder)) for path in paths])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=embeddings,
        paths=relpaths,
        model=np.array(args.model),
        model_id=np.array(embedder.model_id),
    )
    print(f"saved {output} images={len(paths)} shape={embeddings.shape} model={args.model}")


if __name__ == "__main__":
    main()

