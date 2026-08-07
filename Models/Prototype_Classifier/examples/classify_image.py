#!/usr/bin/env python
"""Classify one image with DINO and an existing prototype gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype_classifier import PrototypeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image or crop path to classify")
    parser.add_argument("--gallery", default=None, help="Path to .npz prototype gallery")
    parser.add_argument("--dino-model", default="dinov2-small", choices=["dinov2-small", "dinov3"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--json", default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = PrototypeClassifier(
        gallery_path=args.gallery,
        dino_model=args.dino_model,
        device=args.device,
    )
    result = classifier.classify_image(args.image)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

