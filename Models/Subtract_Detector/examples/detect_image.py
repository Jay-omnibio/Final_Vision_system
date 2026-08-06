#!/usr/bin/env python
"""Run background-subtraction detection on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subtract_detector import BackgroundSubtractionConfig, SubtractDetector
from subtract_detector.image_utils import draw_boxes, load_image_bgr, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("background", help="Empty-background image path")
    parser.add_argument("image", help="Frame image path")
    parser.add_argument("--mode", default="improved", choices=["standard", "improved"])
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--kernel-size", type=int, default=25)
    parser.add_argument("--output", default=None, help="Optional output image with boxes")
    parser.add_argument("--json", default=None, help="Optional output JSON result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BackgroundSubtractionConfig(
        threshold=args.threshold,
        min_area=args.min_area,
        kernel_size=args.kernel_size,
    )
    detector = SubtractDetector.from_background_image(args.background, mode=args.mode, config=config)
    boxes = detector.detect_image(args.image)
    payload = {"boxes": [box.to_dict() for box in boxes], "mode": args.mode}
    print(json.dumps(payload, indent=2))

    if args.output:
        frame = load_image_bgr(args.image)
        save_image(args.output, draw_boxes(frame, boxes))

    if args.json:
        output_json = Path(args.json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

