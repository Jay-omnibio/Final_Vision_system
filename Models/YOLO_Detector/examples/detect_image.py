#!/usr/bin/env python
"""Run YOLO detection on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_detector import YoloDetector
from yolo_detector.image_utils import save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image path to detect")
    parser.add_argument("--weights", default=None, help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None, help="Ultralytics device, for example cpu, 0, or cuda")
    parser.add_argument("--output", default=None, help="Optional output image with boxes")
    parser.add_argument("--json", default=None, help="Optional output JSON result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YoloDetector(weights=args.weights, conf=args.conf, iou=args.iou, device=args.device)
    result = detector.detect(args.image)
    print(json.dumps(result.to_dict(), indent=2))

    if args.output:
        annotated = detector.draw_boxes(args.image, result.boxes)
        save_image(args.output, annotated)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

