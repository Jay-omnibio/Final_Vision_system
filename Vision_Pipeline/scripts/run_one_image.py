#!/usr/bin/env python
"""Run the detector + tracker pipeline on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_core import VisionPipeline, VisionPipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image path")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config YAML path")
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    config = VisionPipelineConfig.from_yaml(args.config)
    if args.yolo_weights is not None:
        config.yolo_weights = args.yolo_weights
    if args.conf is not None:
        config.yolo_conf = args.conf
    if args.iou is not None:
        config.yolo_iou = args.iou
    if args.device is not None:
        config.yolo_device = args.device

    pipeline = VisionPipeline(config)
    result = pipeline.process_frame(frame)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
