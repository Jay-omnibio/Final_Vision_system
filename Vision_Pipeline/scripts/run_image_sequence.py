#!/usr/bin/env python
"""Run the detector + tracker pipeline on a folder of images."""

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(folder: str | Path):
    root = Path(folder)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder of image frames")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config YAML path")
    parser.add_argument("--detector", default=None, choices=["yolo", "subtract"])
    parser.add_argument("--background", default=None, help="Override subtract background image")
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VisionPipelineConfig.from_yaml(args.config)
    if args.detector is not None:
        config.detector_type = args.detector
    if args.yolo_weights is not None:
        config.yolo_weights = args.yolo_weights
    if args.conf is not None:
        config.yolo_conf = args.conf
    if args.iou is not None:
        config.yolo_iou = args.iou
    if args.device is not None:
        config.yolo_device = args.device
    if args.background is not None:
        config.background_image = args.background

    pipeline = VisionPipeline(config)

    results = []
    for index, path in enumerate(iter_images(args.folder), start=1):
        if args.limit is not None and index > args.limit:
            break
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = pipeline.process_frame(frame)
        payload = result.to_dict()
        payload["image"] = str(path)
        results.append(payload)
        print(json.dumps(payload, indent=2))

    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
