#!/usr/bin/env python
"""Run the vision pipeline on saved frames and write an annotated video."""

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
    parser.add_argument("frames_dir", help="Folder containing saved frames")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config YAML path")
    parser.add_argument("--output", default="outputs/pipeline_output.mp4")
    parser.add_argument("--json", default=None, help="Optional JSON result path")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--detector", default=None, choices=["yolo", "subtract"])
    parser.add_argument("--background", default=None, help="Override subtract background image")
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def apply_overrides(config: VisionPipelineConfig, args: argparse.Namespace) -> VisionPipelineConfig:
    if args.detector is not None:
        config.detector_type = args.detector
    if args.background is not None:
        config.background_image = args.background
    if args.yolo_weights is not None:
        config.yolo_weights = args.yolo_weights
    if args.conf is not None:
        config.yolo_conf = args.conf
    if args.iou is not None:
        config.yolo_iou = args.iou
    if args.device is not None:
        config.yolo_device = args.device
    return config


def draw_result(frame, result):
    output = frame.copy()

    for detection in result.detections:
        x1, y1, x2, y2 = detection.xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 80), 2)
        label = detection.class_name or "det"
        cv2.putText(
            output,
            f"{label} {detection.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )

    for track in result.tracks:
        x, y, width, height = track.box
        x2 = x + width
        y2 = y + height
        cv2.rectangle(output, (x, y), (x2, y2), (0, 200, 255), 2)
        cv2.circle(output, track.centroid, 4, (0, 200, 255), -1)
        cv2.putText(
            output,
            f"ID {track.track_id}",
            (x, min(output.shape[0] - 8, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        for start, end in zip(track.history, track.history[1:]):
            cv2.line(output, start, end, (255, 180, 0), 2)

    cv2.putText(
        output,
        f"frame {result.frame_index}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def create_writer(path: str | Path, frame_shape, fps: float):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    return output_path, writer


def main() -> None:
    args = parse_args()
    paths = list(iter_images(args.frames_dir))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No frames found in {args.frames_dir}")

    config = apply_overrides(VisionPipelineConfig.from_yaml(args.config), args)
    pipeline = VisionPipeline(config)

    writer = None
    output_path = None
    results = []

    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"Skipping unreadable frame: {path}")
                continue

            result = pipeline.process_frame(frame)
            annotated = draw_result(frame, result)

            if writer is None:
                output_path, writer = create_writer(args.output, annotated.shape, args.fps)

            writer.write(annotated)
            payload = result.to_dict()
            payload["image"] = str(path)
            results.append(payload)
            print(
                f"[{result.frame_index}] {path.name}: "
                f"detections={len(result.detections)} tracks={len(result.tracks)}"
            )
    finally:
        if writer is not None:
            writer.release()

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if output_path is not None:
        print(f"Wrote video: {output_path}")


if __name__ == "__main__":
    main()

