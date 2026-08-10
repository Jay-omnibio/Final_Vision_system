#!/usr/bin/env python
"""Run the vision pipeline on saved frames and write an annotated video."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_ROOT = PROJECT_ROOT / "Models" / "Prototype_Classifier"
if str(CLASSIFIER_ROOT) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_ROOT))

from pipeline_core import ObjectPassingConfig, ObjectPassingDetector, VisionPipeline, VisionPipelineConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRACK_COLORS = [
    (0, 220, 255),
    (255, 180, 0),
    (0, 180, 80),
    (220, 80, 255),
    (255, 80, 80),
    (80, 160, 255),
]


def iter_images(folder: str | Path):
    root = Path(folder)
    for path in sorted(root.rglob("*"), key=natural_sort_key):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


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
    parser.add_argument("--classify", action="store_true", help="Classify tracked crops with prototype classifier")
    parser.add_argument("--gallery", default=None, help="Prototype gallery path for classification")
    parser.add_argument("--dino-model", default="dinov2-small", choices=["dinov2-small", "dinov3"])
    parser.add_argument("--classifier-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--classify-every", type=int, default=8, help="Reclassify each track every N frames")
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--min-crop-size", type=int, default=24)
    parser.add_argument("--events", action="store_true", help="Emit object_passed events when tracks cross the configured line")
    parser.add_argument("--line-ratio", type=float, default=None, help="Override event trigger line ratio")
    parser.add_argument("--line-axis", default=None, choices=["x", "y"], help="Override event axis")
    parser.add_argument("--line-direction", default=None, choices=["positive", "negative"], help="Override event crossing direction")
    parser.add_argument(
        "--trigger-position",
        default=None,
        choices=["centroid", "leading_edge", "trailing_edge"],
        help="Track position used for crossing: centroid, leading edge, or trailing edge",
    )
    parser.add_argument("--draw-event-line", action="store_true", help="Draw the event trigger line on the output video")
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


def create_event_detector(config: VisionPipelineConfig, args: argparse.Namespace):
    if not args.events and not config.event_enabled:
        return None
    event_config = ObjectPassingConfig(
        axis=args.line_axis or config.event_axis,
        line_ratio=args.line_ratio if args.line_ratio is not None else config.event_line_ratio,
        direction=args.line_direction or config.event_direction,
        trigger_position=args.trigger_position or config.event_trigger_position,
        min_track_age=config.event_min_track_age,
        unknown_label=config.event_unknown_label,
    )
    return ObjectPassingDetector(event_config)


def crop_track(frame, track, padding: int):
    height, width = frame.shape[:2]
    x, y, box_width, box_height = track.box
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)
    return frame[y1:y2, x1:x2]


def update_track_labels(frame, result, classifier, labels: dict[int, dict], args: argparse.Namespace) -> None:
    if classifier is None:
        return
    classify_every = max(1, args.classify_every)
    for track in result.tracks:
        if track.track_id in labels and result.frame_index % classify_every != 0:
            continue
        crop = crop_track(frame, track, args.crop_padding)
        if crop.shape[0] < args.min_crop_size or crop.shape[1] < args.min_crop_size:
            continue
        try:
            prediction = classifier.classify_image(crop, array_format="bgr")
        except Exception as exc:
            labels[track.track_id] = {"label": "classify_error", "score": 0.0, "error": str(exc)}
            continue
        labels[track.track_id] = {
            "label": prediction.label,
            "score": prediction.score,
            "class_name": prediction.class_name,
            "subclass_name": prediction.subclass_name,
        }


def draw_result(
    frame,
    result,
    track_labels: dict[int, dict] | None = None,
    event_detector: ObjectPassingDetector | None = None,
    events: list | None = None,
    draw_event_line: bool = False,
):
    output = frame.copy()
    track_labels = track_labels or {}
    events = events or []

    if draw_event_line and event_detector is not None:
        line_position = event_detector.line_position(output.shape)
        if event_detector.config.axis == "y":
            cv2.line(output, (0, line_position), (output.shape[1], line_position), (0, 0, 255), 2)
            label_position = (12, max(24, line_position - 10))
        else:
            cv2.line(output, (line_position, 0), (line_position, output.shape[0]), (0, 0, 255), 2)
            label_position = (min(output.shape[1] - 180, line_position + 8), 28)
        cv2.putText(
            output,
            "PASS LINE",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for track in result.tracks:
        x, y, width, height = track.box
        x2 = x + width
        y2 = y + height
        color = TRACK_COLORS[(track.track_id - 1) % len(TRACK_COLORS)]
        cv2.rectangle(output, (x, y), (x2, y2), color, 2)
        label = f"ID {track.track_id}"
        if track.track_id in track_labels:
            prediction = track_labels[track.track_id]
            label = f"{label} {prediction['label']} {prediction['score']:.2f}"
        cv2.putText(
            output,
            label,
            (x, min(output.shape[0] - 8, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

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

    for index, event in enumerate(events[:4]):
        cv2.putText(
            output,
            f"PASSED ID {event.track_id}: {event.label}",
            (12, 60 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
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
    event_detector = create_event_detector(config, args)
    classifier = None
    track_labels: dict[int, dict] = {}
    if args.classify:
        from prototype_classifier import PrototypeClassifier

        classifier = PrototypeClassifier(
            gallery_path=args.gallery,
            dino_model=args.dino_model,
            device=args.classifier_device,
        )

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
            update_track_labels(frame, result, classifier, track_labels, args)
            frame_events = []
            if event_detector is not None:
                frame_events = event_detector.update(
                    frame_index=result.frame_index,
                    frame_shape=frame.shape,
                    tracks=result.tracks,
                    track_labels=track_labels,
                )
            annotated = draw_result(
                frame,
                result,
                track_labels,
                event_detector,
                frame_events,
                args.draw_event_line,
            )

            if writer is None:
                output_path, writer = create_writer(args.output, annotated.shape, args.fps)

            writer.write(annotated)
            payload = result.to_dict()
            payload["image"] = str(path)
            if track_labels:
                payload["track_labels"] = dict(track_labels)
            if frame_events:
                payload["events"] = [event.to_dict() for event in frame_events]
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
