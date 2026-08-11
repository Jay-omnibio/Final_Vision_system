#!/usr/bin/env python
"""Headless live vision runtime that prints object-passed events."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISION_ROOT = PROJECT_ROOT / "Vision_Pipeline"
CLASSIFIER_ROOT = PROJECT_ROOT / "Models" / "Prototype_Classifier"
for path in (PROJECT_ROOT, VISION_ROOT, CLASSIFIER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Runtime.live_sources import create_frame_source  # noqa: E402
from Runtime.object_store import ObjectEventStore  # noqa: E402
from pipeline_core import ObjectPassingConfig, ObjectPassingDetector, VisionPipeline, VisionPipelineConfig  # noqa: E402
from prototype_classifier import PrototypeClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames-dir", default=None, help="Optional folder source for local testing")
    parser.add_argument("--repeat-frames", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jsonl", default=None, help="Optional path to write one JSON event per line")
    parser.add_argument("--no-store", action="store_true", help="Disable operator event/crop storage")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--detector", default=None, choices=["yolo", "subtract"])
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classifier-device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--classify-every", type=int, default=1000000)
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--min-crop-size", type=int, default=24)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_overrides(config: VisionPipelineConfig, args: argparse.Namespace) -> VisionPipelineConfig:
    if args.detector is not None:
        config.detector_type = args.detector
    if args.conf is not None:
        config.yolo_conf = args.conf
    if args.device is not None:
        config.yolo_device = args.device
    return config


def resolve_path(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def crop_track(frame, track, padding: int):
    height, width = frame.shape[:2]
    x, y, box_width, box_height = track.box
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)
    return frame[y1:y2, x1:x2]


def update_track_labels(frame, result, classifier, labels: dict[int, dict], args: argparse.Namespace) -> None:
    classify_every = max(1, args.classify_every)
    for track in result.tracks:
        if track.track_id in labels and result.frame_index % classify_every != 0:
            continue
        crop = crop_track(frame, track, args.crop_padding)
        if crop.shape[0] < args.min_crop_size or crop.shape[1] < args.min_crop_size:
            continue
        prediction = classifier.classify_image(crop, array_format="bgr")
        labels[track.track_id] = {
            "label": prediction.label,
            "score": prediction.score,
            "class_name": prediction.class_name,
            "subclass_name": prediction.subclass_name,
        }


def create_event_detector(config: VisionPipelineConfig) -> ObjectPassingDetector:
    return ObjectPassingDetector(
        ObjectPassingConfig(
            axis=config.event_axis,
            line_ratio=config.event_line_ratio,
            direction=config.event_direction,
            trigger_position=config.event_trigger_position,
            min_track_age=config.event_min_track_age,
            unknown_label=config.event_unknown_label,
        )
    )


def print_event(event: dict, *, print_json: bool) -> None:
    if print_json:
        print(json.dumps(event), flush=True)
        return
    print(
        f"object_passed | ID {event['track_id']} | {event['label']} | "
        f"score {event['score']:.3f} | frame {event['frame_index']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    raw_config = load_yaml(args.config)
    pipeline_config = apply_overrides(VisionPipelineConfig.from_yaml(args.config), args)
    pipeline = VisionPipeline(pipeline_config)
    event_detector = create_event_detector(pipeline_config)

    classifier_config = raw_config.get("classifier", {}) or {}
    classifier = PrototypeClassifier(
        gallery_path=resolve_path(classifier_config.get("gallery_path")),
        dino_model=str(classifier_config.get("dino_model", "dinov2-small")),
        device=args.classifier_device or str(classifier_config.get("device", "auto")),
    )

    source = create_frame_source(
        raw_config.get("camera", {}) or {},
        frames_dir=args.frames_dir,
        repeat=args.repeat_frames,
    )
    store = None
    if not args.no_store:
        store = ObjectEventStore.from_config(raw_config.get("operator_store", {}) or {}, project_root=PROJECT_ROOT)
    jsonl_handle = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("a", encoding="utf-8")

    track_labels: dict[int, dict] = {}
    frame_count = 0
    print("Live vision runtime started.", flush=True)
    try:
        for frame in source.frames():
            frame_count += 1
            result = pipeline.process_frame(frame)
            update_track_labels(frame, result, classifier, track_labels, args)
            events = event_detector.update(
                frame_index=result.frame_index,
                frame_shape=frame.shape,
                tracks=result.tracks,
                track_labels=track_labels,
            )
            for event in events:
                payload = event.to_dict()
                payload["timestamp"] = time.time()
                if store is not None:
                    payload = store.record_event(payload, frame)
                print_event(payload, print_json=args.print_json)
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(payload) + "\n")
                    jsonl_handle.flush()

            if args.limit is not None and frame_count >= args.limit:
                break
    finally:
        source.close()
        if jsonl_handle is not None:
            jsonl_handle.close()


if __name__ == "__main__":
    main()
