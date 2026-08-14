#!/usr/bin/env python
"""Headless live vision runtime that prints object-passed events."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
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

TRACK_COLORS = [
    (0, 220, 255),
    (255, 180, 0),
    (0, 180, 80),
    (220, 80, 255),
    (255, 80, 80),
    (80, 160, 255),
]


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
    parser.add_argument("--classify-every", type=int, default=1, help="Sample/classify each visible track every N track frames")
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--min-crop-size", type=int, default=24)
    parser.add_argument("--debug-video", action="store_true", help="Save annotated debug video until runtime stops")
    parser.add_argument("--debug-video-path", default=None, help="Optional debug MP4 output path")
    parser.add_argument("--debug-fps", type=float, default=10.0)
    parser.add_argument("--stop-file", default=None, help="Exit cleanly when this file appears")
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


def l2_normalize(embedding) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def should_sample(track, args: argparse.Namespace) -> bool:
    if track.missing_frames > 0:
        return False
    if track.width < args.min_crop_size or track.height < args.min_crop_size:
        return False
    return track.age % max(1, args.classify_every) == 0


def update_track_labels(frame, result, classifier, labels: dict[int, dict], track_states: dict[int, dict], args: argparse.Namespace) -> None:
    for track in result.tracks:
        state = track_states.setdefault(track.track_id, {"embeddings": []})
        if not should_sample(track, args):
            continue
        crop = crop_track(frame, track, args.crop_padding)
        if crop.size == 0 or crop.shape[0] < args.min_crop_size or crop.shape[1] < args.min_crop_size:
            continue
        state["embeddings"].append(classifier.embedder.embed_image(crop, array_format="bgr"))
        mean_embedding = l2_normalize(np.mean(np.stack(state["embeddings"]), axis=0))
        prediction = classifier.classify_embedding(mean_embedding)
        labels[track.track_id] = {
            "label": prediction.label,
            "score": prediction.score,
            "class_name": prediction.class_name,
            "subclass_name": prediction.subclass_name,
            "embedding_count": len(state["embeddings"]),
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


class DebugVideoWriter:
    def __init__(self, path: str | Path | None, *, fps: float) -> None:
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = PROJECT_ROOT / "outputs" / f"live_debug_{timestamp}.mp4"
        self.path = resolve_path(path) or PROJECT_ROOT / "outputs" / "live_debug.mp4"
        self.fps = fps
        self.writer = None

    def write(self, frame) -> None:
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open debug video writer: {self.path}")
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Debug video written: {self.path}", flush=True)


def draw_debug_frame(frame, result, track_labels: dict[int, dict], events: list, event_detector) -> any:
    output = frame.copy()
    line_position = event_detector.line_position(output.shape)
    if event_detector.config.axis == "y":
        cv2.line(output, (0, line_position), (output.shape[1], line_position), (0, 0, 255), 2)
    else:
        cv2.line(output, (line_position, 0), (line_position, output.shape[0]), (0, 0, 255), 2)

    for track in result.tracks:
        x, y, width, height = track.box
        color = TRACK_COLORS[(track.track_id - 1) % len(TRACK_COLORS)]
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        label = f"ID {track.track_id}"
        if track.track_id in track_labels:
            prediction = track_labels[track.track_id]
            label = f"{label} {prediction['label']} {prediction['score']:.2f}"
        cv2.putText(
            output,
            label,
            (x, min(output.shape[0] - 8, y + height + 20)),
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
        dino_backend=str(classifier_config.get("dino_backend", "torch")),
        dino_onnx_path=resolve_path(classifier_config.get("dino_onnx_path")),
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
    track_states: dict[int, dict] = {}
    frame_count = 0
    debug_writer = DebugVideoWriter(args.debug_video_path, fps=args.debug_fps) if args.debug_video else None
    stop_file = resolve_path(args.stop_file)
    if stop_file is not None and stop_file.exists():
        stop_file.unlink()
    print("Live vision runtime started.", flush=True)
    try:
        for frame in source.frames():
            if stop_file is not None and stop_file.exists():
                print("Live vision runtime stopped by stop file.", flush=True)
                break
            frame_count += 1
            result = pipeline.process_frame(frame)
            update_track_labels(frame, result, classifier, track_labels, track_states, args)
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

            if debug_writer is not None:
                debug_writer.write(draw_debug_frame(frame, result, track_labels, events, event_detector))

            if args.limit is not None and frame_count >= args.limit:
                break
            if stop_file is not None and stop_file.exists():
                print("Live vision runtime stopped by stop file.", flush=True)
                break
    except KeyboardInterrupt:
        print("Live vision runtime stopped by user.", flush=True)
    finally:
        source.close()
        if debug_writer is not None:
            debug_writer.close()
        if jsonl_handle is not None:
            jsonl_handle.close()
        if stop_file is not None and stop_file.exists():
            stop_file.unlink()


if __name__ == "__main__":
    main()
