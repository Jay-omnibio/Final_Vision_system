#!/usr/bin/env python
"""Headless live novelty runtime that prints known/new object-passed events."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISION_ROOT = PROJECT_ROOT / "Vision_Pipeline"
DINO_ROOT = PROJECT_ROOT / "Models" / "DINO"
NOVELTY_ROOT = PROJECT_ROOT / "Models" / "Novelty_Detector"
for path in (PROJECT_ROOT, VISION_ROOT, DINO_ROOT, NOVELTY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Runtime.live_sources import create_frame_source  # noqa: E402
from Runtime.object_store import ObjectEventStore  # noqa: E402
from dino_embedder import create_dino_embedder  # noqa: E402
from novelty_detector import l2_normalize, load_novelty_runtime  # noqa: E402
from pipeline_core import ObjectPassingConfig, ObjectPassingDetector, VisionPipeline, VisionPipelineConfig  # noqa: E402

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
    parser.add_argument("--dino-model", default=None, choices=["dinov2-small", "dinov3"])
    parser.add_argument("--dino-device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dino-backend", default=None, choices=["torch", "pytorch", "onnx"])
    parser.add_argument("--dino-onnx-path", default=None)
    parser.add_argument("--gallery", default=None)
    parser.add_argument("--known-embeddings", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--groups-output", default="Models/Novelty_Detector/artifacts/groups/live_runtime_groups.npz")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--min-embeddings", type=int, default=1)
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--min-crop-size", type=int, default=24)
    parser.add_argument("--debug-video", action="store_true")
    parser.add_argument("--debug-video-path", default=None)
    parser.add_argument("--debug-fps", type=float, default=10.0)
    parser.add_argument("--stop-file", default=None)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def apply_overrides(config: VisionPipelineConfig, args: argparse.Namespace) -> VisionPipelineConfig:
    if args.detector is not None:
        config.detector_type = args.detector
    if args.conf is not None:
        config.yolo_conf = args.conf
    if args.device is not None:
        config.yolo_device = args.device
    return config


def create_runtime_embedder(raw_config: dict, args: argparse.Namespace):
    classifier_config = raw_config.get("classifier", {}) or {}
    backend = args.dino_backend or str(classifier_config.get("dino_backend", "torch"))
    model_name = args.dino_model or str(classifier_config.get("dino_model", "dinov2-small"))
    device = args.dino_device or str(classifier_config.get("device", "auto"))
    onnx_path = args.dino_onnx_path or classifier_config.get("dino_onnx_path")
    return create_dino_embedder(
        backend=backend,
        model_name=model_name,
        device=device,
        onnx_path=resolve_path(onnx_path),
    )


def novelty_paths(raw_config: dict, args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    novelty_config = raw_config.get("novelty", {}) or {}
    gallery = args.gallery or novelty_config.get(
        "gallery_path",
        "Models/Novelty_Detector/artifacts/prototypes/gallery_known_hierarchical.npz",
    )
    known_embeddings = args.known_embeddings or novelty_config.get(
        "known_embeddings_path",
        "Models/Novelty_Detector/artifacts/embeddings/known.npz",
    )
    calibration = args.calibration or novelty_config.get(
        "calibration_path",
        "Models/Novelty_Detector/artifacts/calibration/novelty_mahalanobis.npz",
    )
    return resolve_path(gallery), resolve_path(known_embeddings), resolve_path(calibration)


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


def crop_track(frame, track, padding: int):
    height, width = frame.shape[:2]
    x, y, box_width, box_height = track.box
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)
    return frame[y1:y2, x1:x2]


def should_sample(track, args: argparse.Namespace) -> bool:
    if track.missing_frames > 0:
        return False
    if track.width < args.min_crop_size or track.height < args.min_crop_size:
        return False
    return track.age % max(1, args.sample_every) == 0


def update_track_embeddings(frame, result, embedder, track_states: dict[int, dict[str, Any]], args: argparse.Namespace) -> None:
    for track in result.tracks:
        state = track_states.setdefault(track.track_id, {"embeddings": [], "final": None})
        if not should_sample(track, args):
            continue
        crop = crop_track(frame, track, args.crop_padding)
        if crop.size == 0:
            continue
        state["embeddings"].append(embedder.embed_image(crop, array_format="bgr"))


def finalize_events(events, track_states: dict[int, dict[str, Any]], novelty_runtime, min_embeddings: int) -> list[dict[str, Any]]:
    finalized = []
    for event in events:
        state = track_states.setdefault(event.track_id, {"embeddings": [], "final": None})
        if state["final"] is not None:
            novelty = dict(state["final"])
        elif len(state["embeddings"]) >= min_embeddings:
            mean_embedding = l2_normalize(np.mean(np.stack(state["embeddings"]), axis=0))
            novelty = novelty_runtime.process(mean_embedding, commit=True).to_dict()
            state["final"] = novelty
        else:
            novelty = {
                "status": "new",
                "final_label": "unknown",
                "reason": "not_enough_embeddings",
                "embedding_count": len(state["embeddings"]),
            }
            state["final"] = novelty

        payload = event.to_dict()
        payload["label"] = novelty.get("final_label", payload.get("label", "unknown"))
        payload["score"] = float(novelty.get("subclass_score", payload.get("score", 0.0)) or 0.0)
        payload["novelty"] = novelty
        finalized.append(payload)
    return finalized


def active_label(track_id: int, track_states: dict[int, dict[str, Any]]) -> str:
    state = track_states.get(track_id)
    if not state:
        return "collecting"
    if state.get("final"):
        novelty = state["final"]
        return f"{novelty.get('final_label', 'unknown')} ({novelty.get('status', '?')})"
    if state.get("embeddings"):
        return f"samples:{len(state['embeddings'])}"
    return "collecting"


class DebugVideoWriter:
    def __init__(self, path: str | Path | None, *, fps: float) -> None:
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = PROJECT_ROOT / "outputs" / f"live_novelty_debug_{timestamp}.mp4"
        self.path = resolve_path(path) or PROJECT_ROOT / "outputs" / "live_novelty_debug.mp4"
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


def draw_debug_frame(frame, result, event_detector, frame_events, track_states):
    output = frame.copy()
    line_position = event_detector.line_position(output.shape)
    if event_detector.config.axis == "y":
        cv2.line(output, (0, line_position), (output.shape[1], line_position), (0, 0, 255), 2)
        label_position = (12, max(24, line_position - 10))
    else:
        cv2.line(output, (line_position, 0), (line_position, output.shape[0]), (0, 0, 255), 2)
        label_position = (min(output.shape[1] - 260, line_position + 8), 28)
    cv2.putText(output, "NOVELTY PASS LINE", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    for track in result.tracks:
        x, y, width, height = track.box
        color = TRACK_COLORS[(track.track_id - 1) % len(TRACK_COLORS)]
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        cv2.putText(output, f"ID {track.track_id} {active_label(track.track_id, track_states)}", (x, min(output.shape[0] - 8, y + height + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.putText(output, f"frame {result.frame_index}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    for index, event in enumerate(frame_events[:4]):
        novelty = event.get("novelty", {})
        text = f"PASSED ID {event['track_id']}: {novelty.get('final_label')} ({novelty.get('status')})"
        cv2.putText(output, text, (12, 60 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def print_event(event: dict, *, print_json: bool) -> None:
    if print_json:
        print(json.dumps(event), flush=True)
        return
    novelty = event.get("novelty", {})
    print(
        f"object_passed | ID {event['track_id']} | {event['label']} | "
        f"{novelty.get('status', '?')} | score {event['score']:.3f} | frame {event['frame_index']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    raw_config = load_yaml(args.config)
    pipeline_config = apply_overrides(VisionPipelineConfig.from_yaml(args.config), args)
    pipeline = VisionPipeline(pipeline_config)
    event_detector = create_event_detector(pipeline_config)
    source = create_frame_source(raw_config.get("camera", {}) or {}, frames_dir=args.frames_dir, repeat=args.repeat_frames)
    embedder = create_runtime_embedder(raw_config, args)
    gallery_path, known_embeddings_path, calibration_path = novelty_paths(raw_config, args)
    novelty_runtime = load_novelty_runtime(
        gallery_path=gallery_path,
        known_embeddings_path=known_embeddings_path,
        calibration_path=calibration_path,
    )
    store = None
    if not args.no_store:
        store = ObjectEventStore.from_config(raw_config.get("operator_store", {}) or {}, project_root=PROJECT_ROOT)

    jsonl_handle = None
    if args.jsonl:
        jsonl_path = resolve_path(args.jsonl)
        assert jsonl_path is not None
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("a", encoding="utf-8")

    debug_writer = DebugVideoWriter(args.debug_video_path, fps=args.debug_fps) if args.debug_video else None
    stop_file = resolve_path(args.stop_file)
    if stop_file is not None and stop_file.exists():
        stop_file.unlink()

    track_states: dict[int, dict[str, Any]] = {}
    frame_count = 0
    print("Live novelty runtime started.", flush=True)
    try:
        for frame in source.frames():
            if stop_file is not None and stop_file.exists():
                print("Live novelty runtime stopped by stop file.", flush=True)
                break
            frame_count += 1
            result = pipeline.process_frame(frame)
            update_track_embeddings(frame, result, embedder, track_states, args)
            raw_events = event_detector.update(
                frame_index=result.frame_index,
                frame_shape=frame.shape,
                tracks=result.tracks,
                track_labels={},
            )
            frame_events = finalize_events(raw_events, track_states, novelty_runtime, args.min_embeddings)
            for event in frame_events:
                event["timestamp"] = time.time()
                if store is not None:
                    event = store.record_event(event, frame)
                print_event(event, print_json=args.print_json)
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(event) + "\n")
                    jsonl_handle.flush()

            if debug_writer is not None:
                debug_writer.write(draw_debug_frame(frame, result, event_detector, frame_events, track_states))

            if args.limit is not None and frame_count >= args.limit:
                break
            if stop_file is not None and stop_file.exists():
                print("Live novelty runtime stopped by stop file.", flush=True)
                break
    except KeyboardInterrupt:
        print("Live novelty runtime stopped by user.", flush=True)
    finally:
        source.close()
        if debug_writer is not None:
            debug_writer.close()
        if jsonl_handle is not None:
            jsonl_handle.close()
        if stop_file is not None and stop_file.exists():
            stop_file.unlink()
        groups_output = resolve_path(args.groups_output)
        assert groups_output is not None
        novelty_runtime.grouper.save(groups_output)
        print(f"Wrote novelty groups: {groups_output}", flush=True)


if __name__ == "__main__":
    main()
