#!/usr/bin/env python
"""Run optional novelty pipeline on saved frames and write an annotated video."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_ROOT = PROJECT_ROOT / "Vision_Pipeline"
DINO_ROOT = PROJECT_ROOT / "Models" / "DINO"
NOVELTY_ROOT = PROJECT_ROOT / "Models" / "Novelty_Detector"
for path in (VISION_ROOT, DINO_ROOT, NOVELTY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dino_embedder import DinoEmbedder  # noqa: E402
from novelty_detector import l2_normalize, load_novelty_runtime  # noqa: E402
from pipeline_core import ObjectPassingConfig, ObjectPassingDetector, VisionPipeline, VisionPipelineConfig  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRACK_COLORS = [
    (0, 220, 255),
    (255, 180, 0),
    (0, 180, 80),
    (220, 80, 255),
    (255, 80, 80),
    (80, 160, 255),
]


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def iter_images(folder: str | Path):
    root = Path(folder)
    for path in sorted(root.rglob("*"), key=natural_sort_key):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="outputs/novelty_pipeline_output.mp4")
    parser.add_argument("--json", default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the selected frame sequence while keeping novelty group memory")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dino-model", default="dinov2-small", choices=["dinov2-small", "dinov3"])
    parser.add_argument("--dino-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gallery", default="Models/Novelty_Detector/artifacts/prototypes/gallery_known_hierarchical.npz")
    parser.add_argument("--known-embeddings", default="Models/Novelty_Detector/artifacts/embeddings/known.npz")
    parser.add_argument("--calibration", default="Models/Novelty_Detector/artifacts/calibration/novelty_mahalanobis.npz")
    parser.add_argument("--groups-output", default="Models/Novelty_Detector/artifacts/groups/runtime_groups.npz")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--min-embeddings", type=int, default=1)
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--min-crop-size", type=int, default=24)
    parser.add_argument("--line-ratio", type=float, default=None)
    parser.add_argument("--draw-event-line", action="store_true")
    return parser.parse_args()


def apply_overrides(config: VisionPipelineConfig, args: argparse.Namespace) -> VisionPipelineConfig:
    config.yolo_conf = args.conf
    config.yolo_iou = args.iou
    config.yolo_device = args.device
    return config


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


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
        state = track_states.setdefault(track.track_id, {"embeddings": [], "previews": [], "final": None})
        if not should_sample(track, args):
            continue
        crop = crop_track(frame, track, args.crop_padding)
        if crop.size == 0:
            continue
        embedding = embedder.embed_image(crop, array_format="bgr")
        state["embeddings"].append(embedding)


def finalize_events(events, track_states: dict[int, dict[str, Any]], novelty_runtime, min_embeddings: int) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for event in events:
        state = track_states.setdefault(event.track_id, {"embeddings": [], "previews": [], "final": None})
        if state["final"] is not None:
            final = dict(state["final"])
        elif len(state["embeddings"]) >= min_embeddings:
            mean_embedding = l2_normalize(np.mean(np.stack(state["embeddings"]), axis=0))
            final = novelty_runtime.process(mean_embedding, commit=True).to_dict()
            state["final"] = final
        else:
            final = {
                "status": "new",
                "final_label": "unknown",
                "reason": "not_enough_embeddings",
                "embedding_count": len(state["embeddings"]),
            }
            state["final"] = final

        final_event = event.to_dict()
        final_event["novelty"] = final
        final_event["label"] = final.get("final_label", final_event.get("label", "unknown"))
        finalized.append(final_event)
    return finalized


def active_label(track_id: int, track_states: dict[int, dict[str, Any]]) -> str:
    state = track_states.get(track_id)
    if not state:
        return "collecting"
    if state.get("final"):
        return str(state["final"].get("final_label", "unknown"))
    if state.get("embeddings"):
        return f"samples:{len(state['embeddings'])}"
    return "collecting"


def draw_result(frame, result, event_detector, frame_events, track_states, draw_event_line: bool):
    output = frame.copy()
    if draw_event_line:
        line_position = event_detector.line_position(output.shape)
        cv2.line(output, (0, line_position), (output.shape[1], line_position), (0, 0, 255), 2)
        cv2.putText(output, "NOVELTY PASS LINE", (12, max(24, line_position - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    for track in result.tracks:
        x, y, width, height = track.box
        color = TRACK_COLORS[(track.track_id - 1) % len(TRACK_COLORS)]
        label = active_label(track.track_id, track_states)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        cv2.putText(output, f"ID {track.track_id} {label}", (x, min(output.shape[0] - 8, y + height + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.putText(output, f"frame {result.frame_index}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    for index, event in enumerate(frame_events[:4]):
        novelty = event.get("novelty", {})
        text = f"PASSED ID {event['track_id']}: {novelty.get('final_label')} ({novelty.get('status')})"
        cv2.putText(output, text, (12, 60 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def create_writer(path: str | Path, frame_shape, fps: float):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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
    repeat_count = max(1, int(args.repeat))
    paths = paths * repeat_count

    config = apply_overrides(VisionPipelineConfig.from_yaml(args.config), args)
    pipeline = VisionPipeline(config)
    event_detector = ObjectPassingDetector(
        ObjectPassingConfig(
            axis=config.event_axis,
            line_ratio=args.line_ratio if args.line_ratio is not None else config.event_line_ratio,
            direction=config.event_direction,
            trigger_position=config.event_trigger_position,
            min_track_age=config.event_min_track_age,
            unknown_label=config.event_unknown_label,
        )
    )
    embedder = DinoEmbedder(model_name=args.dino_model, device=args.dino_device)
    novelty_runtime = load_novelty_runtime(
        gallery_path=resolve_project_path(args.gallery),
        known_embeddings_path=resolve_project_path(args.known_embeddings),
        calibration_path=resolve_project_path(args.calibration),
    )

    track_states: dict[int, dict[str, Any]] = {}
    results = []
    all_events = []
    writer = None
    output_path = None

    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"Skipping unreadable frame: {path}")
                continue

            result = pipeline.process_frame(frame)
            update_track_embeddings(frame, result, embedder, track_states, args)
            raw_events = event_detector.update(
                frame_index=result.frame_index,
                frame_shape=frame.shape,
                tracks=result.tracks,
                track_labels={},
            )
            frame_events = finalize_events(raw_events, track_states, novelty_runtime, args.min_embeddings)
            annotated = draw_result(frame, result, event_detector, frame_events, track_states, args.draw_event_line)

            if writer is None:
                output_path, writer = create_writer(args.output, annotated.shape, args.fps)
            writer.write(annotated)

            payload = result.to_dict()
            payload["image"] = str(path)
            payload["events"] = frame_events
            results.append(payload)
            all_events.extend(frame_events)
            print(
                f"[{result.frame_index}] {path.name}: "
                f"detections={len(result.detections)} tracks={len(result.tracks)} "
                f"events={len(frame_events)}"
            )
    finally:
        if writer is not None:
            writer.release()

    groups_output = resolve_project_path(args.groups_output)
    novelty_runtime.grouper.save(groups_output)
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if output_path is not None:
        print(f"Wrote video: {output_path}")
    print(f"Wrote groups: {groups_output}")
    print("Novelty events:", dict(Counter(event["novelty"]["status"] for event in all_events)))


if __name__ == "__main__":
    main()
