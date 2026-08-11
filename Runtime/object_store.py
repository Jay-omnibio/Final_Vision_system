"""Persistent storage for passed-object events and teachable crops."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def safe_name(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9_.\\/-]+", "_", value)
    value = value.strip("_/")
    return value or "unknown"


def crop_box(frame: np.ndarray, box: tuple[int, int, int, int], padding: int = 12) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, box_width, box_height = box
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)
    return frame[y1:y2, x1:x2]


@dataclass(frozen=True)
class ObjectStoreConfig:
    root_dir: Path
    save_crops: bool = True
    crop_padding: int = 12


class ObjectEventStore:
    """Append-only object event store for later UI/teaching workflows."""

    def __init__(self, config: ObjectStoreConfig) -> None:
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.events_path = self.root_dir / "events.jsonl"
        self.crops_dir = self.root_dir / "crops"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if config.save_crops:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, project_root: Path) -> "ObjectEventStore | None":
        if not bool(config.get("enabled", True)):
            return None
        root_value = Path(str(config.get("root_dir", "data/operator_events")))
        root_dir = root_value if root_value.is_absolute() else project_root / root_value
        return cls(
            ObjectStoreConfig(
                root_dir=root_dir,
                save_crops=bool(config.get("save_crops", True)),
                crop_padding=int(config.get("crop_padding", 12)),
            )
        )

    def record_event(self, event: dict[str, Any], frame_bgr: np.ndarray) -> dict[str, Any]:
        record = dict(event)
        record.setdefault("timestamp", time.time())
        record["crop_path"] = None

        if self.config.save_crops:
            crop = crop_box(frame_bgr, tuple(record["box"]), self.config.crop_padding)
            if crop.size:
                label_dir = self.crops_dir / safe_name(str(record.get("label", "unknown")))
                label_dir.mkdir(parents=True, exist_ok=True)
                filename = (
                    f"frame_{int(record['frame_index']):06d}_"
                    f"track_{int(record['track_id']):04d}.jpg"
                )
                crop_path = label_dir / filename
                cv2.imwrite(str(crop_path), crop)
                record["crop_path"] = str(crop_path)

        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
