"""Live frame sources for headless runtime scripts."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROOT = PROJECT_ROOT / "Camera_feed"
if str(CAMERA_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMERA_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


class FrameSource:
    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        return


class FolderFrameSource(FrameSource):
    def __init__(self, folder: str | Path, *, repeat: bool = False, delay: float = 0.0) -> None:
        self.folder = Path(folder)
        self.repeat = repeat
        self.delay = max(0.0, float(delay))

    def frames(self) -> Iterator[np.ndarray]:
        paths = [
            path
            for path in sorted(self.folder.rglob("*"), key=natural_sort_key)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not paths:
            raise RuntimeError(f"No image frames found in {self.folder}")
        while True:
            for path in paths:
                frame = cv2.imread(str(path))
                if frame is not None:
                    yield frame
                    if self.delay:
                        time.sleep(self.delay)
            if not self.repeat:
                return


class OpenCvFrameSource(FrameSource):
    def __init__(self, source: str | int) -> None:
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open OpenCV frame source: {source}")

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return
            yield frame

    def close(self) -> None:
        self.capture.release()


class DirectIsaacFrameSource(FrameSource):
    def __init__(self, *, camera_path: str, width: int, height: int, auto_find_camera: bool) -> None:
        from direct_camera_reader import create_direct_camera_reader

        self.reader = create_direct_camera_reader(
            camera_path,
            width=width,
            height=height,
            auto_find_camera=auto_find_camera,
        )

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.reader.read_frame_bgr()
            if frame is None:
                time.sleep(0.01)
                continue
            yield frame


def create_frame_source(camera_config: dict, *, frames_dir: str | None = None, repeat: bool = False) -> FrameSource:
    if frames_dir:
        return FolderFrameSource(frames_dir, repeat=repeat, delay=float(camera_config.get("frame_delay", 0.0) or 0.0))

    source_type = str(camera_config.get("type", "direct_isaac")).lower()
    if source_type == "direct_isaac":
        return DirectIsaacFrameSource(
            camera_path=str(camera_config.get("camera_path", "/World/Camera")),
            width=int(camera_config.get("width", 640)),
            height=int(camera_config.get("height", 480)),
            auto_find_camera=bool(camera_config.get("auto_find_camera", True)),
        )
    if source_type in {"webcam", "opencv", "video"}:
        source = camera_config.get("source", 0)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        return OpenCvFrameSource(source)

    raise ValueError(f"Unsupported camera.type: {source_type}")
