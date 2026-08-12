"""Live frame sources for headless runtime scripts."""

from __future__ import annotations

import re
import sys
import time
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROOT = PROJECT_ROOT / "Camera_feed"
if str(CAMERA_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMERA_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


class ApiFrameSource(FrameSource):
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str = "",
        endpoint: str = "/api/demo/camera/rgb/frame",
        timeout: float = 10.0,
        delay: float = 0.0,
        retry_delay: float = 0.2,
        wait: bool = False,
    ) -> None:
        api_url = api_url.rstrip("/")
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.frame_url = f"{api_url}{endpoint}"
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.timeout = float(timeout)
        self.delay = max(0.0, float(delay))
        self.retry_delay = max(0.0, float(retry_delay))
        self.params = {"wait": "true"} if wait else None

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            try:
                response = requests.get(
                    self.frame_url,
                    headers=self.headers,
                    params=self.params,
                    timeout=self.timeout,
                )
                if response.status_code != 200:
                    print(f"Frame API returned {response.status_code}: {response.text[:200]}", flush=True)
                    time.sleep(self.retry_delay)
                    continue
                frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print("Frame API response could not be decoded as an image", flush=True)
                    time.sleep(self.retry_delay)
                    continue
            except requests.RequestException as exc:
                print(f"Frame API request failed: {exc}", flush=True)
                time.sleep(self.retry_delay)
                continue
            yield frame
            if self.delay:
                time.sleep(self.delay)


def create_frame_source(camera_config: dict, *, frames_dir: str | None = None, repeat: bool = False) -> FrameSource:
    load_dotenv(PROJECT_ROOT / ".env")
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
    if source_type in {"api", "http_api", "isaac_api"}:
        return ApiFrameSource(
            api_url=str(camera_config.get("api_url") or os.environ.get("CAMERA_API_URL", "http://127.0.0.1:3000")),
            api_key=str(camera_config.get("api_key") or os.environ.get("CAMERA_API_KEY", "")),
            endpoint=str(camera_config.get("endpoint", "/api/demo/camera/rgb/frame")),
            timeout=float(camera_config.get("timeout", 10.0)),
            delay=float(camera_config.get("frame_delay", 0.0) or 0.0),
            retry_delay=float(camera_config.get("retry_delay", 0.2) or 0.0),
            wait=bool(camera_config.get("wait", False)),
        )

    raise ValueError(f"Unsupported camera.type: {source_type}")
