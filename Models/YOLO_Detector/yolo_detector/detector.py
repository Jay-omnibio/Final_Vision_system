"""YOLO inference wrapper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .image_utils import load_image_bgr
from .types import BBox, DetectionResult

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PACKAGE_ROOT / "weights" / "best.pt"


class YoloDetector:
    """Run YOLO bounding-box inference on images or frames."""

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | int | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights) if weights is not None else DEFAULT_WEIGHTS
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")

        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(str(self.weights))

    def detect_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        array_format: str = "bgr",
    ) -> list[BBox]:
        """Return detected bounding boxes for one image."""
        frame = load_image_bgr(image, array_format=array_format)
        return self.detect_frame(frame)

    def detect(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        array_format: str = "bgr",
    ) -> DetectionResult:
        """Return boxes plus image metadata for one image."""
        frame = load_image_bgr(image, array_format=array_format)
        boxes = self.detect_frame(frame)
        height, width = frame.shape[:2]
        return DetectionResult(
            boxes=boxes,
            image_width=width,
            image_height=height,
            model_path=str(self.weights),
        )

    def detect_frame(self, frame_bgr: np.ndarray) -> list[BBox]:
        """Return detected bounding boxes for one BGR frame."""
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        boxes: list[BBox] = []
        names = getattr(result, "names", {}) or {}
        for raw_box in result.boxes:
            x1, y1, x2, y2 = raw_box.xyxy[0].tolist()
            score = float(raw_box.conf[0]) if raw_box.conf is not None else 1.0
            class_id = int(raw_box.cls[0]) if raw_box.cls is not None else None
            boxes.append(
                BBox(
                    x=int(round(x1)),
                    y=int(round(y1)),
                    width=max(0, int(round(x2 - x1))),
                    height=max(0, int(round(y2 - y1))),
                    score=score,
                    class_id=class_id,
                    class_name=names.get(class_id) if class_id is not None else None,
                )
            )
        return boxes

    def draw_boxes(
        self,
        image: str | Path | Image.Image | np.ndarray,
        boxes: list[BBox] | None = None,
        *,
        array_format: str = "bgr",
    ) -> np.ndarray:
        """Draw detected boxes on an image and return a BGR image."""
        frame = load_image_bgr(image, array_format=array_format)
        boxes = boxes if boxes is not None else self.detect_frame(frame)
        output = frame.copy()
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 80), 2)
            label = box.class_name or f"box {index}"
            text = f"{label} {box.score:.2f}"
            cv2.putText(
                output,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 80),
                2,
                cv2.LINE_AA,
            )
        return output

