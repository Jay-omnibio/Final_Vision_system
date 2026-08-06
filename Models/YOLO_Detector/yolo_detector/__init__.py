"""Standalone YOLO bounding-box detector package."""

from .detector import YoloDetector
from .types import BBox, DetectionResult

__all__ = ["BBox", "DetectionResult", "YoloDetector"]

