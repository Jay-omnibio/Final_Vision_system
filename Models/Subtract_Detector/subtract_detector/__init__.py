"""Standalone background-subtraction detector package."""

from .detector import SubtractDetector
from .types import BBox, BackgroundSubtractionConfig, Detection

__all__ = ["BBox", "BackgroundSubtractionConfig", "Detection", "SubtractDetector"]

