"""Standalone centroid-based object tracker package."""

from .tracker import CentroidTracker, box_centroid, centroid_distance
from .types import Box, Point, TrackedObject

__all__ = [
    "Box",
    "CentroidTracker",
    "Point",
    "TrackedObject",
    "box_centroid",
    "centroid_distance",
]

