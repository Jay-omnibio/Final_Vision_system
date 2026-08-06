#!/usr/bin/env python
"""Run a tiny synthetic tracking sequence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from object_tracker import CentroidTracker


def main() -> None:
    frames = [
        [(10, 20, 50, 60)],
        [(12, 32, 50, 60)],
        [(14, 44, 50, 60), (200, 40, 30, 30)],
        [(16, 56, 50, 60), (202, 52, 30, 30)],
        [(18, 68, 50, 60)],
    ]

    tracker = CentroidTracker(max_distance=80, max_missing_frames=1)
    results = []
    for frame_index, boxes in enumerate(frames, start=1):
        tracks = tracker.update(boxes)
        results.append(
            {
                "frame": frame_index,
                "tracks": [track.to_dict() for track in tracks],
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

