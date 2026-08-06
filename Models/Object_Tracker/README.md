# Object Tracker Module

Standalone centroid-based object tracking for the final vision system.

This module only assigns stable track IDs to bounding boxes across frames. It
does not detect boxes, classify crops, make final decisions, or call the
robotics API.

## Python Usage

```python
from object_tracker import CentroidTracker

tracker = CentroidTracker(max_distance=300, max_missing_frames=4)

frame_1_tracks = tracker.update([(10, 20, 50, 60)])
frame_2_tracks = tracker.update([(12, 28, 50, 60)])

print(frame_2_tracks[0].track_id)
```

Boxes use this format:

```text
(x, y, width, height)
```

The default tracker prefers downward motion because the conveyor objects usually
move down the image. Set `prefer_downward_motion=False` for general tracking.

## Example

```powershell
python Models\Object_Tracker\examples\track_sequence.py
```

