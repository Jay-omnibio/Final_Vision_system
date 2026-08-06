# Subtract Detector Module

Standalone background-subtraction detector for the final vision system.

This module finds object bounding boxes by comparing a frame with an empty-belt
background image. It does not train a model, classify crops, track objects, or
call the robotics API.

## Python Usage

```python
from subtract_detector import BackgroundSubtractionConfig, SubtractDetector

detector = SubtractDetector.from_background_image(
    "no_object.jpg",
    mode="improved",
    config=BackgroundSubtractionConfig(threshold=30, kernel_size=25),
)

boxes = detector.detect_image("frame_with_object.jpg")
for box in boxes:
    print(box.x, box.y, box.width, box.height)
```

## Modes

| Mode | Behavior |
| --- | --- |
| `standard` | BGR image difference, grayscale threshold, morphology |
| `improved` | BGR difference intersected with HSV value-channel difference |

The `improved` mode was used to reduce shadow blobs in the old detector lab.

## Example

```powershell
python Models\Subtract_Detector\examples\detect_image.py no_object.jpg frame.jpg --output outputs\subtract.jpg
```

Boxes use this format:

```text
(x, y, width, height)
```

