# YOLO Detector Module

Standalone bounding-box detection code for the final vision system.

This module only runs YOLO inference. It does not train models, prepare datasets,
evaluate detector benchmarks, classify objects, track objects, or call the
robotics API.

## Setup

```powershell
cd D:\Coding\Omnibio\Final_Model_train\Final_Vision_system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Python Usage

```python
from yolo_detector import YoloDetector

detector = YoloDetector()
detections = detector.detect_image("sample.jpg")

for box in detections:
    print(box.x, box.y, box.width, box.height, box.score)
```

## Example

```powershell
python Models\YOLO_Detector\examples\detect_image.py path\to\image.jpg --output outputs\detection.jpg
```

Default weights live at:

```text
Models/YOLO_Detector/weights/best.onnx
```

If `best.onnx` is not available, the detector also accepts:

```text
Models/YOLO_Detector/weights/best.pt
```

The migrated default model is the trained single-class conveyor object detector
from the previous detector lab.

