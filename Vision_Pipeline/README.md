# Vision Pipeline

Fast local pipeline that connects a detector to the object tracker.

Current scope:

- load one detector once
- load one tracker once
- process frames repeatedly
- return raw detection boxes and tracked objects

Not included yet:

- DINO embeddings
- classification
- gallery/prototype matching
- novelty detection
- Streamlit
- robotics API calls

## Detector Options

| Detector | Needs |
| --- | --- |
| `yolo` | `Models/YOLO_Detector/weights/best.pt` |
| `subtract` | an empty-background image |

## Python Usage

```python
from pipeline_core import VisionPipeline, VisionPipelineConfig

pipeline = VisionPipeline(
    VisionPipelineConfig(
        detector_type="yolo",
    )
)

result = pipeline.process_frame(frame_bgr)
print(result.detections)
print(result.tracks)
```

For subtract:

```python
pipeline = VisionPipeline(
    VisionPipelineConfig(
        detector_type="subtract",
        background_image="no_object.jpg",
    )
)
```

Or load root config:

```python
pipeline = VisionPipeline.from_yaml("config.yaml")
```

## Script Usage

Run one image:

```powershell
python Vision_Pipeline\scripts\run_one_image.py path\to\image.jpg --config config.yaml --detector yolo
```

Run subtract with a background image:

```powershell
python Vision_Pipeline\scripts\run_one_image.py frame.jpg --config config.yaml --detector subtract --background no_object.jpg
```

Run a folder in sorted order so tracker IDs persist across frames:

```powershell
python Vision_Pipeline\scripts\run_image_sequence.py path\to\frames --config config.yaml --detector yolo
```
