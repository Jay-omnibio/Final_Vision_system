# Vision Pipeline

Fast local pipeline that connects YOLO detection to the object tracker.

Current scope:

- load the YOLO detector once
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

## Detector

The final pipeline uses YOLO only. Background subtraction was removed from the
final runtime path.

## Python Usage

```python
from pipeline_core import VisionPipeline, VisionPipelineConfig

pipeline = VisionPipeline(
    VisionPipelineConfig()
)

result = pipeline.process_frame(frame_bgr)
print(result.detections)
print(result.tracks)
```

Or load root config:

```python
pipeline = VisionPipeline.from_yaml("config.yaml")
```

## Script Usage

Run one image:

```powershell
python Vision_Pipeline\scripts\run_one_image.py path\to\image.jpg --config config.yaml
```

Run a folder in sorted order so tracker IDs persist across frames:

```powershell
python Vision_Pipeline\scripts\run_image_sequence.py path\to\frames --config config.yaml
```

Run saved frames and write an annotated video:

```powershell
python Vision_Pipeline\scripts\run_frames_to_video.py frames --config config.yaml --output outputs\pipeline_output.mp4
```
