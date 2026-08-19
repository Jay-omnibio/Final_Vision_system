# Final Vision System

Clean standalone project for the final Omnibio vision runtime.

Current migration scope:

- `Models/DINO`: DINOv2-small and DINOv3 image embedding generation.
- `Models/YOLO_Detector`: final trained YOLO bounding-box inference.
- `Models/Object_Tracker`: centroid-based object tracking.
- `Models/Subtract_Detector`: historical/debug background-subtraction detector.
- `Models/Prototype_Classifier`: support classifier/gallery code.
- `Models/Novelty_Detector`: main known/new classification runtime.
- `Camera_feed`: direct Isaac camera frame readers.
- `Vision_Pipeline`: fast local detector + tracker runtime.
- `config.yaml`: project-level runtime configuration.
- `Runtime`: headless live/runtime scripts for VM or robot integration.
- `Teaching`: operator crop labeling and active gallery rebuild helpers.
- `docs/operator_workflow.md`: operator UI, teaching, and unknown handling flow.

## Headless Runtime

Run live from the configured camera source:

```powershell
python Runtime\run_live_vision.py --config config.yaml --detector yolo --conf 0.45 --device cpu
```

Run locally from saved frames:

```powershell
python Runtime\run_live_vision.py --frames-dir D:\Coding\Omnibio\frames --limit 300 --detector yolo --conf 0.45 --device cpu
```

For HTTP/API camera input, set `camera.type: api` in `config.yaml` and put
`CAMERA_API_URL` / `CAMERA_API_KEY` in a local `.env`. Temporary API misses are
retried so the runtime does not stop on a single dropped frame.

Debug a headless run by saving an annotated video:

```powershell
python Runtime\run_live_vision.py --config config.yaml --debug-video
```

The video is written under `outputs/live_debug_<timestamp>.mp4` when the run
stops.

Run live novelty/unknown detection:

```powershell
python Runtime\run_live_novelty.py --config config.yaml --debug-video
```

Novelty live is the main operator path and may output `known` or `new_k`.
Normal live remains available for comparison/debug.

## DINO Backend

Both normal and novelty runtime read DINO settings from `classifier` in
`config.yaml`:

```yaml
classifier:
  dino_model: dinov2-small
  dino_backend: onnx  # torch or onnx
  dino_onnx_path: Models/DINO/onnx/dinov2-small.onnx
  device: auto
```

Use `dino_backend: torch` to run the original PyTorch/Transformers embedder.
Use `dino_backend: onnx` after exporting and comparing
`Models/DINO/onnx/dinov2-small.onnx`.

## Operator App

Start the local operator app:

```powershell
python Operator_App\server.py --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860` to start/stop the runtime, view object-passed
events, review unknown `new_k` crops, assign labels into teaching data, rebuild
the active gallery and novelty calibration, and edit common config values.
Runtime mode is read from `operator_app.runtime_mode` in `config.yaml`. Runtime
stdout/stderr logs are written under `outputs/runtime_logs/`.

## Teaching Objects

After runtime saves event crops in `data/operator_events`, label selected crops
into the teaching dataset:

```powershell
python Teaching\scripts\label_event_crops.py --class-name bottle --object-name cola_bottle --label-filter new_1
```

Rebuild the active classifier gallery:

```powershell
python Teaching\scripts\rebuild_active_gallery.py --update-config
```
