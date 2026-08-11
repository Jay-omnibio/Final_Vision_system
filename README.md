# Final Vision System

Clean standalone project for the final Omnibio vision runtime.

Current migration scope:

- `Models/DINO`: DINOv2-small and DINOv3 image embedding generation.
- `Models/YOLO_Detector`: trained YOLO bounding-box inference.
- `Models/Object_Tracker`: centroid-based object tracking.
- `Models/Subtract_Detector`: background-subtraction bounding-box detection.
- `Models/Prototype_Classifier`: DINO embedding + `.npz` prototype classification.
- `Camera_feed`: direct Isaac camera frame readers.
- `Vision_Pipeline`: fast local detector + tracker runtime.
- `config.yaml`: project-level runtime configuration.
- `Runtime`: headless live/runtime scripts for VM or robot integration.
- `docs/operator_workflow.md`: planned operator UI, teaching, and unknown handling flow.

Other parts such as prototype galleries, classifiers, novelty logic, UI, and
robotics integration will be migrated later as separate modules.

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
`CAMERA_API_URL` / `CAMERA_API_KEY` in a local `.env`.

Debug a headless run by saving an annotated video:

```powershell
python Runtime\run_live_vision.py --config config.yaml --debug-video
```

The video is written under `outputs/live_debug_<timestamp>.mp4` when the run
stops.

## Local Control Panel

Start a small local operator page:

```powershell
python Runtime\control_server.py --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860` to start/stop the runtime, optionally enable debug
video, edit `config.yaml`, and view recent object-passed events.

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
