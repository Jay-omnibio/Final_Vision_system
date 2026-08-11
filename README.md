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
