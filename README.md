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

Other parts such as prototype galleries, classifiers, novelty logic, UI, and
robotics integration will be migrated later as separate modules.
