# Operator Workflow

This document describes the intended user-facing flow for running the system,
teaching products, and handling unknown objects.

## Runtime Control

The first operator UI should be a small control panel, not a heavy vision app.

Core actions:

- Start system
- Stop system
- View passed-object events in order
- Edit/save config values
- Reload runtime after config/model changes

The headless runtime remains the source of truth:

```text
camera -> detector -> tracker -> classifier -> object_passed event
```

## Event History

Every passed object should be stored as:

- event metadata in `data/operator_events/events.jsonl`
- optional crop image under `data/operator_events/crops/<label>/`

This gives the UI real data to show and gives teaching workflows images to reuse.

## Teach New Known Object

Recommended flow:

1. Operator clicks `Teach Object`.
2. Operator enters class and object name.
3. System captures or imports multiple object images/crops.
4. DINO embeddings are generated.
5. Prototype gallery is rebuilt.
6. Runtime reloads the new gallery.

This should not require YOLO retraining if YOLO already detects the object on
the belt.

Current command-line version:

```powershell
python Teaching\scripts\label_event_crops.py `
  --class-name bottle `
  --object-name cola_bottle `
  --label-filter new_1

python Teaching\scripts\rebuild_active_gallery.py --update-config
```

The first command copies saved event crops into:

```text
data/teaching/images/<class>/<object>/
```

The second command writes:

```text
Models/Prototype_Classifier/galleries/active_gallery.npz
```

and can update `config.yaml` to use it.

## Handle Unknown Object

Novelty mode should never permanently add a new object without operator review.

Recommended flow:

1. Novelty runtime emits `new_1`, `new_2`, etc.
2. UI shows grouped unknown events and saved crops.
3. Operator clicks `Name This Object`.
4. Operator chooses existing class or creates a new class.
5. System moves/labels those crops into teaching data.
6. Gallery is rebuilt.
7. Future runs classify it by the operator-given name.

Current command-line version:

```powershell
python Teaching\scripts\label_event_crops.py `
  --class-name bottle `
  --object-name my_new_bottle `
  --label-filter new_3

python Teaching\scripts\rebuild_active_gallery.py --update-config
```

## Later Robotics Output

Robot/control integration should subscribe to final events, not inspect frames:

```json
{"event": "object_passed", "track_id": 12, "label": "bottle/cola_bottle", "score": 0.82}
```

Output adapters can be added later:

- console
- JSONL
- socket
- ROS
- API callback
