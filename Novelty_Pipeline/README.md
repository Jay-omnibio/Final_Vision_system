# Novelty Pipeline

Optional full run path for testing known/new decisions.

It is separate from `Vision_Pipeline`, so the stable detector/tracker/classifier/event pipeline can still run unchanged.

Flow:

```text
frames -> YOLO/subtract -> tracker -> DINO track embeddings -> bbox-edge pass event -> novelty known/new decision
```

Example:

```powershell
python Novelty_Pipeline\scripts\run_frames_to_novelty_video.py D:\Coding\Omnibio\frames `
  --config config.yaml `
  --detector yolo `
  --conf 0.45 `
  --device cpu `
  --fps 10 `
  --limit 600 `
  --draw-event-line
```
