# Camera Feed

Local camera readers for running the final vision system directly inside the VM
or Isaac Sim environment.

Current scope:

- read frames directly from an Isaac/Omniverse camera render product
- avoid Streamlit and HTTP API camera polling
- return NumPy frames for the vision pipeline

## Direct Isaac Camera

```python
from camera_feed import create_direct_camera_reader

reader = create_direct_camera_reader("/World/Camera")
reader.initialize()

frame_bgr = reader.read_frame_bgr()
```

`read_frame_rgb()` returns RGB frames. `read_frame_bgr()` returns OpenCV-style BGR
frames for detector/pipeline code.

## Demo

Run this inside the Isaac Python environment where `omni`, `pxr`, and
`omni.replicator.core` are available:

```powershell
python Camera_feed\scripts\direct_camera_demo.py
```

## Save API Frames

For slower HTTP/API testing, save frames first and run the pipeline offline.
Create a local `.env` from `.env.example` and set your API key:

```powershell
copy .env.example .env
```

Then run:

```powershell
python Camera_feed\save_frames.py --out-dir frames --count 2400 --delay 0.2
```

The script reads:

```text
CAMERA_API_URL
CAMERA_API_KEY
```

from `.env` or the shell environment. Do not commit `.env`.
