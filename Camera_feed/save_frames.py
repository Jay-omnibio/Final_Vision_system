#!/usr/bin/env python
"""Save camera frames from the Isaac demo HTTP endpoint."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("CAMERA_API_URL", "http://127.0.0.1:3000"))
    parser.add_argument("--api-key", default=os.environ.get("CAMERA_API_KEY", ""))
    parser.add_argument("--endpoint", default="/api/demo/camera/rgb/frame")
    parser.add_argument("--out-dir", default="frames")
    parser.add_argument("--count", type=int, default=2400)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--prefix", default="frame")
    parser.add_argument("--wait", action="store_true", help="Add wait=true query param if the API supports it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = args.api_url.rstrip("/")
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    frame_url = f"{api_url}{endpoint}"
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    params = {"wait": "true"} if args.wait else None

    print(f"Saving frames to: {out_dir.resolve()}")
    print(f"Reading from: {frame_url}")

    for offset in range(args.count):
        frame_index = args.start_index + offset
        try:
            response = requests.get(
                frame_url,
                headers=headers,
                params=params,
                timeout=args.timeout,
            )
            if response.status_code != 200:
                print(f"[{frame_index}] error {response.status_code}: {response.text[:200]}")
                time.sleep(args.delay)
                continue

            image = cv2.imdecode(
                np.frombuffer(response.content, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                print(f"[{frame_index}] decode failed")
                time.sleep(args.delay)
                continue

            output_path = out_dir / f"{args.prefix}_{frame_index:06d}.jpg"
            if cv2.imwrite(str(output_path), image):
                print(f"[{frame_index}] saved {output_path}")
            else:
                print(f"[{frame_index}] failed to save {output_path}")

        except Exception as exc:
            print(f"[{frame_index}] exception: {exc}")

        if args.delay > 0:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()

