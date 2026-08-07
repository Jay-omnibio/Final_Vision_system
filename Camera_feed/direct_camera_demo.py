#!/usr/bin/env python3
"""Demo for reading frames directly from the main Isaac camera."""

from __future__ import annotations

import cv2

from direct_camera_reader import create_direct_camera_reader


def main() -> None:
    reader = create_direct_camera_reader("/World/Camera")
    print("Initializing direct camera reader...")
    reader.initialize()

    for i in range(10):
        frame = reader.read_frame_bgr()
        if frame is None:
            print(f"[{i}] no frame")
            continue
        print(f"[{i}] frame shape={frame.shape} dtype={frame.dtype}")
        cv2.imshow("direct_camera", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()