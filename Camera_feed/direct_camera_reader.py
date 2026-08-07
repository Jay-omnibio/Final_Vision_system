"""Direct Isaac camera frame reader for a non-EE camera."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectCameraConfig:
    camera_path: str = "/World/Camera"
    width: int = 640
    height: int = 480
    auto_find_camera: bool = True
    exclude_name_contains: tuple[str, ...] = ("ee_camera", "OmniverseKit")


class DirectCameraReader:
    """Read frames directly from an Isaac camera/render product."""

    def __init__(
        self,
        camera_path: str = "/World/Camera",
        *,
        width: int = 640,
        height: int = 480,
        auto_find_camera: bool = True,
    ) -> None:
        self.config = DirectCameraConfig(
            camera_path=camera_path,
            width=width,
            height=height,
            auto_find_camera=auto_find_camera,
        )
        self.camera_path = camera_path
        self._initialized = False
        self._rep = None
        self._annotator = None
        self._render_product = None

    def initialize(self) -> None:
        if self._initialized:
            return

        importlib.import_module("omni")
        usd = importlib.import_module("omni.usd")
        rep = importlib.import_module("omni.replicator.core")
        from pxr import UsdGeom

        stage = usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No Omniverse stage available")

        camera_path = self.config.camera_path
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid() and self.config.auto_find_camera:
            camera_path, camera_prim = self._find_camera(stage, UsdGeom)

        if not camera_prim.IsValid():
            raise RuntimeError(f"Camera prim not found at {self.config.camera_path}")

        self.camera_path = camera_path
        self._rep = rep
        render_product = rep.create.render_product(
            camera_path,
            (self.config.width, self.config.height),
        )
        self._render_product = getattr(render_product, "path", render_product)
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach(self._render_product)
        self._initialized = True

    def read_frame_rgb(self) -> np.ndarray | None:
        """Return one RGB frame as uint8, or None if no frame is ready."""
        self.initialize()
        if self._rep is None or self._annotator is None:
            return None

        try:
            self._rep.orchestrator.step_async(pause_timeline=False)
        except Exception:
            pass

        data = self._annotator.get_data()
        if data is None:
            return None

        array = np.asarray(data)
        if array.ndim == 3 and array.shape[2] >= 3:
            return array[:, :, :3].astype(np.uint8)
        return None

    def read_frame_bgr(self) -> np.ndarray | None:
        """Return one OpenCV-style BGR frame as uint8."""
        frame_rgb = self.read_frame_rgb()
        if frame_rgb is None:
            return None
        return frame_rgb[:, :, ::-1].copy()

    def _find_camera(self, stage, UsdGeom):
        for prim in stage.Traverse():
            path_text = str(prim.GetPath())
            lowered = path_text.lower()
            if not prim.IsA(UsdGeom.Camera):
                continue
            if any(excluded.lower() in lowered for excluded in self.config.exclude_name_contains):
                continue
            return path_text, stage.GetPrimAtPath(path_text)
        return self.config.camera_path, stage.GetPrimAtPath(self.config.camera_path)


def create_direct_camera_reader(
    camera_path: str = "/World/Camera",
    *,
    width: int = 640,
    height: int = 480,
    auto_find_camera: bool = True,
) -> DirectCameraReader:
    return DirectCameraReader(
        camera_path=camera_path,
        width=width,
        height=height,
        auto_find_camera=auto_find_camera,
    )

