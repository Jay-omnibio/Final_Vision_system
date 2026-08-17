"""Config helpers for operator-safe runtime settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigService:
    def __init__(self, config_path: Path, *, project_root: Path) -> None:
        self.config_path = config_path
        self.project_root = project_root

    def load(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            return {}
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}

    def save(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def runtime_mode(self) -> str:
        config = self.load()
        mode = str(config.get("operator_app", {}).get("runtime_mode", "novelty")).lower()
        return "normal" if mode == "normal" else "novelty"

    def debug_video_default(self) -> bool:
        config = self.load()
        return bool(config.get("operator_app", {}).get("debug_video", False))

    def settings(self) -> dict[str, Any]:
        config = self.load()
        vision = config.get("vision", {}) or {}
        yolo = vision.get("yolo", {}) or {}
        events = vision.get("events", {}) or {}
        classifier = config.get("classifier", {}) or {}
        camera = config.get("camera", {}) or {}
        operator_store = config.get("operator_store", {}) or {}
        operator_app = config.get("operator_app", {}) or {}
        return {
            "runtime_mode": str(operator_app.get("runtime_mode", "novelty")),
            "debug_video": bool(operator_app.get("debug_video", False)),
            "detector_type": str(vision.get("detector_type", "yolo")),
            "yolo_weights": str(yolo.get("weights", "")),
            "yolo_conf": float(yolo.get("conf", 0.45)),
            "yolo_device": "" if yolo.get("device") is None else str(yolo.get("device")),
            "event_line_ratio": float(events.get("line_ratio", 0.88)),
            "event_trigger_position": str(events.get("trigger_position", "leading_edge")),
            "gallery_path": str(classifier.get("gallery_path", "")),
            "dino_backend": str(classifier.get("dino_backend", "torch")),
            "dino_onnx_path": str(classifier.get("dino_onnx_path", "")),
            "classifier_device": str(classifier.get("device", "auto")),
            "camera_type": str(camera.get("type", "api")),
            "camera_api_url": str(camera.get("api_url", "")),
            "camera_endpoint": str(camera.get("endpoint", "")),
            "camera_timeout": float(camera.get("timeout", 10.0)),
            "store_root_dir": str(operator_store.get("root_dir", "data/operator_events")),
        }

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self.load()
        vision = config.setdefault("vision", {})
        yolo = vision.setdefault("yolo", {})
        events = vision.setdefault("events", {})
        classifier = config.setdefault("classifier", {})
        camera = config.setdefault("camera", {})
        operator_store = config.setdefault("operator_store", {})
        operator_app = config.setdefault("operator_app", {})

        if "runtime_mode" in payload:
            operator_app["runtime_mode"] = "normal" if payload["runtime_mode"] == "normal" else "novelty"
        if "debug_video" in payload:
            operator_app["debug_video"] = bool(payload["debug_video"])
        if "detector_type" in payload:
            vision["detector_type"] = str(payload["detector_type"])
        if "yolo_weights" in payload:
            yolo["weights"] = str(payload["yolo_weights"])
        if "yolo_conf" in payload:
            yolo["conf"] = float(payload["yolo_conf"])
        if "yolo_device" in payload:
            value = str(payload["yolo_device"]).strip()
            yolo["device"] = value or None
        if "event_line_ratio" in payload:
            events["line_ratio"] = float(payload["event_line_ratio"])
        if "event_trigger_position" in payload:
            events["trigger_position"] = str(payload["event_trigger_position"])
        if "gallery_path" in payload:
            classifier["gallery_path"] = str(payload["gallery_path"])
        if "dino_backend" in payload:
            classifier["dino_backend"] = str(payload["dino_backend"])
        if "dino_onnx_path" in payload:
            classifier["dino_onnx_path"] = str(payload["dino_onnx_path"])
        if "classifier_device" in payload:
            classifier["device"] = str(payload["classifier_device"])
        if "camera_type" in payload:
            camera["type"] = str(payload["camera_type"])
        if "camera_api_url" in payload:
            camera["api_url"] = str(payload["camera_api_url"])
        if "camera_endpoint" in payload:
            camera["endpoint"] = str(payload["camera_endpoint"])
        if "camera_timeout" in payload:
            camera["timeout"] = float(payload["camera_timeout"])
        if "store_root_dir" in payload:
            operator_store["root_dir"] = str(payload["store_root_dir"])

        self.save(config)
        return self.settings()
