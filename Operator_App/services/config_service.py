"""Config helpers for operator-safe runtime settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


BASE_CLASSIFIER_GALLERY = "Models/Prototype_Classifier/galleries/default_gallery.npz"
BASE_NOVELTY_GALLERY = "Models/Novelty_Detector/artifacts/prototypes/gallery_known_hierarchical.npz"
BASE_NOVELTY_KNOWN = "Models/Novelty_Detector/artifacts/embeddings/known.npz"
BASE_NOVELTY_CALIBRATION = "Models/Novelty_Detector/artifacts/calibration/novelty_mahalanobis.npz"


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

    def events_path(self) -> Path:
        config = self.load()
        operator_store = config.get("operator_store", {}) or {}
        root = str(operator_store.get("root_dir", "data/operator_events"))
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = self.project_root / root_path
        return root_path / "events.jsonl"

    def settings(self) -> dict[str, Any]:
        config = self.load()
        vision = config.get("vision", {}) or {}
        yolo = vision.get("yolo", {}) or {}
        events = vision.get("events", {}) or {}
        classifier = config.get("classifier", {}) or {}
        novelty = config.get("novelty", {}) or {}
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
            "novelty_gallery_path": str(novelty.get("gallery_path", "")),
            "novelty_known_embeddings_path": str(novelty.get("known_embeddings_path", "")),
            "novelty_calibration_path": str(novelty.get("calibration_path", "")),
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
        novelty = config.setdefault("novelty", {})
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
        if "novelty_gallery_path" in payload:
            novelty["gallery_path"] = str(payload["novelty_gallery_path"])
        if "novelty_known_embeddings_path" in payload:
            novelty["known_embeddings_path"] = str(payload["novelty_known_embeddings_path"])
        if "novelty_calibration_path" in payload:
            novelty["calibration_path"] = str(payload["novelty_calibration_path"])
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

    def model_status(self) -> dict[str, Any]:
        settings = self.settings()
        paths = {
            "classifier_gallery": settings["gallery_path"],
            "novelty_gallery": settings["novelty_gallery_path"],
            "known_embeddings": settings["novelty_known_embeddings_path"],
            "calibration": settings["novelty_calibration_path"],
            "dino_onnx": settings["dino_onnx_path"],
            "yolo_weights": settings["yolo_weights"],
        }
        runtime_mode = settings["runtime_mode"].lower()
        required_artifacts = {
            "classifier_gallery": runtime_mode == "normal",
            "novelty_gallery": runtime_mode == "novelty",
            "known_embeddings": runtime_mode == "novelty",
            "calibration": runtime_mode == "novelty",
            "dino_onnx": settings["dino_backend"].lower() == "onnx",
            "yolo_weights": True,
        }
        artifacts = {
            name: self._artifact_status(path, required=required_artifacts[name])
            for name, path in paths.items()
        }
        using_active = any("active_" in str(path) or "active_gallery" in str(path) for path in paths.values())
        warnings = [
            f"{name} missing: {item['path']}"
            for name, item in artifacts.items()
            if item["required"] and not item["exists"]
        ]
        return {
            "runtime_mode": settings["runtime_mode"],
            "dino_backend": settings["dino_backend"],
            "using_active_model": using_active,
            "artifacts": artifacts,
            "warnings": warnings,
        }

    def reset_base_model(self) -> dict[str, Any]:
        config = self.load()
        classifier = config.setdefault("classifier", {})
        novelty = config.setdefault("novelty", {})
        classifier["gallery_path"] = BASE_CLASSIFIER_GALLERY
        novelty["gallery_path"] = BASE_NOVELTY_GALLERY
        novelty["known_embeddings_path"] = BASE_NOVELTY_KNOWN
        novelty["calibration_path"] = BASE_NOVELTY_CALIBRATION
        self.save(config)
        return self.model_status()

    def _artifact_status(self, path: str, *, required: bool = True) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        status: dict[str, Any] = {
            "path": path,
            "resolved_path": str(resolved) if resolved else "",
            "exists": bool(resolved and resolved.is_file()),
            "required": bool(path and required),
        }
        if not status["exists"] or resolved is None:
            return status
        status["size_bytes"] = resolved.stat().st_size
        if resolved.suffix.lower() == ".npz":
            try:
                with np.load(resolved, allow_pickle=True) as data:
                    if "subclass_names" in data.files:
                        status["subclass_count"] = int(len(data["subclass_names"]))
                    if "class_names" in data.files:
                        status["class_count"] = int(len(data["class_names"]))
                    if "labels" in data.files:
                        status["label_count"] = int(len(data["labels"]))
                    if "embeddings" in data.files:
                        status["embedding_count"] = int(len(data["embeddings"]))
            except Exception as exc:
                status["warning"] = str(exc)
        return status

    def _resolve_path(self, path: str) -> Path | None:
        if not path:
            return None
        value = Path(path)
        return value if value.is_absolute() else self.project_root / value
