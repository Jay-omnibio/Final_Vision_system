"""Teaching-data actions for reviewed operator events."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from Teaching.teaching_store import TeachingStore


class TeachingService:
    def __init__(self, *, project_root: Path, python_exe: str, config_path: Path) -> None:
        self.project_root = project_root
        self.python_exe = python_exe
        self.config_path = config_path
        self.teaching_dir = project_root / "data" / "teaching"

    def assign_crops(
        self,
        crop_paths: list[str],
        *,
        class_name: str,
        object_name: str,
        events_by_crop: dict[str, dict[str, Any] | None],
    ) -> dict[str, Any]:
        if not class_name.strip() or not object_name.strip():
            raise ValueError("class_name and object_name are required")
        store = TeachingStore(self.teaching_dir)
        imported = []
        for crop_path in crop_paths:
            source_event = events_by_crop.get(crop_path)
            samples = store.add_images(
                [crop_path],
                class_name=class_name,
                object_name=object_name,
                source_event=source_event,
            )
            imported.extend(samples)
        return {
            "imported": len(imported),
            "label": f"{class_name}/{object_name}",
            "teaching_dir": str(self.teaching_dir),
        }

    def rebuild_gallery(self) -> dict[str, Any]:
        classifier = self._classifier_config()
        command = [
            self.python_exe,
            "Teaching/scripts/rebuild_active_gallery.py",
            "--update-config",
            "--dino-model",
            str(classifier.get("dino_model", "dinov2-small")),
            "--dino-backend",
            str(classifier.get("dino_backend", "torch")),
            "--device",
            str(classifier.get("device", "auto")),
        ]
        if classifier.get("dino_onnx_path"):
            command.extend(["--dino-onnx-path", str(classifier["dino_onnx_path"])])
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        return {
            "returncode": completed.returncode,
            "output": completed.stdout[-8000:],
            "ok": completed.returncode == 0,
        }

    def _classifier_config(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            return {}
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        return config.get("classifier", {}) or {}
