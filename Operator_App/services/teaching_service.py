"""Teaching-data actions for reviewed operator events."""

from __future__ import annotations

import subprocess
import threading
import time
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
        self._rebuild_job = {
            "running": False,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "output": "",
            "ok": None,
        }

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
            "samples": [
                {
                    "label": sample.label,
                    "image_path": str(sample.image_path),
                    "source_path": str(sample.source_path),
                }
                for sample in imported
            ],
        }

    def teaching_samples(self) -> list[dict[str, Any]]:
        store = TeachingStore(self.teaching_dir)
        samples = []
        for class_name, object_name, image_path in store.iter_labeled_images():
            samples.append(
                {
                    "label": f"{class_name}/{object_name}",
                    "class_name": class_name,
                    "object_name": object_name,
                    "image_path": str(image_path),
                }
            )
        return samples

    def delete_samples(self, image_paths: list[str]) -> dict[str, Any]:
        deleted = 0
        for image_path in image_paths:
            resolved = self.resolve_sample(image_path)
            if resolved.is_file():
                resolved.unlink()
                deleted += 1
        return {"deleted": deleted}

    def resolve_sample(self, image_path: str) -> Path:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.project_root / path
        resolved = path.resolve()
        root = self.teaching_dir.resolve()
        if root not in resolved.parents:
            raise ValueError("Sample path is outside teaching data")
        if not resolved.is_file():
            raise FileNotFoundError(f"Teaching sample not found: {resolved}")
        return resolved

    def start_rebuild_gallery(self) -> dict[str, Any]:
        if self._rebuild_job["running"]:
            return self.rebuild_status()
        self._rebuild_job = {
            "running": True,
            "started_at": time.time(),
            "finished_at": None,
            "returncode": None,
            "output": "Rebuild started...",
            "ok": None,
        }
        thread = threading.Thread(target=self._run_rebuild_gallery, daemon=True)
        thread.start()
        return self.rebuild_status()

    def rebuild_status(self) -> dict[str, Any]:
        return dict(self._rebuild_job)

    def _run_rebuild_gallery(self) -> None:
        classifier = self._classifier_config()
        command = [
            self.python_exe,
            "Teaching/scripts/rebuild_active_gallery.py",
            "--update-config",
            "--base-gallery",
            str(classifier.get("base_gallery_path", "Models/Novelty_Detector/artifacts/prototypes/gallery_known_hierarchical.npz")),
            "--base-known-embeddings",
            str(classifier.get("base_known_embeddings_path", "Models/Novelty_Detector/artifacts/embeddings/known.npz")),
            "--dino-model",
            str(classifier.get("dino_model", "dinov2-small")),
            "--dino-backend",
            str(classifier.get("dino_backend", "torch")),
            "--device",
            str(classifier.get("device", "auto")),
        ]
        if classifier.get("dino_onnx_path"):
            command.extend(["--dino-onnx-path", str(classifier["dino_onnx_path"])])
        try:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
            self._rebuild_job.update(
                {
                    "returncode": completed.returncode,
                    "output": completed.stdout[-12000:],
                    "ok": completed.returncode == 0,
                }
            )
        except Exception as exc:
            self._rebuild_job.update({"returncode": -1, "output": str(exc), "ok": False})
        finally:
            self._rebuild_job.update({"running": False, "finished_at": time.time()})

    def _classifier_config(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            return {}
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        return config.get("classifier", {}) or {}
