"""Utilities for maintaining operator-taught image datasets."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def safe_segment(value: str) -> str:
    value = value.strip().replace("\\", "/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "unknown"


def safe_label(class_name: str, object_name: str) -> tuple[str, str]:
    return safe_segment(class_name), safe_segment(object_name)


@dataclass(frozen=True)
class TeachingSample:
    class_name: str
    object_name: str
    image_path: Path
    source_path: Path
    source_event: dict | None = None

    @property
    def label(self) -> str:
        return f"{self.class_name}/{self.object_name}"


class TeachingStore:
    """Image dataset organized as class/object/*.jpg."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.manifest_path = self.root_dir / "manifest.jsonl"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def add_images(
        self,
        image_paths: Iterable[str | Path],
        *,
        class_name: str,
        object_name: str,
        source_event: dict | None = None,
    ) -> list[TeachingSample]:
        class_name, object_name = safe_label(class_name, object_name)
        target_dir = self.images_dir / class_name / object_name
        target_dir.mkdir(parents=True, exist_ok=True)

        samples: list[TeachingSample] = []
        for source in image_paths:
            source_path = Path(source)
            if not source_path.is_file():
                continue
            target_path = self._unique_target(target_dir, source_path.name)
            shutil.copy2(source_path, target_path)
            sample = TeachingSample(
                class_name=class_name,
                object_name=object_name,
                image_path=target_path,
                source_path=source_path,
                source_event=source_event,
            )
            self._append_manifest(sample)
            samples.append(sample)
        return samples

    def iter_labeled_images(self):
        for class_dir in sorted(self.images_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for object_dir in sorted(class_dir.iterdir()):
                if not object_dir.is_dir():
                    continue
                for path in sorted(object_dir.iterdir()):
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        yield class_dir.name, object_dir.name, path

    def _append_manifest(self, sample: TeachingSample) -> None:
        record = {
            "label": sample.label,
            "class_name": sample.class_name,
            "object_name": sample.object_name,
            "image_path": str(sample.image_path),
            "source_path": str(sample.source_path),
            "source_event": sample.source_event,
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _unique_target(self, target_dir: Path, filename: str) -> Path:
        stem = safe_segment(Path(filename).stem)
        suffix = Path(filename).suffix.lower() or ".jpg"
        candidate = target_dir / f"{stem}{suffix}"
        index = 1
        while candidate.exists():
            candidate = target_dir / f"{stem}_{index:03d}{suffix}"
            index += 1
        return candidate
