"""Event and crop review helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote


class EventService:
    def __init__(self, events_path: Path, *, project_root: Path) -> None:
        self.events_path = events_path
        self.project_root = project_root.resolve()

    def events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not self.events_path.is_file():
            return []
        lines = self.events_path.read_text(encoding="utf-8", errors="replace").splitlines()
        rows: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(self._decorate_event(event))
        return rows

    def unknown_groups(self, *, limit: int = 500) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for event in self.events(limit=limit):
            group_name = self._unknown_group_name(event)
            if group_name is None:
                continue
            group = groups.setdefault(group_name, {"name": group_name, "count": 0, "events": [], "crops": []})
            group["count"] += 1
            group["events"].append(event)
            if event.get("crop_path"):
                group["crops"].append({"path": event["crop_path"], "url": event.get("crop_url")})
        return sorted(groups.values(), key=lambda item: item["name"])

    def event_by_crop(self, crop_path: str) -> dict[str, Any] | None:
        for event in self.events(limit=5000):
            if str(event.get("crop_path", "")) == crop_path:
                return event
        return None

    def resolve_crop(self, crop_path: str) -> Path:
        path = Path(crop_path)
        if not path.is_absolute():
            path = self.project_root / path
        resolved = path.resolve()
        if self.project_root not in resolved.parents and resolved != self.project_root:
            raise ValueError("Crop path is outside project")
        if not resolved.is_file():
            raise FileNotFoundError(f"Crop not found: {resolved}")
        return resolved

    def _decorate_event(self, event: dict[str, Any]) -> dict[str, Any]:
        item = dict(event)
        stable = f"{item.get('frame_index', '')}:{item.get('track_id', '')}:{item.get('crop_path', '')}"
        item["event_id"] = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:12]
        crop_path = item.get("crop_path")
        item["crop_url"] = f"/api/crop?path={quote(str(crop_path))}" if crop_path else None
        novelty = item.get("novelty") or {}
        item["status"] = novelty.get("status", "normal")
        item["new_group"] = novelty.get("new_group")
        return item

    def _unknown_group_name(self, event: dict[str, Any]) -> str | None:
        novelty = event.get("novelty") or {}
        if novelty.get("status") == "new":
            return str(novelty.get("new_group") or event.get("label") or "new")
        label = str(event.get("label", ""))
        if label.startswith("new_") or label == "unknown":
            return label
        return None
