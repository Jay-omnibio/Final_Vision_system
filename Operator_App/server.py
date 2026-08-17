#!/usr/bin/env python
"""Local operator web app for runtime control, review, and teaching."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Operator_App.runtime_manager import RuntimeManager
from Operator_App.services.config_service import ConfigService
from Operator_App.services.event_service import EventService
from Operator_App.services.teaching_service import TeachingService

STATIC_ROOT = PROJECT_ROOT / "Operator_App" / "static"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "operator_events" / "events.jsonl"


class OperatorHandler(BaseHTTPRequestHandler):
    runtime: RuntimeManager
    config_service: ConfigService
    event_service: EventService
    teaching_service: TeachingService

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send_file(STATIC_ROOT / "index.html")
            elif path.startswith("/static/"):
                self._send_file(STATIC_ROOT / path.removeprefix("/static/"))
            elif path == "/api/status":
                payload = self.runtime.status()
                payload["configured_mode"] = self.config_service.runtime_mode()
                payload["debug_video_default"] = self.config_service.debug_video_default()
                self._send_json(payload)
            elif path == "/api/settings":
                self._send_json(self.config_service.settings())
            elif path == "/api/events":
                limit = self._query_int("limit", 100)
                self._send_json({"events": self.event_service.events(limit=limit)})
            elif path == "/api/unknowns":
                self._send_json({"groups": self.event_service.unknown_groups()})
            elif path == "/api/crop":
                crop_path = parse_qs(urlparse(self.path).query).get("path", [""])[0]
                self._send_file(self.event_service.resolve_crop(crop_path))
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self._send_error(exc)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/start":
                payload = self._read_json()
                debug_video = bool(payload.get("debug_video", self.config_service.debug_video_default()))
                self.runtime.start(mode=self.config_service.runtime_mode(), debug_video=debug_video)
                self._send_json(self.runtime.status())
            elif path == "/api/stop":
                self.runtime.stop()
                self._send_json(self.runtime.status())
            elif path == "/api/settings":
                self._send_json(self.config_service.update_settings(self._read_json()))
            elif path == "/api/assign":
                payload = self._read_json()
                crop_paths = [str(path) for path in payload.get("crop_paths", []) if str(path).strip()]
                events_by_crop = {path: self.event_service.event_by_crop(path) for path in crop_paths}
                result = self.teaching_service.assign_crops(
                    crop_paths,
                    class_name=str(payload.get("class_name", "")),
                    object_name=str(payload.get("object_name", "")),
                    events_by_crop=events_by_crop,
                )
                self._send_json(result)
            elif path == "/api/rebuild-gallery":
                self._send_json(self.teaching_service.rebuild_gallery())
            else:
                self.send_error(404, "Not found")
        except Exception as exc:
            self._send_error(exc)

    def log_message(self, format: str, *args) -> None:
        return

    def _query_int(self, name: str, default: int) -> int:
        value = parse_qs(urlparse(self.path).query).get(name, [str(default)])[0]
        try:
            return max(1, int(value))
        except ValueError:
            return default

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, exc: Exception) -> None:
        body = json.dumps({"error": str(exc)}).encode("utf-8")
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        body = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS_PATH))
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    events_path = Path(args.events)

    handler = OperatorHandler
    handler.config_service = ConfigService(
        config_path if config_path.is_absolute() else PROJECT_ROOT / config_path,
        project_root=PROJECT_ROOT,
    )
    handler.event_service = EventService(
        events_path if events_path.is_absolute() else PROJECT_ROOT / events_path,
        project_root=PROJECT_ROOT,
    )
    handler.teaching_service = TeachingService(
        project_root=PROJECT_ROOT,
        python_exe=args.python,
        config_path=handler.config_service.config_path,
    )
    handler.runtime = RuntimeManager(
        python_exe=args.python,
        config_path=handler.config_service.config_path,
        project_root=PROJECT_ROOT,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Operator app: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        handler.runtime.stop()
        server.server_close()


if __name__ == "__main__":
    main()
