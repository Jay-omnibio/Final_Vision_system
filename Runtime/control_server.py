#!/usr/bin/env python
"""Small local control panel for the headless vision runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "operator_events" / "events.jsonl"


INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Final Vision Control</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }
    button { margin: 4px; padding: 8px 12px; border-radius: 6px; border: 0; cursor: pointer; }
    textarea { width: 100%; height: 360px; background: #1e1e1e; color: #eee; border: 1px solid #444; padding: 12px; }
    pre { background: #1e1e1e; padding: 12px; overflow: auto; border: 1px solid #444; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .ok { color: #69db7c; }
    .bad { color: #ff8787; }
  </style>
</head>
<body>
  <h1>Final Vision Control</h1>
  <p>Status: <span id="status">loading...</span></p>
  <label><input id="debugVideo" type="checkbox" /> Save debug video</label>
  <br />
  <button onclick="startRuntime()">Start System</button>
  <button onclick="stopRuntime()">Stop System</button>
  <button onclick="loadConfig()">Reload Config</button>
  <button onclick="saveConfig()">Save Config</button>
  <button onclick="loadEvents()">Refresh Events</button>
  <div class="row">
    <section>
      <h2>Config YAML</h2>
      <textarea id="config"></textarea>
    </section>
    <section>
      <h2>Latest Events</h2>
      <pre id="events"></pre>
    </section>
  </div>
<script>
async function api(path, options) {
  const res = await fetch(path, options || {});
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
async function status() {
  const data = await api('/api/status');
  const el = document.getElementById('status');
  el.textContent = data.running ? `running pid ${data.pid}` : 'stopped';
  el.className = data.running ? 'ok' : 'bad';
}
async function loadConfig() {
  const data = await api('/api/config');
  document.getElementById('config').value = data.yaml;
}
async function saveConfig() {
  const yaml = document.getElementById('config').value;
  await api('/api/config', {method: 'POST', body: JSON.stringify({yaml})});
  await status();
}
async function startRuntime() {
  const debug_video = document.getElementById('debugVideo').checked;
  await api('/api/start', {method: 'POST', body: JSON.stringify({debug_video})});
  await status();
}
async function stopRuntime() {
  await api('/api/stop', {method: 'POST'});
  await status();
}
async function loadEvents() {
  const data = await api('/api/events');
  document.getElementById('events').textContent = data.events.map(e =>
    `${e.frame_index || '-'} | ID ${e.track_id || '-'} | ${e.label || '-'} | ${Number(e.score || 0).toFixed(3)}`
  ).join('\n');
}
loadConfig(); status(); loadEvents(); setInterval(() => { status(); loadEvents(); }, 3000);
</script>
</body>
</html>"""


class RuntimeManager:
    def __init__(self, *, python_exe: str, config_path: Path, project_root: Path) -> None:
        self.python_exe = python_exe
        self.config_path = config_path
        self.project_root = project_root
        self.stop_file = project_root / "data" / "runtime" / "stop_live_vision.flag"
        self.process: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, *, debug_video: bool = False) -> None:
        if self.is_running():
            return
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        if self.stop_file.exists():
            self.stop_file.unlink()
        command = [
            self.python_exe,
            "Runtime/run_live_vision.py",
            "--config",
            str(self.config_path),
            "--stop-file",
            str(self.stop_file),
        ]
        if debug_video:
            command.append("--debug-video")
        self.process = subprocess.Popen(
            command,
            cwd=self.project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if not self.is_running():
            self.process = None
            return
        assert self.process is not None
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.write_text("stop", encoding="utf-8")
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=8)
        finally:
            if self.stop_file.exists():
                self.stop_file.unlink()
        self.process = None

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "pid": self.process.pid if self.is_running() and self.process else None,
        }


class ControlHandler(BaseHTTPRequestHandler):
    manager: RuntimeManager
    config_path: Path
    events_path: Path

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(INDEX_HTML)
        elif path == "/api/status":
            self._send_json(self.manager.status())
        elif path == "/api/config":
            self._send_json({"yaml": self.config_path.read_text(encoding="utf-8")})
        elif path == "/api/events":
            self._send_json({"events": self._read_events(limit=100)})
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/start":
            payload = self._read_json()
            self.manager.start(debug_video=bool(payload.get("debug_video", False)))
            self._send_json(self.manager.status())
        elif path == "/api/stop":
            self.manager.stop()
            self._send_json(self.manager.status())
        elif path == "/api/config":
            payload = self._read_json()
            yaml_text = str(payload.get("yaml", ""))
            yaml.safe_load(yaml_text)
            self.config_path.write_text(yaml_text, encoding="utf-8")
            self._send_json({"saved": True})
        else:
            self.send_error(404, "Not found")

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _read_events(self, *, limit: int) -> list[dict]:
        if not self.events_path.is_file():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()[-limit:]
        events = []
        for line in lines:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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
    handler_class = ControlHandler
    handler_class.config_path = Path(args.config)
    handler_class.events_path = Path(args.events)
    handler_class.manager = RuntimeManager(
        python_exe=args.python,
        config_path=handler_class.config_path,
        project_root=PROJECT_ROOT,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    print(f"Control panel: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        handler_class.manager.stop()
        server.server_close()


if __name__ == "__main__":
    main()
