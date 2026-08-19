"""Runtime process management for the operator app."""

from __future__ import annotations

import subprocess
from pathlib import Path


class RuntimeManager:
    def __init__(self, *, python_exe: str, config_path: Path, project_root: Path) -> None:
        self.python_exe = python_exe
        self.config_path = config_path
        self.project_root = project_root
        self.stop_file = project_root / "data" / "runtime" / "stop_live_vision.flag"
        self.logs_dir = project_root / "outputs" / "runtime_logs"
        self.current_mode = "stopped"
        self.current_log_path: Path | None = None
        self.last_exit_code: int | None = None
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def is_running(self) -> bool:
        if self.process is None:
            return False
        exit_code = self.process.poll()
        if exit_code is None:
            return True
        self.last_exit_code = exit_code
        self.process = None
        self._close_log()
        return False

    def start(self, *, mode: str, debug_video: bool = False) -> None:
        if self.is_running():
            return

        mode = "novelty" if mode == "novelty" else "normal"
        script = "Runtime/run_live_novelty.py" if mode == "novelty" else "Runtime/run_live_vision.py"

        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        if self.stop_file.exists():
            self.stop_file.unlink()

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._close_log()
        self.current_mode = mode
        self.last_exit_code = None
        self.current_log_path = self.logs_dir / f"{mode}_latest.log"
        self._log_handle = self.current_log_path.open("w", encoding="utf-8")

        command = [
            self.python_exe,
            script,
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
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )

    def stop(self) -> None:
        if not self.is_running():
            self.process = None
            self._close_log()
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
            self._close_log()
            self.last_exit_code = self.process.returncode
            self.process = None

    def status(self) -> dict:
        running = self.is_running()
        return {
            "running": running,
            "pid": self.process.pid if running and self.process else None,
            "mode": self.current_mode,
            "last_exit_code": self.last_exit_code,
            "log_path": str(self.current_log_path) if self.current_log_path else None,
            "log_tail": self.log_tail(),
        }

    def log_tail(self, *, max_chars: int = 6000) -> str:
        if not self.current_log_path or not self.current_log_path.is_file():
            return ""
        try:
            text = self.current_log_path.read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:]
        except OSError:
            return ""

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
