"""Process harness for launching and controlling OpenApps visual env."""

from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import json
from pathlib import Path
from typing import Any

from envs.openapps_wrapper import OpenAppsWrapper, PreparedOpenAppsRuntime


class BrowserEnv:
    """Launches OpenApps from a prepared runtime config and manages teardown."""

    def __init__(
        self,
        runtime_root: str | Path = "data/runtime_openapps",
        project_root: str | Path | None = None,
    ):
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.wrapper = OpenAppsWrapper(runtime_root=runtime_root)
        self.process: subprocess.Popen[str] | None = None
        self.base_url: str | None = None
        self.prepared_runtime: PreparedOpenAppsRuntime | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stdout_history: list[str] = []
        self._observed_port: int | None = None

    def launch(
        self,
        scenario: dict[str, Any],
        run_id: str | None = None,
        timeout_seconds: float = 120.0,
        strict_state_mapping: bool = True,
    ) -> str:
        if self.process is not None:
            raise RuntimeError("BrowserEnv is already running. Call teardown() first.")

        self.prepared_runtime = self.wrapper.prepare_runtime(
            scenario=scenario,
            run_id=run_id,
            strict=strict_state_mapping,
        )
        command = [
            "uv",
            "run",
            "launch.py",
            "--config-path",
            str(self.prepared_runtime.config_dir),
            "--config-name",
            self.prepared_runtime.config_name,
            "use_wandb=False",
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._start_stdout_reader()
        self.base_url = self._wait_until_ready(timeout_seconds=timeout_seconds)
        return self.base_url

    def teardown(self, timeout_seconds: float = 20.0) -> None:
        process = self.process
        if process is None:
            return

        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.2)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5.0)

        self.process = None
        self.base_url = None
        self.prepared_runtime = None
        self._stop_stdout_reader()

    def get_recent_logs(self, num_lines: int = 200) -> list[str]:
        if num_lines <= 0:
            return []
        return self._stdout_history[-num_lines:]

    def get_current_state(self) -> dict[str, Any]:
        if self.base_url is None:
            raise RuntimeError("BrowserEnv is not running. Call launch() first.")
        base = self.base_url
        return {
            "todo": self._get_json_or_default(f"{base}/todo_all", []),
            "calendar": self._get_json_or_default(f"{base}/calendar_all", []),
            "map": self._get_json_or_default(f"{base}/maps/landmarks", []),
            "messenger": self._get_json_or_default(f"{base}/messages_all", []),
            "codeeditor": self._get_json_or_default(f"{base}/codeeditor_all", []),
            "online_shop": self._get_json_or_default(f"{base}/onlineshop_all", []),
        }

    def __enter__(self) -> "BrowserEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.teardown()

    def _start_stdout_reader(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        self._stdout_history = []
        self._stdout_queue = queue.Queue()
        self._observed_port = None

        def _reader() -> None:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                self._stdout_history.append(line)
                self._stdout_queue.put(line)

        self._stdout_thread = threading.Thread(target=_reader, daemon=True)
        self._stdout_thread.start()

    def _stop_stdout_reader(self) -> None:
        thread = self._stdout_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._stdout_thread = None

    def _wait_until_ready(self, timeout_seconds: float) -> str:
        assert self.process is not None
        deadline = time.time() + timeout_seconds
        candidate_url: str | None = None

        while time.time() < deadline:
            self._raise_if_process_exited()
            while True:
                try:
                    line = self._stdout_queue.get_nowait()
                except queue.Empty:
                    break
                found = self._extract_base_url(line)
                if found:
                    candidate_url = found
                found_port = self._extract_listen_port(line)
                if found_port is not None:
                    self._observed_port = found_port
            if candidate_url and self._healthcheck(candidate_url):
                return candidate_url
            if self._observed_port is not None:
                port_url = f"http://localhost:{self._observed_port}"
                if self._healthcheck(port_url):
                    return port_url
            time.sleep(0.4)

        debug_tail = "\n".join(self.get_recent_logs(60))
        raise TimeoutError(
            "Timed out waiting for OpenApps to become ready. "
            f"Last output:\n{debug_tail}"
        )

    def _raise_if_process_exited(self) -> None:
        assert self.process is not None
        rc = self.process.poll()
        if rc is None:
            return
        debug_tail = "\n".join(self.get_recent_logs(60))
        raise RuntimeError(
            f"OpenApps process exited early with code {rc}. "
            f"Last output:\n{debug_tail}"
        )

    def _extract_base_url(self, line: str) -> str | None:
        match = re.search(r"(http://localhost:\d+)", line)
        if not match:
            return None
        return match.group(1)

    def _extract_listen_port(self, line: str) -> int | None:
        match = re.search(r"Using port (\d+) for the web app", line)
        if not match:
            return None
        return int(match.group(1))

    def _healthcheck(self, base_url: str) -> bool:
        return self._is_http_ok(f"{base_url}/environment_variables")

    def _is_http_ok(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=0.15) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    def _get_json_or_default(self, url: str, default: Any) -> Any:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status != 200:
                    return default
                raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
        except Exception:
            return default
