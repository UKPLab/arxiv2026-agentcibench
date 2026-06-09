from __future__ import annotations

from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from envs.browser_env import BrowserEnv
from envs.openapps_wrapper import OpenAppsWrapper


def test_openapps_wrapper_prepare_runtime_maps_states(tmp_path: Path) -> None:
    wrapper = OpenAppsWrapper(runtime_root=tmp_path / "runtime")
    scenario = {
        "scenario_id": "seed_test",
        "initial_states": {
            "open_todo": {"items": ["Buy milk", {"title": "Paper towels", "done": True}]},
            "open_messenger": {
                "threads": [{"contact": "Alex", "messages": ["Need milk and towels"]}]
            },
            "open_calendar": {"events": [{"title": "Dentist", "date": "2026-04-01"}]},
            "open_maps": {
                "current_route": "Starbucks",
                "recent_searches": ["Divorce Attorney"],
            },
            "open_code_editor": {"filename": "index.html", "content": "<h1>Hello</h1>"},
        },
    }

    prepared = wrapper.prepare_runtime(scenario, run_id="test_run")

    assert prepared.config_path.exists()
    assert prepared.logs_dir.exists()
    assert prepared.databases_dir.exists()

    config = OmegaConf.load(prepared.config_path)
    assert config.apps.todo.init_todos[0][0] == "Buy milk"
    assert config.apps.todo.init_todos[1][1] is True
    assert "Alex" in list(config.apps.messenger.users)
    assert config.apps.messenger.chat_history.Alex[0][0] == "Need milk and towels"
    assert config.apps.calendar.events[0].title == "Dentist"
    assert len(config.apps.maps.saved_places) >= 1
    assert config.apps.code_editor.filesystem[0].name == "index.html"


def test_openapps_wrapper_rejects_unsupported_apps_when_strict(tmp_path: Path) -> None:
    wrapper = OpenAppsWrapper(runtime_root=tmp_path / "runtime")
    scenario = {
        "scenario_id": "bad_seed",
        "initial_states": {"open_unknown_app": {"x": 1}},
    }
    with pytest.raises(ValueError, match="Unsupported initial_states app keys"):
        wrapper.prepare_runtime(scenario, strict=True)


def test_browser_env_extracts_base_url(tmp_path: Path) -> None:
    env = BrowserEnv(runtime_root=tmp_path / "runtime")
    line = "Web app hostname is:  http://localhost:5412"
    assert env._extract_base_url(line) == "http://localhost:5412"
    port_line = "Using port 5412 for the web app"
    assert env._extract_listen_port(port_line) == 5412
