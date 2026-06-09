"""Scenario-to-OpenApps bridge for visual benchmark execution."""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


SUPPORTED_INITIAL_STATE_APPS: tuple[str, ...] = (
    "open_todo",
    "open_messenger",
    "open_calendar",
    "open_maps",
    "open_shop",
    "open_code_editor",
)


@dataclass(frozen=True)
class PreparedOpenAppsRuntime:
    """Runtime config artifact ready to pass to `launch.py`."""

    config_path: Path
    config_name: str
    config_dir: Path
    logs_dir: Path
    databases_dir: Path
    warnings: list[str]
    onlineshop_enabled: bool


def _utc_now_compact() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-") or "run"


def _normalize_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


class OpenAppsWrapper:
    """Builds Hydra config files from AgentCI scenario initial states."""

    def __init__(self, runtime_root: str | Path = "data/runtime_openapps"):
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._config_root = Path(__file__).resolve().parents[1] / "config"

    def prepare_runtime(
        self,
        scenario: dict[str, Any],
        run_id: str | None = None,
        strict: bool = True,
    ) -> PreparedOpenAppsRuntime:
        if not isinstance(scenario, dict):
            raise ValueError("Scenario must be a mapping.")
        scenario_id = str(scenario.get("scenario_id", "")).strip() or "scenario"
        initial_states = scenario.get("initial_states", {})
        if not isinstance(initial_states, dict):
            raise ValueError("Scenario initial_states must be a mapping.")

        run_name = run_id or f"{_utc_now_compact()}_{_slugify(scenario_id)}"
        run_dir = self.runtime_root / run_name
        logs_dir = run_dir / "logs"
        databases_dir = logs_dir / "databases"
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        databases_dir.mkdir(parents=True, exist_ok=True)

        config, warnings, onlineshop_enabled = self._build_config(
            initial_states=initial_states,
            logs_dir=logs_dir,
            databases_dir=databases_dir,
            strict=strict,
        )

        config_path = config_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            OmegaConf.save(config=config, f=handle, resolve=True)

        meta_path = run_dir / "bridge_metadata.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "scenario_id": scenario_id,
                    "run_name": run_name,
                    "warnings": warnings,
                    "supported_apps": list(SUPPORTED_INITIAL_STATE_APPS),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        return PreparedOpenAppsRuntime(
            config_path=config_path,
            config_name=config_path.name,
            config_dir=config_dir,
            logs_dir=logs_dir,
            databases_dir=databases_dir,
            warnings=warnings,
            onlineshop_enabled=onlineshop_enabled,
        )

    def _build_config(
        self,
        initial_states: dict[str, Any],
        logs_dir: Path,
        databases_dir: Path,
        strict: bool,
    ) -> tuple[DictConfig, list[str], bool]:
        overrides = [
            f"logs_dir={logs_dir}",
            f"databases_dir={databases_dir}",
            "use_wandb=False",
        ]
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(
            version_base=None,
            config_dir=str(self._config_root),
        ):
            config = compose(config_name="config", overrides=overrides)

        warnings: list[str] = []
        unsupported = sorted(
            set(initial_states.keys()) - set(SUPPORTED_INITIAL_STATE_APPS)
        )
        if unsupported and strict:
            raise ValueError(
                "Unsupported initial_states app keys: "
                + ", ".join(unsupported)
                + "."
            )
        if unsupported:
            warnings.append("Dropped unsupported apps: " + ", ".join(unsupported))

        if "open_todo" in initial_states:
            config.apps.todo.init_todos = self._convert_todo(initial_states["open_todo"])
        if "open_messenger" in initial_states:
            users, chat_history = self._convert_messenger(initial_states["open_messenger"])
            config.apps.messenger.users = users
            config.apps.messenger.chat_history = chat_history
        if "open_calendar" in initial_states:
            config.apps.calendar.events = self._convert_calendar(initial_states["open_calendar"])
        if "open_maps" in initial_states:
            maps_payload, map_warnings = self._convert_maps(initial_states["open_maps"])
            config.apps.maps.saved_places = maps_payload["saved_places"]
            if maps_payload.get("init_location") is not None:
                config.apps.maps.init_location = maps_payload["init_location"]
            warnings.extend(map_warnings)
        if "open_code_editor" in initial_states:
            config.apps.code_editor.filesystem = self._convert_code_editor(
                initial_states["open_code_editor"]
            )
        onlineshop_enabled = False
        if "open_shop" in initial_states:
            onlineshop_enabled = True
            shop_payload, shop_warnings = self._convert_shop(initial_states["open_shop"])
            config.apps.onlineshop.enable = True
            if "cart" in shop_payload:
                config.apps.onlineshop.cart = shop_payload["cart"]
            if "orders" in shop_payload:
                config.apps.onlineshop.orders = shop_payload["orders"]
            if "additional_info_to_item" in shop_payload:
                config.apps.onlineshop.additional_info_to_item = shop_payload[
                    "additional_info_to_item"
                ]
            warnings.extend(shop_warnings)

        return config, warnings, onlineshop_enabled

    def _convert_todo(self, todo_state: Any) -> list[list[Any]]:
        if not isinstance(todo_state, dict):
            raise ValueError("open_todo state must be a mapping.")
        # Scenarios may use "items" or "todo" as the list key.
        raw_items = todo_state.get("items") or todo_state.get("todo")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("open_todo must have a non-empty 'items' or 'todo' list.")
        converted: list[list[Any]] = []
        for item in raw_items:
            if isinstance(item, str):
                title = item.strip()
                if title:
                    converted.append([title, False])
                continue
            if isinstance(item, dict):
                # Accept "title", "content", or "text" as the task name.
                title = (
                    str(item.get("title", "")).strip()
                    or str(item.get("content", "")).strip()
                    or str(item.get("text", "")).strip()
                )
                if not title:
                    continue
                done = bool(item.get("done", item.get("status", "") == "done"))
                converted.append([title, done])
                continue
            title = str(item).strip()
            if title:
                converted.append([title, False])
        if not converted:
            raise ValueError("open_todo produced an empty task list after normalization.")
        return converted

    def _convert_messenger(
        self, messenger_state: Any
    ) -> tuple[list[str], dict[str, list[list[Any]]]]:
        if not isinstance(messenger_state, dict):
            raise ValueError("open_messenger state must be a mapping.")
        threads = messenger_state.get("threads")
        if not isinstance(threads, list) or not threads:
            raise ValueError("open_messenger.threads must be a non-empty list.")

        users: list[str] = []
        history: dict[str, list[list[Any]]] = {}
        default_time = dt.datetime.now().strftime("%b %d, %I:%M%p")
        for idx, thread in enumerate(threads):
            if not isinstance(thread, dict):
                raise ValueError(f"open_messenger.threads[{idx}] must be a mapping.")
            # Scenarios may use "contact" or "name" as the thread identifier.
            contact = str(thread.get("contact", thread.get("name", ""))).strip()
            if not contact:
                raise ValueError(f"open_messenger.threads[{idx}] missing non-empty contact/name.")
            users.append(contact)
            entries: list[list[Any]] = []
            messages = thread.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError(
                    f"open_messenger.threads[{idx}].messages must be a list."
                )
            for msg in messages:
                if isinstance(msg, str):
                    text = msg.strip()
                    if text:
                        entries.append([text, False, contact, default_time])
                    continue
                if isinstance(msg, dict):
                    text = str(msg.get("text", msg.get("message", ""))).strip()
                    if not text:
                        continue
                    sender = str(msg.get("sender", contact)).strip() or contact
                    timestamp = str(msg.get("timestamp", default_time)).strip() or default_time
                    entries.append([text, False, sender, timestamp])
                    continue
                text = str(msg).strip()
                if text:
                    entries.append([text, False, contact, default_time])
            history[contact] = entries
        return users, history

    def _convert_calendar(self, calendar_state: Any) -> list[dict[str, Any]]:
        if not isinstance(calendar_state, dict):
            raise ValueError("open_calendar state must be a mapping.")
        raw_events = calendar_state.get("events")
        if raw_events is None:
            return []
        if not isinstance(raw_events, list):
            raise ValueError("open_calendar.events must be a list.")
        converted: list[dict[str, Any]] = []
        today = dt.date.today().isoformat()
        for idx, raw in enumerate(raw_events):
            if isinstance(raw, str):
                # Scenario stored event as a plain string title.
                title = raw.strip()
                if title:
                    converted.append({"title": title, "date": today, "description": ""})
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"open_calendar.events[{idx}] must be a mapping or string.")
            title = str(raw.get("title", "")).strip()
            if not title:
                continue
            date_value = str(raw.get("date", today)).strip() or today
            event = {
                "title": title,
                "date": date_value,
                "description": str(raw.get("description", "")).strip(),
                "url": str(raw.get("url", "")).strip() or None,
                "location": str(raw.get("location", "")).strip() or None,
            }
            invitees = raw.get("invitees")
            if isinstance(invitees, list):
                event["invitees"] = _normalize_strings(invitees) or None
            elif isinstance(invitees, str):
                items = [item.strip() for item in invitees.split(",")]
                event["invitees"] = _normalize_strings(items) or None
            else:
                event["invitees"] = None
            converted.append(event)
        return converted

    def _convert_maps(
        self, maps_state: Any
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(maps_state, dict):
            raise ValueError("open_maps state must be a mapping.")
        warnings: list[str] = []
        saved_places: list[dict[str, Any]] = []

        if isinstance(maps_state.get("saved_places"), list):
            for idx, place in enumerate(maps_state["saved_places"]):
                if not isinstance(place, dict):
                    raise ValueError(f"open_maps.saved_places[{idx}] must be a mapping.")
                name = str(place.get("name", "")).strip()
                if not name:
                    continue
                lat = float(place.get("lat", place.get("latitude", 40.7831)))
                lng = float(place.get("lng", place.get("longitude", -73.9712)))
                saved_places.append({"name": name, "lat": lat, "lng": lng})

        recent_searches = maps_state.get("recent_searches")
        if isinstance(recent_searches, list):
            for idx, term in enumerate(_normalize_strings(recent_searches)):
                saved_places.append(
                    {
                        "name": term,
                        "lat": 40.70 + (idx * 0.01),
                        "lng": -73.99 + (idx * 0.01),
                    }
                )
            warnings.append(
                "Mapped open_maps.recent_searches to maps.saved_places for OpenApps compatibility."
            )

        init_location = maps_state.get("init_location")
        if not isinstance(init_location, list) or len(init_location) != 2:
            init_location = [40.7831, -73.9712]

        if not saved_places:
            route_target = str(
                maps_state.get("current_route", maps_state.get("destination", "Destination"))
            ).strip()
            if route_target:
                saved_places.append(
                    {"name": route_target, "lat": 40.7580, "lng": -73.9855}
                )
                warnings.append(
                    "Mapped open_maps current route/destination into one saved place entry."
                )

        return {"saved_places": saved_places, "init_location": init_location}, warnings

    def _convert_code_editor(self, editor_state: Any) -> list[dict[str, Any]]:
        if not isinstance(editor_state, dict):
            raise ValueError("open_code_editor state must be a mapping.")
        if isinstance(editor_state.get("filesystem"), list):
            return editor_state["filesystem"]
        filename = str(editor_state.get("filename", "scenario_file.txt")).strip()
        if not filename:
            filename = "scenario_file.txt"
        content = str(editor_state.get("content", ""))
        return [{"type": "file", "name": filename, "content": content}]

    def _convert_shop(self, shop_state: Any) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(shop_state, dict):
            raise ValueError("open_shop state must be a mapping.")
        warnings: list[str] = []
        payload: dict[str, Any] = {}

        if isinstance(shop_state.get("cart"), list):
            payload["cart"] = shop_state["cart"]
        if isinstance(shop_state.get("orders"), list):
            payload["orders"] = shop_state["orders"]
        products = shop_state.get("products")
        if isinstance(products, list):
            normalized = _normalize_strings(products)
            if normalized:
                payload["additional_info_to_item"] = (
                    "Scenario seeded products: " + ", ".join(normalized)
                )
                warnings.append(
                    "Mapped open_shop.products into onlineshop.additional_info_to_item."
                )
        return payload, warnings
