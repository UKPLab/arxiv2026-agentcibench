"""OpenApps-supported scenario app constraints for AgentCI generation."""

from __future__ import annotations

import copy
from typing import Any


# Scenario-level app keys allowed in `initial_states`.
SUPPORTED_OPENAPPS_INITIAL_STATE_APPS: tuple[str, ...] = (
    "open_todo",
    "open_messenger",
    "open_calendar",
    "open_maps",
    "open_shop",
    "open_code_editor",
)


# Common LLM alias spellings observed in generation outputs.
_APP_KEY_ALIASES: dict[str, str] = {
    "open_todos": "open_todo",
    "open_messages": "open_messenger",
    "open_message": "open_messenger",
    "open_map": "open_maps",
    "open_shop_app": "open_shop",
    "open_store": "open_shop",
    "open_codeeditor": "open_code_editor",
}

_SUPPORTED_APP_SET = set(SUPPORTED_OPENAPPS_INITIAL_STATE_APPS)


def _normalize_app_key(app_name: str) -> str:
    key = str(app_name).strip().lower()
    if key in _APP_KEY_ALIASES:
        return _APP_KEY_ALIASES[key]
    return key


def sanitize_initial_states(
    initial_states: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Keep only supported OpenApps app keys.

    Returns:
      - sanitized mapping with normalized supported keys
      - dropped unsupported original keys
    """
    if not isinstance(initial_states, dict):
        return {}, []

    sanitized: dict[str, Any] = {}
    dropped: list[str] = []
    for app_name, app_state in initial_states.items():
        normalized_key = _normalize_app_key(str(app_name))
        if normalized_key not in _SUPPORTED_APP_SET:
            dropped.append(str(app_name))
            continue
        if not isinstance(app_state, dict):
            # Keep pipeline strict to per-app mapping states.
            dropped.append(str(app_name))
            continue
        sanitized[normalized_key] = copy.deepcopy(app_state)
    return sanitized, dropped


def merge_supported_initial_states(
    parent_initial_states: dict[str, Any],
    candidate_initial_states: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Merge candidate states onto parent states while enforcing supported app keys.

    Unsupported apps in candidate are dropped; supported parent apps are preserved
    unless overridden by supported candidate apps.
    """
    parent_sanitized, _ = sanitize_initial_states(parent_initial_states)
    candidate_sanitized, dropped = sanitize_initial_states(candidate_initial_states)
    merged = copy.deepcopy(parent_sanitized)
    for app_name, app_state in candidate_sanitized.items():
        merged[app_name] = app_state
    return merged, dropped

