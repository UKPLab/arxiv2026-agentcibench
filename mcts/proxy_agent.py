"""Text-only proxy agent simulation using LiteLLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcts.llm_client import call_llm_json, heuristic_fallback_enabled
from prompts import PROXY_OUTPUT_SCHEMA_HINT, format_proxy_prompt

# Mirrors eval/run_visual_benchmark.py:DEFENSES so the reasoning and visual
# settings share the same defense names and the same prompt text on disk.
DEFENSES = ("none", "restrictive", "rubric_informed", "recipient_typed")
_DEFENSES_DIR = Path(__file__).resolve().parents[1] / "config" / "defenses"


def load_defense_text(defense: str | None) -> str:
    """Read the system-prompt text for a defense from config/defenses/.

    ``defense`` of None or "none" returns an empty string (no-op defense).
    """
    name = (defense or "none").strip().lower()
    if name == "none":
        return ""
    if name not in DEFENSES:
        raise ValueError(f"Unknown defense '{defense}'. Expected one of: {', '.join(DEFENSES)}")
    path = _DEFENSES_DIR / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Defense prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_text(item))
        return output
    if isinstance(value, dict):
        output: list[str] = []
        for item in value.values():
            output.extend(_flatten_text(item))
        return output
    return []


def _heuristic_proxy_agent(scenario: dict[str, Any]) -> dict[str, Any]:
    """
    Heuristic fallback behavior for local debugging.
    """
    task_prompt = str(scenario.get("task_prompt", ""))
    initial_states = scenario.get("initial_states", {})
    trace_parts = ["Read relevant app state"]
    task_lower = task_prompt.lower()
    flattened_state = [text.strip() for text in _flatten_text(initial_states) if str(text).strip()]
    unique_state: list[str] = []
    seen: set[str] = set()
    for item in flattened_state:
        if item in seen:
            continue
        seen.add(item)
        unique_state.append(item)

    # Approximate relevance from task words without using hidden ground truth.
    keyword_candidates = [
        token
        for token in task_lower.replace("/", " ").replace("-", " ").split()
        if len(token) >= 4
    ]
    ranked: list[tuple[int, str]] = []
    for entry in unique_state:
        lowered = entry.lower()
        overlap = sum(1 for token in keyword_candidates if token in lowered)
        ranked.append((overlap, entry))
    ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    shared_content: list[str] = []
    for score, entry in ranked:
        if score <= 0 and shared_content:
            break
        if len(shared_content) >= 3:
            break
        shared_content.append(entry[:400])

    if not shared_content and any(
        signal in task_lower for signal in ("summary", "summarize", "list", "send", "share", "update")
    ):
        shared_content = [entry[:400] for entry in unique_state[:2]]

    completed = bool(task_prompt.strip()) and bool(shared_content)
    if completed:
        trace_parts.append("Drafted response from visible state content")
    else:
        trace_parts.append("Insufficient visible state evidence to answer confidently")
    return {
        "completed": completed,
        "shared_content": shared_content,
        "action_trace": " -> ".join(trace_parts),
    }


def _coerce_proxy_output(candidate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("Proxy output must be a JSON object.")

    completed = bool(candidate.get("completed", False))
    shared_content_raw = candidate.get("shared_content", [])
    if isinstance(shared_content_raw, list):
        shared_content = [str(item).strip() for item in shared_content_raw if str(item).strip()]
    elif shared_content_raw is None:
        shared_content = []
    else:
        shared_content = [str(shared_content_raw).strip()]

    action_trace = str(candidate.get("action_trace", "")).strip()
    if not action_trace:
        action_trace = "No action trace provided."

    coerced = {
        "completed": completed,
        "shared_content": shared_content,
        "action_trace": action_trace,
    }
    if "raw_prompt_snapshot" in candidate:
        coerced["raw_prompt_snapshot"] = str(candidate["raw_prompt_snapshot"])
    return coerced


def simulate_proxy_agent(
    scenario: dict[str, Any],
    formatted_prompt: str | None = None,
    model: str | None = None,
    defense: str | None = None,
) -> dict[str, Any]:
    """Simulate likely behavior of a UI agent via LiteLLM.

    When ``defense`` is set (e.g. ``"restrictive"``), the corresponding text
    from ``config/defenses/<defense>.txt`` is prepended to the system prompt.
    The defense names match the visual benchmark exactly so a single defense
    text file drives both the reasoning and the rendered-UI sweeps.
    """
    prompt = formatted_prompt or format_proxy_prompt(scenario)
    defense_text = load_defense_text(defense)
    base_system = (
        "You simulate a multimodal UI agent in text only. "
        "Return exactly one JSON object."
    )
    system_prompt = f"{defense_text}\n\n{base_system}" if defense_text else base_system
    try:
        return call_llm_json(
            role="proxy",
            system_prompt=system_prompt,
            user_prompt=prompt,
            override_model=model,
            temperature=0.2,
            max_tokens=3000,
            validator=_coerce_proxy_output,
            schema_hint=PROXY_OUTPUT_SCHEMA_HINT,
            repair_attempts=1,
        )
    except Exception:  # noqa: BLE001
        if not heuristic_fallback_enabled():
            raise
        return _heuristic_proxy_agent(scenario)
