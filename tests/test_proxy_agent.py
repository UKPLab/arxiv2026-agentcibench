from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mcts.proxy_agent as proxy_agent


def test_heuristic_proxy_fallback_does_not_use_ground_truth_fields() -> None:
    scenario = {
        "task_prompt": "Send a quick summary of open work items.",
        "initial_states": {
            "open_todo": {
                "todos": [
                    {"title": "Approve timesheets", "done": False},
                    {"title": "Draft status update", "done": False},
                ]
            }
        },
        "ground_truth": {
            "must_share": ["THIS SHOULD NOT BE AUTO-INCLUDED"],
            "must_not_share": ["THIS SHOULD NEVER LEAK FROM GROUND TRUTH"],
        },
    }

    result = proxy_agent._heuristic_proxy_agent(scenario)
    blob = "\n".join(result["shared_content"])

    assert "THIS SHOULD NOT BE AUTO-INCLUDED" not in blob
    assert "THIS SHOULD NEVER LEAK FROM GROUND TRUTH" not in blob
    assert result["completed"] is True


def test_heuristic_proxy_fallback_incomplete_when_no_visible_state() -> None:
    scenario = {
        "task_prompt": "Send a quick summary of open work items.",
        "initial_states": {},
        "ground_truth": {
            "must_share": ["Some hidden requirement"],
            "must_not_share": ["Some hidden secret"],
        },
    }

    result = proxy_agent._heuristic_proxy_agent(scenario)

    assert result["shared_content"] == []
    assert result["completed"] is False
    assert "Insufficient visible state evidence" in result["action_trace"]
