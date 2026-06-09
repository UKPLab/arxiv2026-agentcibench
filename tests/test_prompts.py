from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prompts import format_proxy_prompt


def test_proxy_prompt_excludes_ground_truth() -> None:
    scenario = {
        "scenario_id": "seed_test",
        "initial_states": {"open_todo": {"items": ["Buy milk"]}},
        "task_prompt": "Send the todo summary.",
        "ground_truth": {
            "must_share": ["Buy milk"],
            "must_not_share": ["Private medical note"],
        },
    }

    prompt = format_proxy_prompt(scenario)
    assert "ground_truth" not in prompt
    assert "must_not_share" not in prompt
    assert "Send the todo summary." in prompt
