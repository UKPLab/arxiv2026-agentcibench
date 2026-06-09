from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mcts import llm_client  # noqa: E402


def _fake_response(json_content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json_content))],
        usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    )


def test_retry_uses_retry_after_for_rate_limit_errors(monkeypatch) -> None:
    class DummyRateLimitError(Exception):
        pass

    err = DummyRateLimitError("429 too many requests")
    err.status_code = 429
    err.response = SimpleNamespace(status_code=429, headers={"Retry-After": "2.5"})

    call_count = {"value": 0}

    def fake_completion(**_: object) -> SimpleNamespace:
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise err
        return _fake_response('{"ok": true}')

    sleep_calls: list[float] = []

    monkeypatch.setattr(llm_client, "completion", fake_completion)
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setenv("AGENTCI_LITELLM_BACKOFF_JITTER_SECONDS", "0")
    monkeypatch.setenv("AGENTCI_LITELLM_MAX_REQUESTS_PER_MINUTE", "0")
    monkeypatch.delenv("AGENTCI_LITELLM_MAX_REQUESTS_PER_MINUTE_BY_MODEL_JSON", raising=False)
    llm_client.reset_llm_rate_limit_state()

    result = llm_client.call_llm_json(
        role="proxy",
        system_prompt="system",
        user_prompt="user",
        override_model="test/model",
        retries=1,
        repair_attempts=0,
    )

    assert result == {"ok": True}
    assert call_count["value"] == 2
    assert sleep_calls == [2.5]


def test_proactive_rpm_limiter_waits_for_next_window(monkeypatch) -> None:
    clock = {"value": 0.0}
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        return clock["value"]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        clock["value"] += seconds

    monkeypatch.setattr(llm_client.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(llm_client.time, "sleep", fake_sleep)
    monkeypatch.setenv("AGENTCI_LITELLM_MAX_REQUESTS_PER_MINUTE", "1")
    monkeypatch.delenv("AGENTCI_LITELLM_MAX_REQUESTS_PER_MINUTE_BY_MODEL_JSON", raising=False)
    monkeypatch.setenv("AGENTCI_LITELLM_RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("AGENTCI_LITELLM_RATE_LIMIT_SAFETY_MARGIN_SECONDS", "0")
    llm_client.reset_llm_rate_limit_state()

    llm_client._apply_proactive_rate_limit("test/model")
    llm_client._apply_proactive_rate_limit("test/model")

    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 60.0


def test_compute_retry_backoff_scales_for_rate_limits(monkeypatch) -> None:
    class DummyRateLimitError(Exception):
        pass

    class DummyOtherError(Exception):
        pass

    monkeypatch.setenv("AGENTCI_LITELLM_BACKOFF_BASE_SECONDS", "1.5")
    monkeypatch.setenv("AGENTCI_LITELLM_BACKOFF_JITTER_SECONDS", "0")
    monkeypatch.setenv("AGENTCI_LITELLM_BACKOFF_MAX_SECONDS", "120")

    rate_err = DummyRateLimitError("rate limit exceeded")
    other_err = DummyOtherError("connection reset")

    rate_backoff_attempt_2 = llm_client._compute_retry_sleep_seconds(rate_err, attempt=2)
    other_backoff_attempt_2 = llm_client._compute_retry_sleep_seconds(other_err, attempt=2)

    assert rate_backoff_attempt_2 == 6.0
    assert other_backoff_attempt_2 == 4.5
