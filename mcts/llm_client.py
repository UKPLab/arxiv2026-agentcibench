"""LiteLLM client utilities for provider-agnostic JSON responses."""

from __future__ import annotations

from collections import deque
from email.utils import parsedate_to_datetime
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    from litellm import completion
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    completion = None


_DEFAULT_MODELS = {
    "mutation": "openai/gpt-4o",
    "proxy": "openai/gpt-4o-mini",
    "judge": "openai/gpt-4o",
}

_MODEL_ENV_MAP = {
    "mutation": "AGENTCI_MUTATOR_MODEL",
    "proxy": "AGENTCI_PROXY_MODEL",
    "judge": "AGENTCI_JUDGE_MODEL",
}

_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass
class LLMCallStat:
    """One LiteLLM completion attempt with usage/cost metadata."""

    role: str
    operation: str
    model: str
    attempt: int
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    success: bool
    error: str | None = None


_CALL_STATS: list[LLMCallStat] = []
_RATE_LIMIT_LOCK = threading.Lock()
_GLOBAL_REQUEST_TIMESTAMPS: deque[float] = deque()
_MODEL_REQUEST_TIMESTAMPS: dict[str, deque[float]] = {}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truncate_error(value: Any, limit: int = 400) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_usage_stats(response: Any) -> tuple[int, int, int, float | None]:
    usage = _get_attr_or_key(response, "usage")
    prompt_tokens = _to_int(
        _get_attr_or_key(usage, "prompt_tokens", _get_attr_or_key(usage, "input_tokens", 0))
    )
    completion_tokens = _to_int(
        _get_attr_or_key(
            usage,
            "completion_tokens",
            _get_attr_or_key(usage, "output_tokens", 0),
        )
    )
    total_tokens = _to_int(_get_attr_or_key(usage, "total_tokens", 0))
    if total_tokens <= 0 and (prompt_tokens > 0 or completion_tokens > 0):
        total_tokens = prompt_tokens + completion_tokens

    cost_usd: float | None = _to_float(_get_attr_or_key(usage, "cost"))
    hidden = _get_attr_or_key(response, "_hidden_params")
    if isinstance(hidden, dict):
        if cost_usd is None:
            cost_usd = _to_float(hidden.get("response_cost"))
        if cost_usd is None:
            cost_usd = _to_float(hidden.get("cost"))
    if cost_usd is None:
        cost_usd = _to_float(_get_attr_or_key(response, "response_cost"))
    return prompt_tokens, completion_tokens, total_tokens, cost_usd


def _record_stat(
    *,
    role: str,
    operation: str,
    model: str,
    attempt: int,
    latency_seconds: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: float | None = None,
    success: bool,
    error: str | None = None,
) -> None:
    _CALL_STATS.append(
        LLMCallStat(
            role=role,
            operation=operation,
            model=model,
            attempt=attempt,
            latency_seconds=latency_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            success=success,
            error=error,
        )
    )


def reset_llm_usage_stats() -> None:
    """Clear in-memory LiteLLM call stats."""
    _CALL_STATS.clear()


def reset_llm_rate_limit_state() -> None:
    """Clear in-memory request timestamps used by proactive rate limiting."""
    with _RATE_LIMIT_LOCK:
        _GLOBAL_REQUEST_TIMESTAMPS.clear()
        _MODEL_REQUEST_TIMESTAMPS.clear()


def mark_llm_usage() -> int:
    """Return a cursor for calculating usage deltas."""
    return len(_CALL_STATS)


def _aggregate_stats(stats: list[LLMCallStat]) -> dict[str, Any]:
    total_calls = len(stats)
    succeeded = sum(1 for stat in stats if stat.success)
    failed = total_calls - succeeded
    prompt_tokens = sum(stat.prompt_tokens for stat in stats)
    completion_tokens = sum(stat.completion_tokens for stat in stats)
    total_tokens = sum(stat.total_tokens for stat in stats)
    latency_total = sum(stat.latency_seconds for stat in stats)

    cost_values = [stat.cost_usd for stat in stats if stat.cost_usd is not None]
    estimated_cost_usd = float(sum(cost_values)) if cost_values else None

    by_role: dict[str, dict[str, Any]] = {}
    for stat in stats:
        role_key = stat.role if stat.operation == "primary" else f"{stat.role}_repair"
        bucket = by_role.setdefault(
            role_key,
            {
                "calls_total": 0,
                "calls_succeeded": 0,
                "calls_failed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": None,
                "avg_latency_seconds": 0.0,
            },
        )
        bucket["calls_total"] += 1
        if stat.success:
            bucket["calls_succeeded"] += 1
        else:
            bucket["calls_failed"] += 1
        bucket["prompt_tokens"] += stat.prompt_tokens
        bucket["completion_tokens"] += stat.completion_tokens
        bucket["total_tokens"] += stat.total_tokens
        if stat.cost_usd is not None:
            prior = bucket["estimated_cost_usd"] or 0.0
            bucket["estimated_cost_usd"] = float(prior + stat.cost_usd)
        bucket["avg_latency_seconds"] += stat.latency_seconds

    for bucket in by_role.values():
        calls = bucket["calls_total"]
        bucket["avg_latency_seconds"] = bucket["avg_latency_seconds"] / calls if calls else 0.0

    return {
        "calls_total": total_calls,
        "calls_succeeded": succeeded,
        "calls_failed": failed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "avg_latency_seconds": latency_total / total_calls if total_calls else 0.0,
        "by_role": by_role,
    }


def llm_usage_since(mark: int) -> dict[str, Any]:
    """Summarize usage for calls made after a prior mark."""
    if mark < 0:
        mark = 0
    return _aggregate_stats(_CALL_STATS[mark:])


def get_llm_usage_summary() -> dict[str, Any]:
    """Summarize usage for all calls since the last reset."""
    return _aggregate_stats(_CALL_STATS)


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def heuristic_fallback_enabled() -> bool:
    """Whether local non-LLM fallback behavior is allowed."""
    return _is_true(os.getenv("AGENTCI_ALLOW_HEURISTIC_FALLBACK", "0"))


def resolve_model(role: str, override_model: str | None = None) -> str:
    """Resolve model name from explicit arg or environment defaults."""
    if override_model:
        return override_model
    if role not in _DEFAULT_MODELS:
        known = ", ".join(sorted(_DEFAULT_MODELS))
        raise ValueError(f"Unknown role '{role}'. Expected one of: {known}")
    return os.getenv(_MODEL_ENV_MAP[role], _DEFAULT_MODELS[role])


def _require_litellm() -> None:
    if completion is None:
        raise ImportError(
            "litellm is required but not installed. Install it with "
            "`uv add litellm` or `pip install litellm`."
        )


def _load_rpm_by_model() -> dict[str, int]:
    raw = os.getenv("AGENTCI_LITELLM_MAX_REQUESTS_PER_MINUTE_BY_MODEL_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    output: dict[str, int] = {}
    for key, value in parsed.items():
        model = str(key).strip()
        if not model:
            continue
        try:
            rpm = int(value)
        except (TypeError, ValueError):
            continue
        if rpm > 0:
            output[model] = rpm
    return output


def _resolve_max_requests_per_minute(model: str) -> tuple[int, int]:
    global_rpm = _env_int("AGENTCI_LITELLM_MAX_REQUESTS_PER_MINUTE", 0)
    model_rpm = _load_rpm_by_model().get(model, 0)
    return max(0, global_rpm), max(0, model_rpm)


def _prune_request_window(queue: deque[float], now: float, window_seconds: float) -> None:
    while queue and (now - queue[0]) >= window_seconds:
        queue.popleft()


def _apply_proactive_rate_limit(model: str) -> None:
    global_rpm, model_rpm = _resolve_max_requests_per_minute(model)
    if global_rpm <= 0 and model_rpm <= 0:
        return

    window_seconds = max(1.0, _env_float("AGENTCI_LITELLM_RATE_LIMIT_WINDOW_SECONDS", 60.0))
    safety_margin = max(0.0, _env_float("AGENTCI_LITELLM_RATE_LIMIT_SAFETY_MARGIN_SECONDS", 0.01))

    while True:
        sleep_for = 0.0
        with _RATE_LIMIT_LOCK:
            now = time.monotonic()
            _prune_request_window(_GLOBAL_REQUEST_TIMESTAMPS, now, window_seconds)
            model_queue = _MODEL_REQUEST_TIMESTAMPS.setdefault(model, deque())
            _prune_request_window(model_queue, now, window_seconds)

            if global_rpm > 0 and len(_GLOBAL_REQUEST_TIMESTAMPS) >= global_rpm:
                sleep_for = max(sleep_for, window_seconds - (now - _GLOBAL_REQUEST_TIMESTAMPS[0]))
            if model_rpm > 0 and len(model_queue) >= model_rpm:
                sleep_for = max(sleep_for, window_seconds - (now - model_queue[0]))

            if sleep_for <= 0:
                if global_rpm > 0:
                    _GLOBAL_REQUEST_TIMESTAMPS.append(now)
                if model_rpm > 0:
                    model_queue.append(now)
                return

        time.sleep(max(0.0, sleep_for + safety_margin))


def _load_extra_kwargs() -> dict[str, Any]:
    raw = os.getenv("AGENTCI_LITELLM_KWARGS_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("AGENTCI_LITELLM_KWARGS_JSON must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AGENTCI_LITELLM_KWARGS_JSON must decode to a JSON object.")
    return parsed


def _extract_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("LiteLLM response has no choices.")
    message = choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content)


def _extract_first_json_object(text: str) -> str | None:
    for start_idx, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for idx in range(start_idx, len(text)):
            current = text[idx]
            if escape:
                escape = False
                continue
            if current == "\\":
                escape = True
                continue
            if current == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : idx + 1]
    return None


def _parse_json_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty LLM response.")

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    if "```" in stripped:
        chunks = stripped.split("```")
        for chunk in chunks:
            candidate = chunk.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    obj_text = _extract_first_json_object(stripped)
    if obj_text is None:
        raise ValueError("Could not find JSON object in LLM response.")
    parsed = json.loads(obj_text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")
    return parsed


def _validate_candidate(
    candidate: dict[str, Any],
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if validator is None:
        return candidate
    validated = validator(candidate)
    if not isinstance(validated, dict):
        raise ValueError("Validator must return a JSON object.")
    return validated


def _completion_with_stats(
    *,
    role: str,
    operation: str,
    model: str,
    attempt: int,
    request_kwargs: dict[str, Any],
) -> Any:
    _apply_proactive_rate_limit(model)
    started = time.perf_counter()
    try:
        response = completion(**request_kwargs)
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - started
        _record_stat(
            role=role,
            operation=operation,
            model=model,
            attempt=attempt,
            latency_seconds=latency,
            success=False,
            error=_truncate_error(exc),
        )
        raise

    latency = time.perf_counter() - started
    prompt_tokens, completion_tokens, total_tokens, cost_usd = _extract_usage_stats(response)
    _record_stat(
        role=role,
        operation=operation,
        model=model,
        attempt=attempt,
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        success=True,
    )
    return response


def _extract_http_status_code(exc: Exception) -> int | None:
    direct_status = _to_int(
        _get_attr_or_key(
            exc,
            "status_code",
            _get_attr_or_key(exc, "status", _get_attr_or_key(exc, "http_status")),
        ),
        default=0,
    )
    if direct_status > 0:
        return direct_status

    response = _get_attr_or_key(exc, "response")
    if response is not None:
        response_status = _to_int(_get_attr_or_key(response, "status_code"), default=0)
        if response_status > 0:
            return response_status
    return None


def _parse_retry_after_value(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))

    value = str(raw).strip()
    if not value:
        return None

    as_float = _to_float(value)
    if as_float is not None:
        return max(0.0, as_float)

    try:
        retry_at = parsedate_to_datetime(value)
        delta_seconds = retry_at.timestamp() - time.time()
        return max(0.0, delta_seconds)
    except Exception:  # noqa: BLE001
        return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    for key in ("retry_after", "retry_after_seconds", "retry_after_s"):
        parsed = _parse_retry_after_value(_get_attr_or_key(exc, key))
        if parsed is not None:
            return parsed

    response = _get_attr_or_key(exc, "response")
    headers = _get_attr_or_key(response, "headers")
    header_items: Any = []
    if headers is not None:
        if hasattr(headers, "items"):
            header_items = headers.items()
        elif isinstance(headers, dict):
            header_items = headers.items()
    for raw_key, raw_value in header_items:
        if str(raw_key).strip().lower() != "retry-after":
            continue
        parsed = _parse_retry_after_value(raw_value)
        if parsed is not None:
            return parsed

    text = str(exc).lower()
    patterns = (
        r"retry\s+after\s+([0-9]+(?:\.[0-9]+)?)",
        r"try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|second|seconds)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        parsed = _to_float(match.group(1))
        if parsed is not None:
            return max(0.0, parsed)
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = _extract_http_status_code(exc)
    if status_code == 429:
        return True

    text = str(exc).lower()
    indicators = (
        "rate limit",
        "too many requests",
        "resource_exhausted",
        "resource exhausted",
        "quota exceeded",
        "requests per minute",
        "tokens per minute",
        "ratelimit",
        "http 429",
    )
    return any(indicator in text for indicator in indicators)


def _resolve_retry_count(retries: int | None) -> int:
    if retries is None:
        return max(0, _env_int("AGENTCI_LITELLM_RETRIES", 2))
    return max(0, retries)


def _compute_retry_sleep_seconds(exc: Exception, attempt: int) -> float:
    base_seconds = max(0.1, _env_float("AGENTCI_LITELLM_BACKOFF_BASE_SECONDS", 1.5))
    jitter_seconds = max(0.0, _env_float("AGENTCI_LITELLM_BACKOFF_JITTER_SECONDS", 0.25))
    max_sleep_seconds = max(0.1, _env_float("AGENTCI_LITELLM_BACKOFF_MAX_SECONDS", 90.0))

    if _is_rate_limit_error(exc):
        retry_after = _extract_retry_after_seconds(exc)
        if retry_after is not None:
            sleep_for = retry_after
        else:
            sleep_for = base_seconds * (2**attempt)
    else:
        sleep_for = base_seconds * (attempt + 1)

    if jitter_seconds > 0:
        sleep_for += random.uniform(0.0, jitter_seconds)
    return min(max(0.0, sleep_for), max_sleep_seconds)


def _repair_json_with_llm(
    *,
    role: str,
    model: str,
    raw_response_text: str,
    schema_hint: str,
    validation_error: Exception,
    attempt: int,
) -> dict[str, Any]:
    raw_response_text = raw_response_text.strip()
    if len(raw_response_text) > 7000:
        raw_response_text = f"{raw_response_text[:7000]}..."

    repair_prompt = (
        "Fix this malformed JSON output so it becomes one valid JSON object. "
        "Do not add markdown or explanations.\n\n"
        f"Schema hint:\n{schema_hint.strip() or '{}'}\n\n"
        f"Validation error:\n{_truncate_error(validation_error)}\n\n"
        f"Original output:\n{raw_response_text}"
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON repair tool. "
                    "Return exactly one valid JSON object and nothing else."
                ),
            },
            {"role": "user", "content": repair_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
        "drop_params": True,
        "timeout": float(os.getenv("AGENTCI_LITELLM_TIMEOUT_SECONDS", "120")),
        "response_format": {"type": "json_object"},
    }
    request_kwargs.update(_load_extra_kwargs())

    response = _completion_with_stats(
        role=role,
        operation="repair",
        model=model,
        attempt=attempt,
        request_kwargs=request_kwargs,
    )
    content = _extract_message_text(response)
    return _parse_json_output(content)


def call_llm_json(
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
    override_model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    retries: int | None = None,
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    schema_hint: str = "{}",
    repair_attempts: int = 1,
) -> dict[str, Any]:
    """Call LiteLLM and parse a JSON object response."""
    _require_litellm()
    model = resolve_model(role, override_model=override_model)

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "drop_params": True,
        "timeout": float(os.getenv("AGENTCI_LITELLM_TIMEOUT_SECONDS", "120")),
        "response_format": {"type": "json_object"},
    }
    request_kwargs.update(_load_extra_kwargs())

    last_error: Exception | None = None
    retry_count = _resolve_retry_count(retries)
    for attempt in range(retry_count + 1):
        try:
            response = _completion_with_stats(
                role=role,
                operation="primary",
                model=model,
                attempt=attempt,
                request_kwargs=request_kwargs,
            )
            content = _extract_message_text(response)
            try:
                parsed = _parse_json_output(content)
                return _validate_candidate(parsed, validator=validator)
            except Exception as parse_or_validation_error:  # noqa: BLE001
                if repair_attempts <= 0:
                    raise
                repair_error: Exception | None = None
                for repair_attempt in range(repair_attempts):
                    try:
                        repaired = _repair_json_with_llm(
                            role=role,
                            model=model,
                            raw_response_text=content,
                            schema_hint=schema_hint,
                            validation_error=parse_or_validation_error,
                            attempt=repair_attempt,
                        )
                        return _validate_candidate(repaired, validator=validator)
                    except Exception as exc:  # noqa: BLE001
                        repair_error = exc
                if repair_error is not None:
                    raise RuntimeError(
                        f"JSON repair failed after {repair_attempts} attempt(s): {repair_error}"
                    ) from parse_or_validation_error
                raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retry_count:
                break
            time.sleep(_compute_retry_sleep_seconds(exc, attempt))

    raise RuntimeError(f"LiteLLM call failed for role='{role}', model='{model}': {last_error}")
