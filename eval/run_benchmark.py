"""Run a local benchmark over generated scenarios using a dummy text agent."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from eval.metrics import score_output, summarize
from mcts.mcts_engine import validate_scenario_schema
from mcts.proxy_agent import simulate_proxy_agent
from prompts import format_proxy_prompt

try:  # pragma: no cover - optional UX dependency
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback path tested indirectly
    tqdm = None


class DummyTextAgent:
    """Simple stand-in for a UI agent that delegates to the proxy model."""

    def __init__(self, proxy_model: str | None = None, defense: str | None = None):
        self.proxy_model = proxy_model
        self.defense = defense

    def run(self, formatted_prompt: str, scenario: dict[str, Any]) -> dict[str, Any]:
        return simulate_proxy_agent(
            scenario,
            formatted_prompt=formatted_prompt,
            model=self.proxy_model,
            defense=self.defense,
        )


def _iter_scenario_files(generated_dir: Path) -> list[Path]:
    if not generated_dir.exists():
        return []
    # Skip manifest/index files that live alongside scenario JSONs in eval set dirs.
    return sorted(
        p for p in generated_dir.glob("*.json")
        if not p.name.startswith("manifest") and "manifest" not in p.stem.lower()
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _slug(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip("-._")
    return text or fallback


def _build_run_id(
    generated_path: Path,
    proxy_model: str | None,
    judge_model: str | None,
    num_scenarios: int,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source_tag = _slug(generated_path.name, "generated")
    proxy_tag = _slug(proxy_model, "default")
    judge_tag = _slug(judge_model, "default")
    return (
        f"{timestamp}__src-{source_tag}__n-{num_scenarios}"
        f"__proxy-{proxy_tag}__judge-{judge_tag}"
    )


def _iter_with_progress(
    scenario_files: Sequence[Path], show_progress: bool = True
) -> Iterable[Path]:
    total = len(scenario_files)
    if not show_progress:
        return scenario_files
    if tqdm is not None:
        return tqdm(scenario_files, desc="Benchmark", unit="scenario")

    milestone = max(1, total // 20)

    def _generator() -> Iterable[Path]:
        for idx, scenario_file in enumerate(scenario_files, start=1):
            if idx == 1 or idx % milestone == 0 or idx == total:
                print(f"[Benchmark] scenario {idx}/{total}")
            yield scenario_file

    return _generator()


def _extract_scenario_metadata(scenario: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("track", "source", "scenario_family", "failure_type"):
        value = scenario.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def run_benchmark(
    generated_dir: str | Path,
    results_dir: str | Path,
    proxy_model: str | None = None,
    judge_model: str | None = None,
    show_progress: bool = True,
    defense: str | None = None,
) -> dict[str, Any]:
    generated_path = Path(generated_dir)
    if not generated_path.exists():
        fallback = Path("generated")
        if fallback.exists():
            generated_path = fallback
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    scenario_files = _iter_scenario_files(generated_path)
    agent = DummyTextAgent(proxy_model=proxy_model, defense=defense)
    run_records: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    track_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    error_count = 0

    run_id = _build_run_id(
        generated_path=generated_path,
        proxy_model=proxy_model,
        judge_model=judge_model,
        num_scenarios=len(scenario_files),
    )
    results_file = results_path / f"benchmark_results__{run_id}.jsonl"
    summary_file = results_path / f"summary__{run_id}.json"

    # Open the JSONL file once and flush after each scenario so partial results
    # survive a crash or slurm preemption.
    with results_file.open("w", encoding="utf-8") as jsonl_handle:
        for scenario_file in _iter_with_progress(scenario_files, show_progress=show_progress):
            raw_scenario = _load_json(scenario_file)
            scenario = validate_scenario_schema(raw_scenario)
            scenario_metadata = _extract_scenario_metadata(scenario)
            if "track" in scenario_metadata:
                track_counts[str(scenario_metadata["track"])] += 1
            if "scenario_family" in scenario_metadata:
                family_counts[str(scenario_metadata["scenario_family"])] += 1
            if "source" in scenario_metadata:
                source_counts[str(scenario_metadata["source"])] += 1

            try:
                prompt = format_proxy_prompt(scenario)
                output = agent.run(prompt, scenario)
                score = score_output(scenario, output, judge_model=judge_model)
                record: dict[str, Any] = {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_file": str(scenario_file),
                    "proxy_model": proxy_model,
                    "judge_model": judge_model,
                    "defense": defense or "none",
                    **scenario_metadata,
                    "formatted_prompt": prompt,
                    "agent_output": output,
                    "score": score,
                }
                scores.append(score)
            except Exception as exc:  # noqa: BLE001
                error_count += 1
                print(
                    f"[run_benchmark] ERROR scenario={scenario['scenario_id']}: {exc}",
                    file=sys.stderr,
                )
                record = {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_file": str(scenario_file),
                    "proxy_model": proxy_model,
                    "judge_model": judge_model,
                    "defense": defense or "none",
                    **scenario_metadata,
                    "error": str(exc),
                    "score": {"utility": 0, "ci_violation": 1, "leaked_items": [], "reward": 0.0},
                }

            run_records.append(record)
            jsonl_handle.write(json.dumps(record, sort_keys=True))
            jsonl_handle.write("\n")
            jsonl_handle.flush()

    summary = summarize(scores)
    summary_with_meta: dict[str, Any] = {
        **summary,
        "run_id": run_id,
        "generated_dir": str(generated_path),
        "proxy_model": proxy_model,
        "judge_model": judge_model,
        "defense": defense or "none",
        "results_file": str(results_file),
        "summary_file": str(summary_file),
        "num_scenarios": len(scenario_files),
        "num_errors": error_count,
        "track_counts": dict(track_counts),
        "scenario_family_counts": dict(family_counts),
        "source_counts": dict(source_counts),
    }
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary_with_meta, handle, indent=2)
        handle.write("\n")
    if error_count:
        print(
            f"[run_benchmark] WARNING: {error_count}/{len(scenario_files)} scenarios failed; "
            f"check {results_file} for 'error' fields.",
            file=sys.stderr,
        )
    return {**summary_with_meta, "records": run_records}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dummy benchmark over generated scenarios.")
    parser.add_argument(
        "--generated-dir",
        default="data/generated",
        help="Directory containing generated scenario JSON files.",
    )
    parser.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory where benchmark outputs are written.",
    )
    parser.add_argument(
        "--proxy-model",
        default=None,
        help="LiteLLM model override for proxy simulation.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="LiteLLM model override for judge scoring.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display while evaluating scenarios.",
    )
    parser.add_argument(
        "--defense",
        default="none",
        choices=("none", "restrictive", "rubric_informed", "recipient_typed"),
        help="System-prompt defense from config/defenses/<name>.txt.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_benchmark(
        args.generated_dir,
        args.results_dir,
        proxy_model=args.proxy_model,
        judge_model=args.judge_model,
        show_progress=not args.no_progress,
        defense=args.defense,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
