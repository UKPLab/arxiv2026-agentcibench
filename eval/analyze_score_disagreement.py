"""Analyze disagreement between deterministic and LLM judge score tracks."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DisagreementRow:
    source_file: str
    scenario_id: str
    utility_det: int | None
    utility_llm: int | None
    ci_det: int | None
    ci_llm: int | None
    reward_det: float | None
    reward_llm: float | None
    utility_diff: bool
    ci_diff: bool
    reward_diff: bool


def _iter_input_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            paths.append(p)
            continue
        if p.is_dir():
            paths.extend(sorted(p.rglob("benchmark_results__*.jsonl")))
            paths.extend(sorted(p.rglob("run_result.json")))
            continue
        # Try a glob pattern.
        parent = p.parent if str(p.parent) != "." else Path(".")
        paths.extend(sorted(parent.glob(p.name)))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_score_blob(record: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    # text benchmark record
    if "score" in record and isinstance(record.get("score"), dict):
        scenario_id = str(record.get("scenario_id", "unknown"))
        return scenario_id, dict(record["score"])
    # visual run_result record
    if "judge_score" in record and isinstance(record.get("judge_score"), dict):
        scenario_id = str(record.get("scenario_id", "unknown"))
        return scenario_id, dict(record["judge_score"])
    return None


def _extract_dual_values(score: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None, float | None, float | None]:
    utility_det = _to_int_or_none(score.get("utility_deterministic"))
    utility_llm = _to_int_or_none(score.get("utility_llm"))
    ci_det = _to_int_or_none(score.get("ci_violation_deterministic"))
    ci_llm = _to_int_or_none(score.get("ci_violation_llm"))
    reward_det = _to_float_or_none(score.get("reward_deterministic"))
    reward_llm = _to_float_or_none(score.get("reward_llm"))

    det_blob = score.get("deterministic_score")
    if isinstance(det_blob, dict):
        if utility_det is None:
            utility_det = _to_int_or_none(det_blob.get("utility"))
        if ci_det is None:
            ci_det = _to_int_or_none(det_blob.get("ci_violation"))
        if reward_det is None:
            reward_det = _to_float_or_none(det_blob.get("reward"))

    llm_blob = score.get("llm_score")
    if isinstance(llm_blob, dict):
        if utility_llm is None:
            utility_llm = _to_int_or_none(llm_blob.get("utility"))
        if ci_llm is None:
            ci_llm = _to_int_or_none(llm_blob.get("ci_violation"))
        if reward_llm is None:
            reward_llm = _to_float_or_none(llm_blob.get("reward"))

    return utility_det, utility_llm, ci_det, ci_llm, reward_det, reward_llm


def _reward_diff(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        records.append(json.loads(text))
    return records


def _parse_json(path: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8"))]


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _parse_jsonl(path)
    if path.suffix == ".json":
        return _parse_json(path)
    return []


def analyze(paths: list[Path]) -> dict[str, Any]:
    rows: list[DisagreementRow] = []
    total_records = 0
    missing_dual = 0

    for path in paths:
        records = _load_records(path)
        for record in records:
            total_records += 1
            extracted = _extract_score_blob(record)
            if extracted is None:
                missing_dual += 1
                continue
            scenario_id, score = extracted
            utility_det, utility_llm, ci_det, ci_llm, reward_det, reward_llm = _extract_dual_values(score)
            if (
                utility_det is None
                or utility_llm is None
                or ci_det is None
                or ci_llm is None
                or reward_det is None
                or reward_llm is None
            ):
                missing_dual += 1
                continue

            rows.append(
                DisagreementRow(
                    source_file=str(path),
                    scenario_id=scenario_id,
                    utility_det=utility_det,
                    utility_llm=utility_llm,
                    ci_det=ci_det,
                    ci_llm=ci_llm,
                    reward_det=reward_det,
                    reward_llm=reward_llm,
                    utility_diff=(utility_det != utility_llm),
                    ci_diff=(ci_det != ci_llm),
                    reward_diff=_reward_diff(reward_det, reward_llm),
                )
            )

    compared = len(rows)
    utility_disagreements = sum(1 for row in rows if row.utility_diff)
    ci_disagreements = sum(1 for row in rows if row.ci_diff)
    reward_disagreements = sum(1 for row in rows if row.reward_diff)

    mean_abs_reward_gap = (
        sum(abs((row.reward_det or 0.0) - (row.reward_llm or 0.0)) for row in rows) / compared
        if compared
        else 0.0
    )

    return {
        "total_records_seen": total_records,
        "records_with_dual_scores": compared,
        "records_missing_dual_scores": missing_dual,
        "dual_score_coverage": round((compared / total_records), 4) if total_records else 0.0,
        "utility_disagreement_count": utility_disagreements,
        "utility_disagreement_rate": round((utility_disagreements / compared), 4) if compared else 0.0,
        "ci_disagreement_count": ci_disagreements,
        "ci_disagreement_rate": round((ci_disagreements / compared), 4) if compared else 0.0,
        "reward_disagreement_count": reward_disagreements,
        "reward_disagreement_rate": round((reward_disagreements / compared), 4) if compared else 0.0,
        "mean_abs_reward_gap": round(mean_abs_reward_gap, 4),
        "rows": [
            {
                "source_file": row.source_file,
                "scenario_id": row.scenario_id,
                "utility_deterministic": row.utility_det,
                "utility_llm": row.utility_llm,
                "ci_violation_deterministic": row.ci_det,
                "ci_violation_llm": row.ci_llm,
                "reward_deterministic": row.reward_det,
                "reward_llm": row.reward_llm,
                "utility_diff": row.utility_diff,
                "ci_diff": row.ci_diff,
                "reward_diff": row.reward_diff,
            }
            for row in rows
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze deterministic-vs-LLM scorer disagreement for benchmark result files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Files/directories/globs to scan. Supports benchmark_results__*.jsonl "
            "and visual run_result.json."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write full JSON report.",
    )
    parser.add_argument(
        "--show-rows",
        action="store_true",
        help="Print per-scenario rows to stdout.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = _iter_input_paths(args.inputs)
    report = analyze(paths)

    summary = {
        key: value
        for key, value in report.items()
        if key != "rows"
    }
    print(json.dumps(summary, indent=2))
    if args.show_rows:
        print(json.dumps(report["rows"], indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
