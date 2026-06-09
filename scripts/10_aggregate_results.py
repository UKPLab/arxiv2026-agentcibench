#!/usr/bin/env python3
"""Aggregate every benchmark run into the tables and CSVs used by the paper.

Walks the results tree, normalizes per-scenario records from the text
(reasoning) and visual benchmarks into one schema, and emits:

  data/results/aggregated/main_table.csv
      one row per (setting, access_mode, model). utility_rate, leak_rate,
      avg_ci_violation, n. The headline numbers for Table 1.

  data/results/aggregated/per_family.csv
      one row per (setting, access_mode, model, scenario_family).

  data/results/aggregated/per_mode.csv
      one row per (setting, access_mode, model, failure_mode). The per-mode
      breakdown referenced in section 5.1 of the paper.

  data/results/aggregated/access_mode_pivot.csv
      one row per model with columns for each access mode. Drives the
      axtree-vs-screenshot table.

  data/results/aggregated/defenses_pivot.csv
      one row per model with columns for each defense. Drives the
      defenses Pareto plot.

  data/results/aggregated/confusion.csv
      one row per (setting, access_mode, model). Counts of
      (utility, leaked) cells: completed_clean / completed_leak /
      incomplete_clean / incomplete_leak.

  data/results/aggregated/text_vs_visual_agreement.csv
      one row per model with the per-scenario agreement between the
      reasoning and the ui_only setting (overlap n, leak agreement,
      Cohen's kappa).

  data/results/aggregated/reasoning_pareto.csv
      one row per model from the reasoning setting only, for the
      utility-vs-leakage scatter plot.

  data/results/aggregated/report.md
      A short human-readable summary referencing every table above.

Inputs (all optional - the script ignores missing dirs):

  data/results/text/<model_slug>/benchmark_results__*.jsonl
  data/results/visual_main/runs/<run_id>/ui_only/<model>/<scen>/run_result.json
  data/results/visual_main/runs/<run_id>/<mode>/<model>/benchmark_results*.jsonl
  data/results/visual_defenses/<defense>/runs/...

Usage:

  uv run python scripts/10_aggregate_results.py
  uv run python scripts/10_aggregate_results.py \
      --results-root data/results --out-dir data/results/aggregated \
      --scenario-dirs data/generated data/seeds data/eval_set_main
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Normalized record schema
# --------------------------------------------------------------------------- #


@dataclass
class Record:
    """A single (scenario, agent run) outcome, normalized across settings."""

    setting: str               # "reasoning" | "visual"
    access_mode: str           # "text" (reasoning), "ui_only", "axtree_only", "mixed"
    defense: str               # "none" | "restrictive" | "rubric_informed" | "recipient_typed"
    model: str                 # raw LiteLLM model string
    scenario_id: str
    scenario_family: str       # mcts mutation family, e.g. "semantic_entanglement"
    failure_mode: str          # CI failure mode, e.g. "visual_co_location"
    utility: int               # 0/1
    leaked: int                # 0/1
    refused: int               # 0/1: agent declined (completed=False, no shared_content)
    num_leaked: int            # count of items in leaked_items
    ci_violation: int          # 1..5
    reward: float
    judge_model: str | None
    source_file: str           # provenance for debugging


# Mirrors mcts/mutators.py:MUTATION_STRATEGY_TO_FAILURE_MODE so the aggregator
# can repair records that have one tag but not the other.
FAMILY_TO_MODE = {
    "semantic_entanglement": "visual_co_location",
    "ambiguity_trap": "task_ambiguity_overshare",
    "identity_bleed": "recipient_misalignment",
}
MODE_TO_FAMILY = {v: k for k, v in FAMILY_TO_MODE.items()}


# --------------------------------------------------------------------------- #
# Scenario metadata lookup
# --------------------------------------------------------------------------- #


def build_scenario_index(scenario_dirs: list[Path]) -> dict[str, dict[str, str]]:
    """Scan scenario JSONs and return scenario_id -> {family, failure_mode}."""
    index: dict[str, dict[str, str]] = {}
    for root in scenario_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name.startswith("manifest"):
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            sid = data.get("scenario_id")
            if not sid:
                continue
            family = str(data.get("scenario_family") or "").lower()
            mode = str(data.get("failure_mode") or data.get("failure_type") or "").lower()
            if family and not mode:
                mode = FAMILY_TO_MODE.get(family, "")
            if mode and not family:
                family = MODE_TO_FAMILY.get(mode, "")
            index[sid] = {"scenario_family": family, "failure_mode": mode}
    return index


def lookup_family_mode(
    scenario_id: str,
    record: dict[str, Any],
    score: dict[str, Any] | None,
    index: dict[str, dict[str, str]],
) -> tuple[str, str]:
    family = str(record.get("scenario_family") or "").lower()
    mode = str(record.get("failure_mode") or record.get("failure_type") or "").lower()
    if score is not None:
        family = family or str(score.get("scenario_family") or "").lower()
        mode = mode or str(score.get("failure_mode") or "").lower()
    if scenario_id in index:
        family = family or index[scenario_id]["scenario_family"]
        mode = mode or index[scenario_id]["failure_mode"]
    if family and not mode:
        mode = FAMILY_TO_MODE.get(family, "")
    if mode and not family:
        family = MODE_TO_FAMILY.get(mode, "")
    return family or "unknown", mode or "unknown"


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_text_jsonl(
    path: Path,
    index: dict[str, dict[str, str]],
    defense_override: str | None = None,
) -> Iterator[Record]:
    """eval/run_benchmark.py emits records with a nested `score` dict."""
    for row in _iter_jsonl(path):
        score = row.get("score") or {}
        utility = int(bool(score.get("utility", 0)))
        leaked_items = score.get("leaked_items") or []
        leaked = 1 if leaked_items else 0
        ci_violation = int(score.get("ci_violation", 1))
        reward = float(score.get("reward", 0.0))
        scenario_id = str(row.get("scenario_id", ""))
        family, mode = lookup_family_mode(scenario_id, row, score, index)
        # Refusal: agent declared the task incomplete AND shared nothing.
        # Mechanically equivalent to "did not engage with the task at all".
        agent_out = row.get("agent_output") or {}
        completed_flag = bool(agent_out.get("completed", False))
        shared = agent_out.get("shared_content") or []
        refused = 1 if (not completed_flag and not shared) else 0
        # Defense provenance: prefer the record's own field (added by
        # run_benchmark.py once defenses became a reasoning-side feature);
        # fall back to the directory-derived override; default to "none".
        defense = str(
            row.get("defense") or defense_override or "none"
        ).strip().lower()
        yield Record(
            setting="reasoning",
            access_mode="text",
            defense=defense,
            model=str(row.get("proxy_model") or "unknown"),
            scenario_id=scenario_id,
            scenario_family=family,
            failure_mode=mode,
            utility=utility,
            leaked=leaked,
            refused=refused,
            num_leaked=len(leaked_items),
            ci_violation=ci_violation,
            reward=reward,
            judge_model=row.get("judge_model"),
            source_file=str(path),
        )


def parse_visual_jsonl(
    path: Path,
    index: dict[str, dict[str, str]],
    defense: str = "none",
) -> Iterator[Record]:
    """eval/run_visual_benchmark.py emits flat per-run records."""
    for row in _iter_jsonl(path):
        leaked_items = row.get("leaked_items") or []
        utility = int(bool(row.get("utility", 0)))
        leaked = 1 if leaked_items else 0
        ci_violation = int(row.get("ci_violation", 1))
        reward = float(row.get("reward", 0.0))
        scenario_id = str(row.get("scenario_id", ""))
        family, mode = lookup_family_mode(scenario_id, row, None, index)
        # Visual refusal: completion_assessment says not completed AND no
        # outbound shared_content was produced through the UI.
        ca = row.get("completion_assessment") or {}
        completed_flag = bool(ca.get("completed", False))
        shared = ca.get("shared_content") or []
        refused = 1 if (not completed_flag and not shared) else 0
        yield Record(
            setting="visual",
            access_mode=str(row.get("access_mode") or "ui_only"),
            defense=defense,
            model=str(row.get("model_name") or row.get("model_pretty_name") or "unknown"),
            scenario_id=scenario_id,
            scenario_family=family,
            failure_mode=mode,
            utility=utility,
            leaked=leaked,
            refused=refused,
            num_leaked=len(leaked_items),
            ci_violation=ci_violation,
            reward=reward,
            judge_model=row.get("judge_model"),
            source_file=str(path),
        )


def discover_text_records(
    results_root: Path, index: dict[str, dict[str, str]]
) -> Iterator[Record]:
    # No-defense reasoning sweep (scripts/02_text_benchmark.sh).
    text_root = results_root / "text"
    if text_root.exists():
        for jsonl in text_root.rglob("benchmark_results__*.jsonl"):
            yield from parse_text_jsonl(jsonl, index)

    # Reasoning defenses sweep (scripts/07_text_defenses.sh). Layout:
    #   data/results/text_defenses/<defense>/<model_slug>/benchmark_results*.jsonl
    # The defense name is the first path component; the record's own
    # `defense` field, written by run_benchmark.py, takes precedence and
    # the override only kicks in for older jsonls that pre-date that field.
    text_def_root = results_root / "text_defenses"
    if text_def_root.exists():
        for defense_dir in sorted(p for p in text_def_root.iterdir() if p.is_dir()):
            defense = defense_dir.name.lower()
            for jsonl in defense_dir.rglob("benchmark_results__*.jsonl"):
                yield from parse_text_jsonl(jsonl, index, defense_override=defense)


def discover_visual_records(
    results_root: Path, index: dict[str, dict[str, str]]
) -> Iterator[Record]:
    # Visual E2E runs (mixed access mode is the current standard).
    for subdir in ("visual_mixed", "visual_main", "visual_pilot"):
        root = results_root / subdir
        if not root.exists():
            continue
        for jsonl in root.rglob("benchmark_results__*.jsonl"):
            yield from parse_visual_jsonl(jsonl, index, defense="none")

    # Legacy visual_defenses tree (kept for back-compat with runs already on
    # disk; scripts/07 no longer emits into here).
    defenses_root = results_root / "visual_defenses"
    if defenses_root.exists():
        for defense_dir in sorted(p for p in defenses_root.iterdir() if p.is_dir()):
            defense = defense_dir.name
            for jsonl in defense_dir.rglob("benchmark_results__*.jsonl"):
                yield from parse_visual_jsonl(jsonl, index, defense=defense)


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #


@dataclass
class Bucket:
    """Running counters for a single (setting, model, ...) cell."""

    n: int = 0
    utility_sum: int = 0
    leaked_sum: int = 0
    refused_sum: int = 0
    ci_violation_sum: int = 0
    num_leaked_sum: int = 0
    reward_sum: float = 0.0
    completed_clean: int = 0    # utility=1, leaked=0
    completed_leak: int = 0     # utility=1, leaked=1
    incomplete_clean: int = 0   # utility=0, leaked=0
    incomplete_leak: int = 0    # utility=0, leaked=1

    def add(self, record: Record) -> None:
        self.n += 1
        self.utility_sum += record.utility
        self.leaked_sum += record.leaked
        self.refused_sum += record.refused
        self.ci_violation_sum += record.ci_violation
        self.num_leaked_sum += record.num_leaked
        self.reward_sum += record.reward
        if record.utility == 1 and record.leaked == 0:
            self.completed_clean += 1
        elif record.utility == 1 and record.leaked == 1:
            self.completed_leak += 1
        elif record.utility == 0 and record.leaked == 0:
            self.incomplete_clean += 1
        else:
            self.incomplete_leak += 1

    def metrics(self) -> dict[str, float]:
        if self.n == 0:
            return {
                "n": 0,
                "utility_rate": 0.0,
                "leak_rate": 0.0,
                "refusal_rate": 0.0,
                "engagement_n": 0,
                "engagement_leak_rate": 0.0,
                "avg_ci_violation": 0.0,
                "avg_num_leaked": 0.0,
                "avg_reward": 0.0,
            }
        # Engagement-conditional leak: leak rate over scenarios where the agent
        # did NOT refuse. Mathematically a refusal can't leak (shared_content is
        # empty), so leaked_sum is already over the engagement subset.
        engagement_n = self.n - self.refused_sum
        engagement_leak_rate = (
            round(self.leaked_sum / engagement_n, 4) if engagement_n > 0 else 0.0
        )
        return {
            "n": self.n,
            "utility_rate": round(self.utility_sum / self.n, 4),
            "leak_rate": round(self.leaked_sum / self.n, 4),
            "refusal_rate": round(self.refused_sum / self.n, 4),
            "engagement_n": engagement_n,
            "engagement_leak_rate": engagement_leak_rate,
            "avg_ci_violation": round(self.ci_violation_sum / self.n, 4),
            "avg_num_leaked": round(self.num_leaked_sum / self.n, 4),
            "avg_reward": round(self.reward_sum / self.n, 4),
        }


def bucketize(records: Iterable[Record], key) -> dict[Any, Bucket]:
    buckets: dict[Any, Bucket] = defaultdict(Bucket)
    for record in records:
        buckets[key(record)].add(record)
    return buckets


# --------------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------------- #


def _slug(value: str | None, fallback: str = "unknown") -> str:
    if not value:
        return fallback
    text = re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-._")
    return text or fallback


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _baseline_records(records: list[Record]) -> list[Record]:
    """Records eligible for headline (non-defenses) views.

    Keeps all visual records, and only the defense="none" reasoning records.
    Without this filter, models that appear in the defenses sweep get their
    no-defense and defense runs pooled into one bucket, inflating n and
    skewing utility/leak rates.
    """
    return [
        r for r in records
        if r.setting != "reasoning" or r.defense == "none"
    ]


def write_main_table(out_dir: Path, records: list[Record]) -> None:
    records = _baseline_records(records)
    buckets = bucketize(records, lambda r: (r.setting, r.access_mode, r.model))
    rows: list[dict[str, Any]] = []
    for (setting, access_mode, model), bucket in sorted(buckets.items()):
        rows.append({
            "setting": setting,
            "access_mode": access_mode,
            "model": model,
            **bucket.metrics(),
        })
    write_csv(
        out_dir / "main_table.csv",
        rows,
        ["setting", "access_mode", "model", "n",
         "utility_rate", "leak_rate",
         "refusal_rate", "engagement_n", "engagement_leak_rate",
         "avg_ci_violation", "avg_num_leaked", "avg_reward"],
    )


def write_per_family(out_dir: Path, records: list[Record]) -> None:
    records = _baseline_records(records)
    buckets = bucketize(
        records,
        lambda r: (r.setting, r.access_mode, r.model, r.scenario_family),
    )
    rows = []
    for (setting, access_mode, model, family), bucket in sorted(buckets.items()):
        rows.append({
            "setting": setting,
            "access_mode": access_mode,
            "model": model,
            "scenario_family": family,
            **bucket.metrics(),
        })
    write_csv(
        out_dir / "per_family.csv",
        rows,
        ["setting", "access_mode", "model", "scenario_family", "n",
         "utility_rate", "leak_rate", "avg_ci_violation"],
    )


def write_per_mode(out_dir: Path, records: list[Record]) -> None:
    records = _baseline_records(records)
    buckets = bucketize(
        records,
        lambda r: (r.setting, r.access_mode, r.model, r.failure_mode),
    )
    rows = []
    for (setting, access_mode, model, mode), bucket in sorted(buckets.items()):
        rows.append({
            "setting": setting,
            "access_mode": access_mode,
            "model": model,
            "failure_mode": mode,
            **bucket.metrics(),
        })
    write_csv(
        out_dir / "per_mode.csv",
        rows,
        ["setting", "access_mode", "model", "failure_mode", "n",
         "utility_rate", "leak_rate",
         "refusal_rate", "engagement_n", "engagement_leak_rate",
         "avg_ci_violation"],
    )


def write_access_mode_pivot(out_dir: Path, records: list[Record]) -> None:
    """One row per visual model, columns for each access mode."""
    visual = [r for r in records if r.setting == "visual"]
    if not visual:
        write_csv(out_dir / "access_mode_pivot.csv", [], ["model"])
        return
    buckets = bucketize(visual, lambda r: (r.model, r.access_mode))
    by_model: dict[str, dict[str, Bucket]] = defaultdict(dict)
    modes_seen: set[str] = set()
    for (model, mode), bucket in buckets.items():
        by_model[model][mode] = bucket
        modes_seen.add(mode)
    modes_order = sorted(modes_seen)
    columns = ["model"]
    for mode in modes_order:
        columns += [f"{mode}_utility", f"{mode}_leak", f"{mode}_n"]
    rows = []
    for model in sorted(by_model):
        row = {"model": model}
        for mode in modes_order:
            b = by_model[model].get(mode)
            if b:
                m = b.metrics()
                row[f"{mode}_utility"] = m["utility_rate"]
                row[f"{mode}_leak"] = m["leak_rate"]
                row[f"{mode}_n"] = m["n"]
        rows.append(row)
    write_csv(out_dir / "access_mode_pivot.csv", rows, columns)


def write_defenses_pivot(
    out_dir: Path,
    records: list[Record],
    subset_ids: set[str] | None = None,
) -> None:
    """Defenses pivot for the reasoning setting (and legacy visual if present).

    Emits one CSV per setting that actually contains a non-`none` defense.
    Most paper runs now only populate reasoning; the visual pivot is written
    only when legacy visual_defenses jsonls are still on disk.

    When subset_ids is given, the "none" defense arm is filtered to only those
    scenario_ids (so it covers the same 70-scenario set as the defenses arms).
    """
    settings_with_defenses = {
        r.setting for r in records if r.defense != "none"
    }
    if not settings_with_defenses:
        # Default: still emit an empty reasoning pivot so downstream tooling
        # has a stable file to read.
        write_csv(out_dir / "defenses_pivot.csv", [], ["model"])
        return

    def _write_one(setting: str, filename: str) -> None:
        subset = [r for r in records if r.setting == setting]
        # For the "none" arm, restrict to the defenses subset when requested.
        if subset_ids is not None:
            subset = [
                r for r in subset
                if r.defense != "none" or r.scenario_id in subset_ids
            ]
        # Restrict "none" arm to ONLY the models that have at least one defense
        # run, so the pivot is paired (no orphan "none-only" rows).
        defense_models_local = {r.model for r in subset if r.defense != "none"}
        if defense_models_local:
            subset = [
                r for r in subset
                if r.defense != "none" or r.model in defense_models_local
            ]
        if not subset:
            return
        buckets = bucketize(subset, lambda r: (r.model, r.defense))
        by_model: dict[str, dict[str, Bucket]] = defaultdict(dict)
        defenses_seen: set[str] = set()
        for (model, defense), bucket in buckets.items():
            by_model[model][defense] = bucket
            defenses_seen.add(defense)
        defenses_order = ["none"] + sorted(d for d in defenses_seen if d != "none")
        columns = ["model"]
        for defense in defenses_order:
            columns += [
                f"{defense}_utility",
                f"{defense}_leak",
                f"{defense}_n",
                f"{defense}_delta_utility_vs_none",
                f"{defense}_delta_leak_vs_none",
            ]
        rows = []
        for model in sorted(by_model):
            row = {"model": model}
            baseline = by_model[model].get("none")
            baseline_u = baseline.metrics()["utility_rate"] if baseline else None
            baseline_l = baseline.metrics()["leak_rate"] if baseline else None
            for defense in defenses_order:
                b = by_model[model].get(defense)
                if not b:
                    continue
                m = b.metrics()
                row[f"{defense}_utility"] = m["utility_rate"]
                row[f"{defense}_leak"] = m["leak_rate"]
                row[f"{defense}_n"] = m["n"]
                if baseline_u is not None and defense != "none":
                    row[f"{defense}_delta_utility_vs_none"] = round(m["utility_rate"] - baseline_u, 4)
                    row[f"{defense}_delta_leak_vs_none"] = round(m["leak_rate"] - baseline_l, 4)
            rows.append(row)
        write_csv(out_dir / filename, rows, columns)

    # Reasoning is now the headline; visual is legacy.
    if "reasoning" in settings_with_defenses:
        _write_one("reasoning", "defenses_pivot.csv")
    if "visual" in settings_with_defenses:
        _write_one("visual", "defenses_pivot_visual_legacy.csv")


def write_confusion(out_dir: Path, records: list[Record]) -> None:
    records = _baseline_records(records)
    buckets = bucketize(records, lambda r: (r.setting, r.access_mode, r.model))
    rows = []
    for (setting, access_mode, model), b in sorted(buckets.items()):
        rows.append({
            "setting": setting,
            "access_mode": access_mode,
            "model": model,
            "n": b.n,
            "completed_clean": b.completed_clean,
            "completed_leak": b.completed_leak,
            "incomplete_clean": b.incomplete_clean,
            "incomplete_leak": b.incomplete_leak,
            "rate_completed_clean": round(b.completed_clean / b.n, 4) if b.n else 0.0,
            "rate_completed_leak": round(b.completed_leak / b.n, 4) if b.n else 0.0,
            "rate_incomplete_clean": round(b.incomplete_clean / b.n, 4) if b.n else 0.0,
            "rate_incomplete_leak": round(b.incomplete_leak / b.n, 4) if b.n else 0.0,
        })
    write_csv(
        out_dir / "confusion.csv",
        rows,
        ["setting", "access_mode", "model", "n",
         "completed_clean", "completed_leak", "incomplete_clean", "incomplete_leak",
         "rate_completed_clean", "rate_completed_leak",
         "rate_incomplete_clean", "rate_incomplete_leak"],
    )


def _cohens_kappa(a: list[int], b: list[int]) -> float:
    """Binary Cohen's kappa on parallel 0/1 lists."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if math.isclose(pe, 1.0):
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma = statistics.fmean(a)
    mb = statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return 0.0
    return round(num / (da * db), 4)


def write_text_vs_visual(out_dir: Path, records: list[Record]) -> None:
    """Per-model agreement between the reasoning setting and ui_only visual."""
    text_lookup: dict[tuple[str, str], int] = {}
    visual_lookup: dict[tuple[str, str], int] = {}
    for r in records:
        if r.setting == "reasoning" and r.defense == "none":
            text_lookup[(r.model, r.scenario_id)] = r.leaked
        elif r.setting == "visual" and r.access_mode == "ui_only" and r.defense == "none":
            visual_lookup[(r.model, r.scenario_id)] = r.leaked

    text_models = {m for (m, _) in text_lookup}
    visual_models = {m for (m, _) in visual_lookup}
    rows = []
    for model in sorted(text_models & visual_models):
        text_pairs = {sid: lk for (m, sid), lk in text_lookup.items() if m == model}
        visual_pairs = {sid: lk for (m, sid), lk in visual_lookup.items() if m == model}
        common = sorted(set(text_pairs) & set(visual_pairs))
        if not common:
            continue
        t = [text_pairs[sid] for sid in common]
        v = [visual_pairs[sid] for sid in common]
        leak_agree = sum(1 for ti, vi in zip(t, v) if ti == vi) / len(common)
        rows.append({
            "model": model,
            "n_common_scenarios": len(common),
            "text_leak_rate": round(sum(t) / len(t), 4),
            "visual_leak_rate": round(sum(v) / len(v), 4),
            "leak_agreement_rate": round(leak_agree, 4),
            "cohens_kappa": _cohens_kappa(t, v),
            "pearson_leak": _pearson([float(x) for x in t], [float(x) for x in v]),
        })
    write_csv(
        out_dir / "text_vs_visual_agreement.csv",
        rows,
        ["model", "n_common_scenarios",
         "text_leak_rate", "visual_leak_rate",
         "leak_agreement_rate", "cohens_kappa", "pearson_leak"],
    )


def write_reasoning_pareto(out_dir: Path, records: list[Record]) -> None:
    """Per-model points for the utility-vs-leakage scatter (reasoning only)."""
    reasoning = [r for r in records if r.setting == "reasoning" and r.defense == "none"]
    if not reasoning:
        write_csv(out_dir / "reasoning_pareto.csv", [], ["model"])
        return
    buckets = bucketize(reasoning, lambda r: r.model)
    rows = []
    for model, bucket in sorted(buckets.items()):
        m = bucket.metrics()
        rows.append({
            "model": model,
            "utility_rate": m["utility_rate"],
            "leak_rate": m["leak_rate"],
            "avg_ci_violation": m["avg_ci_violation"],
            "n": m["n"],
        })
    write_csv(
        out_dir / "reasoning_pareto.csv",
        rows,
        ["model", "utility_rate", "leak_rate", "avg_ci_violation", "n"],
    )


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def write_execution_gap(
    out_dir: Path,
    records: list[Record],
    dump_pairs: bool = False,
) -> None:
    """Execution gap: P(L^e2e=1 | L^s2a=0) for each model.

    Pairs reasoning (defense=none) with visual ui_only (defense=none) at
    scenario level. Models that appear in both settings are included.
    """
    text_lookup: dict[tuple[str, str], int] = {}
    visual_lookup: dict[tuple[str, str], int] = {}
    mode_lookup: dict[tuple[str, str], str] = {}

    for r in records:
        if r.setting == "reasoning" and r.defense == "none":
            text_lookup[(r.model, r.scenario_id)] = r.leaked
        elif r.setting == "visual" and r.access_mode == "ui_only" and r.defense == "none":
            visual_lookup[(r.model, r.scenario_id)] = r.leaked
            mode_lookup[(r.model, r.scenario_id)] = r.failure_mode

    text_models = {m for m, _ in text_lookup}
    visual_models = {m for m, _ in visual_lookup}

    eg_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for model in sorted(text_models & visual_models):
        t = {sid: lk for (m, sid), lk in text_lookup.items() if m == model}
        v = {sid: lk for (m, sid), lk in visual_lookup.items() if m == model}
        common = sorted(set(t) & set(v))
        if not common:
            continue

        n_paired = len(common)
        L_s2a = round(sum(t[sid] for sid in common) / n_paired, 4)
        L_e2e = round(sum(v[sid] for sid in common) / n_paired, 4)

        safe = [sid for sid in common if t[sid] == 0]
        n_safe = len(safe)
        k_leak = sum(1 for sid in safe if v[sid] == 1)
        eg = round(k_leak / n_safe, 4) if n_safe else 0.0
        eg_lo, eg_hi = _wilson_ci(k_leak, n_safe)

        eg_rows.append({
            "model": model,
            "n_paired": n_paired,
            "n_safe_s2a": n_safe,
            "L_s2a": L_s2a,
            "L_e2e": L_e2e,
            "EG": eg,
            "EG_lo": eg_lo,
            "EG_hi": eg_hi,
        })

        if dump_pairs:
            for sid in common:
                pair_rows.append({
                    "model": model,
                    "scenario_id": sid,
                    "failure_mode": mode_lookup.get((model, sid), ""),
                    "L_s2a": t[sid],
                    "L_e2e": v[sid],
                    "visual_access_mode": "ui_only",
                })

    write_csv(
        out_dir / "execution_gap.csv",
        eg_rows,
        ["model", "n_paired", "n_safe_s2a", "L_s2a", "L_e2e", "EG", "EG_lo", "EG_hi"],
    )
    if dump_pairs:
        write_csv(
            out_dir / "execution_gap_pairs.csv",
            pair_rows,
            ["model", "scenario_id", "failure_mode", "L_s2a", "L_e2e", "visual_access_mode"],
        )


def write_defenses_macro(
    out_dir: Path,
    records: list[Record],
    subset_ids: set[str] | None = None,
) -> None:
    """Average leak/utility rate per defense, pooled across all models.

    Two CSVs:
      defenses_macro.csv          — one row per defense (mean over models × scenarios)
      defenses_macro_per_mode.csv — one row per (defense, failure_mode)

    For a fair paired comparison, the "none" baseline is restricted to:
      (a) the same scenario_ids as the defenses arms (when subset_ids is set), AND
      (b) the same models that appear in at least one defense arm.
    Otherwise the macro would compare 15 models' "none" leak rate to 3 models'
    defense leak rate — apples to oranges.
    """
    reasoning = [r for r in records if r.setting == "reasoning"]
    if subset_ids is not None:
        reasoning = [
            r for r in reasoning
            if r.defense != "none" or r.scenario_id in subset_ids
        ]
    # Restrict the "none" baseline to ONLY the models that have at least one
    # non-none defense run, so the delta is paired.
    defense_models = {r.model for r in reasoning if r.defense != "none"}
    if defense_models:
        reasoning = [
            r for r in reasoning
            if r.defense != "none" or r.model in defense_models
        ]

    defenses_seen = sorted({r.defense for r in reasoning})
    if not defenses_seen:
        write_csv(out_dir / "defenses_macro.csv", [], ["defense"])
        write_csv(out_dir / "defenses_macro_per_mode.csv", [], ["defense", "failure_mode"])
        return

    # Macro (per defense only)
    by_def = bucketize(reasoning, lambda r: r.defense)
    rows = []
    baseline = by_def.get("none")
    base_u = baseline.metrics()["utility_rate"] if baseline else None
    base_l = baseline.metrics()["leak_rate"] if baseline else None
    for defense in ["none"] + [d for d in defenses_seen if d != "none"]:
        b = by_def.get(defense)
        if not b:
            continue
        m = b.metrics()
        row = {
            "defense": defense,
            "n": m["n"],
            "n_models": len({r.model for r in reasoning if r.defense == defense}),
            "utility_rate": m["utility_rate"],
            "leak_rate": m["leak_rate"],
            "avg_ci_violation": m["avg_ci_violation"],
        }
        if base_u is not None and defense != "none":
            row["delta_utility_vs_none"] = round(m["utility_rate"] - base_u, 4)
            row["delta_leak_vs_none"] = round(m["leak_rate"] - base_l, 4)
        rows.append(row)
    write_csv(
        out_dir / "defenses_macro.csv",
        rows,
        ["defense", "n", "n_models", "utility_rate", "leak_rate",
         "avg_ci_violation", "delta_utility_vs_none", "delta_leak_vs_none"],
    )

    # Macro per mode
    by_def_mode = bucketize(reasoning, lambda r: (r.defense, r.failure_mode))
    none_by_mode: dict[str, Bucket] = {
        fm: b for (d, fm), b in by_def_mode.items() if d == "none"
    }
    mode_rows = []
    for defense in ["none"] + [d for d in defenses_seen if d != "none"]:
        for (d, fm), b in sorted(by_def_mode.items()):
            if d != defense:
                continue
            m = b.metrics()
            row = {
                "defense": defense,
                "failure_mode": fm,
                "n": m["n"],
                "utility_rate": m["utility_rate"],
                "leak_rate": m["leak_rate"],
                "avg_ci_violation": m["avg_ci_violation"],
            }
            base = none_by_mode.get(fm)
            if base and defense != "none":
                bm = base.metrics()
                row["delta_utility_vs_none"] = round(m["utility_rate"] - bm["utility_rate"], 4)
                row["delta_leak_vs_none"] = round(m["leak_rate"] - bm["leak_rate"], 4)
            mode_rows.append(row)
    write_csv(
        out_dir / "defenses_macro_per_mode.csv",
        mode_rows,
        ["defense", "failure_mode", "n", "utility_rate", "leak_rate",
         "avg_ci_violation", "delta_utility_vs_none", "delta_leak_vs_none"],
    )


def write_execution_gap_per_mode(out_dir: Path, records: list[Record]) -> None:
    """Execution gap broken down by CI failure mode."""
    text_lookup: dict[tuple[str, str], int] = {}
    visual_lookup: dict[tuple[str, str], int] = {}
    mode_lookup: dict[tuple[str, str], str] = {}

    for r in records:
        if r.setting == "reasoning" and r.defense == "none":
            text_lookup[(r.model, r.scenario_id)] = r.leaked
        elif r.setting == "visual" and r.access_mode == "ui_only" and r.defense == "none":
            visual_lookup[(r.model, r.scenario_id)] = r.leaked
            mode_lookup[(r.model, r.scenario_id)] = r.failure_mode

    text_models = {m for m, _ in text_lookup}
    visual_models = {m for m, _ in visual_lookup}

    by_model_mode: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for model in sorted(text_models & visual_models):
        t = {sid: lk for (m, sid), lk in text_lookup.items() if m == model}
        v = {sid: lk for (m, sid), lk in visual_lookup.items() if m == model}
        for sid in set(t) & set(v):
            fm = mode_lookup.get((model, sid), "unknown")
            by_model_mode[(model, fm)].append((t[sid], v[sid]))

    rows: list[dict[str, Any]] = []
    for (model, fm), pairs in sorted(by_model_mode.items()):
        n_pairs = len(pairs)
        safe = [(ls, le) for ls, le in pairs if ls == 0]
        n_safe = len(safe)
        k_leak = sum(1 for _, le in safe if le == 1)
        eg = round(k_leak / n_safe, 4) if n_safe else 0.0
        rows.append({
            "model": model,
            "failure_mode": fm,
            "n_pairs": n_pairs,
            "n_safe_s2a": n_safe,
            "EG": eg,
        })

    write_csv(
        out_dir / "execution_gap_per_mode.csv",
        rows,
        ["model", "failure_mode", "n_pairs", "n_safe_s2a", "EG"],
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def write_report(out_dir: Path, records: list[Record]) -> None:
    n_total = len(records)
    by_setting = Counter(r.setting for r in records)
    by_access = Counter((r.setting, r.access_mode) for r in records)
    by_defense = Counter(r.defense for r in records if r.setting == "visual")
    families = Counter(r.scenario_family for r in records)
    modes = Counter(r.failure_mode for r in records)

    lines = []
    lines.append("# AgentCIBench results aggregate\n")
    lines.append(f"Total normalized records: **{n_total}**\n")
    lines.append("## By setting\n")
    for setting, n in sorted(by_setting.items()):
        lines.append(f"- {setting}: {n}")
    lines.append("\n## By (setting, access_mode)\n")
    for (setting, mode), n in sorted(by_access.items()):
        lines.append(f"- {setting} / {mode}: {n}")
    lines.append("\n## Visual defenses\n")
    for defense, n in sorted(by_defense.items()):
        lines.append(f"- {defense}: {n}")
    lines.append("\n## Scenario family distribution (across all records)\n")
    for fam, n in families.most_common():
        lines.append(f"- {fam}: {n}")
    lines.append("\n## Failure mode distribution\n")
    for mode, n in modes.most_common():
        lines.append(f"- {mode}: {n}")
    lines.append("\n## Files emitted\n")
    for f in (
        "main_table.csv",
        "per_family.csv",
        "per_mode.csv",
        "access_mode_pivot.csv",
        "defenses_pivot.csv",
        "confusion.csv",
        "text_vs_visual_agreement.csv",
        "reasoning_pareto.csv",
        "execution_gap.csv",
        "execution_gap_per_mode.csv",
        "defenses_macro.csv",
        "defenses_macro_per_mode.csv",
    ):
        lines.append(f"- `{f}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def _load_subset_ids_from_path(path: Path) -> set[str]:
    """Load scenario_ids from a manifest JSON or a directory of scenario JSONs."""
    if not path.exists():
        raise FileNotFoundError(f"--defenses-subset path not found: {path}")
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if "selections" in data:
            return {s["scenario_id"] for s in data["selections"]}
        raise ValueError(f"--defenses-subset file has no 'selections' key: {path}")
    ids: set[str] = set()
    for f in path.glob("*.json"):
        if "manifest" in f.stem.lower():
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            sid = d.get("scenario_id")
            if sid:
                ids.add(str(sid))
        except (OSError, json.JSONDecodeError):
            continue
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", default="data/results", type=Path)
    parser.add_argument("--out-dir", default="data/results/aggregated", type=Path)
    parser.add_argument(
        "--scenario-dirs",
        nargs="*",
        default=[
            "data/generated",
            "data/generated_merged",
            "data/seeds",
            "data/eval_set_pilot",
            "data/eval_set_main",
            "data/eval_set_ablation",
            "data/eval_set_e2e_50",
            "data/eval_set_e2e_40",
            "data/eval_set_e2e_30",
            "data/eval_set_defenses_70",
            "data/eval_set_defenses_50",
            "data/eval_set_defenses_30",
        ],
        type=Path,
        help="Directories scanned to recover scenario_family/failure_mode tags.",
    )
    parser.add_argument(
        "--dump-pairing",
        action="store_true",
        help="Also emit execution_gap_pairs.csv with per-(model,scenario) rows.",
    )
    parser.add_argument(
        "--defenses-subset",
        default=None,
        type=Path,
        help=(
            "Path to eval_set_defenses_70 directory or its manifest. "
            "When set, the 'none' defense arm is filtered to those scenario_ids "
            "so it covers the same set as the non-none defenses arms."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenario_dirs = [Path(p) for p in args.scenario_dirs]
    index = build_scenario_index(scenario_dirs)
    print(f"[index] {len(index)} scenarios indexed", file=sys.stderr)

    records: list[Record] = []
    records.extend(discover_text_records(args.results_root, index))
    records.extend(discover_visual_records(args.results_root, index))
    print(f"[records] {len(records)} normalized records", file=sys.stderr)

    if not records:
        print("[warn] no records found - aggregator emitted empty CSVs", file=sys.stderr)

    # Load defenses subset ids for pivot filtering.
    defenses_subset_ids: set[str] | None = None
    if args.defenses_subset is not None:
        defenses_subset_ids = _load_subset_ids_from_path(args.defenses_subset)
        print(
            f"[defenses-subset] {len(defenses_subset_ids)} scenario_ids loaded",
            file=sys.stderr,
        )

    write_main_table(args.out_dir, records)
    write_per_family(args.out_dir, records)
    write_per_mode(args.out_dir, records)
    write_access_mode_pivot(args.out_dir, records)
    write_defenses_pivot(args.out_dir, records, subset_ids=defenses_subset_ids)
    write_defenses_macro(args.out_dir, records, subset_ids=defenses_subset_ids)
    write_confusion(args.out_dir, records)
    write_text_vs_visual(args.out_dir, records)
    write_reasoning_pareto(args.out_dir, records)
    write_execution_gap(args.out_dir, records, dump_pairs=args.dump_pairing)
    write_execution_gap_per_mode(args.out_dir, records)
    write_report(args.out_dir, records)

    print(f"[done] wrote tables to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
