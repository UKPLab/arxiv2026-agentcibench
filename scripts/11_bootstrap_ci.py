#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals for all key metrics in the paper.

Outputs to data/results/aggregated/:
  - s2a_ci.csv            per-model utility/leak/ci with 95% CIs
  - s2a_per_mode_ci.csv   per-model per-mode utility/leak with 95% CIs
  - defenses_ci.csv       per-defense (macro avg over models) with 95% CIs
  - defenses_per_model_ci.csv  per-defense per-model with 95% CIs
  - defenses_per_mode_ci.csv   per-defense per-mode (macro avg) with 95% CIs
  - e2e_ci.csv            per-model E2E utility/leak with 95% CIs

Usage:
  uv run python scripts/11_bootstrap_ci.py
  uv run python scripts/11_bootstrap_ci.py --results-root data/results --out-dir data/results/aggregated --n-boot 10000
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

SCENARIO_FAMILY_TO_MODE = {
    "ambiguity_trap": "task_ambiguity_overshare",
    "identity_bleed": "recipient_misalignment",
    "semantic_entanglement": "visual_co_location",
}


# ---------------------------------------------------------------------------
# Bootstrap core
# ---------------------------------------------------------------------------

def bootstrap_ci(values, stat_fn=np.mean, n_boot=5000, ci=0.95, rng=None):
    """Return (point_estimate, lower, upper) with percentile bootstrap CI."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(values, dtype=float)
    point = stat_fn(arr)
    if len(arr) < 2:
        return point, float("nan"), float("nan")
    boots = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_text_records(results_root):
    """Load all S2A text benchmark records. Returns list of dicts."""
    records = []
    pattern = os.path.join(results_root, "text", "*", "benchmark_results__*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                score = d.get("score", {}) or {}
                records.append({
                    "model": d.get("proxy_model") or d.get("model_name", "?"),
                    "scenario_id": d.get("scenario_id", "?"),
                    "scenario_family": d.get("scenario_family", "?"),
                    "failure_mode": SCENARIO_FAMILY_TO_MODE.get(d.get("scenario_family", ""), "?"),
                    "defense": d.get("defense", "none"),
                    "utility": int(bool(score.get("utility", 0))),
                    "leaked": int(bool(score.get("leaked_items"))),
                    "ci_violation": float(score.get("ci_violation", 0) or 0),
                    "refusal": int(not bool(score.get("utility")) and not bool(score.get("leaked_items")) and score.get("ci_violation", 1) == 1),
                })
    return records


def load_defenses_records(results_root):
    """Load defenses sweep records (text_defenses dir). Returns list of dicts."""
    records = []
    pattern = os.path.join(results_root, "text_defenses", "*", "*", "benchmark_results__*.jsonl")
    for path in sorted(glob.glob(pattern)):
        # defense arm and model from path: text_defenses/<defense>/<model>/benchmark_...
        parts = path.split(os.sep)
        defense = parts[-3]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                score = d.get("score", {}) or {}
                records.append({
                    "model": d.get("proxy_model") or d.get("model_name", "?"),
                    "scenario_id": d.get("scenario_id", "?"),
                    "scenario_family": d.get("scenario_family", "?"),
                    "failure_mode": SCENARIO_FAMILY_TO_MODE.get(d.get("scenario_family", ""), "?"),
                    "defense": defense,
                    "utility": int(bool(score.get("utility", 0))),
                    "leaked": int(bool(score.get("leaked_items"))),
                    "ci_violation": float(score.get("ci_violation", 0) or 0),
                    "refusal": int(not bool(score.get("utility")) and not bool(score.get("leaked_items")) and score.get("ci_violation", 1) == 1),
                })
    return records


def load_visual_records(results_root):
    """Load E2E visual benchmark records. Returns list of dicts."""
    records = []
    pattern = os.path.join(results_root, "visual_mixed", "runs", "**", "run_result.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        r = json.load(open(path))
        records.append({
            "model": r.get("model_name", "?"),
            "scenario_id": r.get("scenario_id", "?"),
            "scenario_family": r.get("scenario_family", "?"),
            "failure_mode": SCENARIO_FAMILY_TO_MODE.get(r.get("scenario_family", ""), "?"),
            "defense": r.get("defense", "none"),
            "utility": int(bool(r.get("utility", 0))),
            "leaked": int(bool(r.get("leaked_items"))),
            "ci_violation": float(r.get("ci_violation", 0) or 0),
        })
    return records


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def group_by(records, keys):
    """Group list of dicts by tuple of key values."""
    groups = defaultdict(list)
    for r in records:
        k = tuple(r[key] for key in keys)
        groups[k].append(r)
    return groups


def compute_stats(recs, n_boot, rng):
    """Compute mean + CI for utility, leak, ci_violation on a list of records."""
    n = len(recs)
    utility = [r["utility"] for r in recs]
    leaked = [r["leaked"] for r in recs]
    ci_v = [r["ci_violation"] for r in recs]

    u, u_lo, u_hi = bootstrap_ci(utility, n_boot=n_boot, rng=rng)
    l, l_lo, l_hi = bootstrap_ci(leaked, n_boot=n_boot, rng=rng)
    c, c_lo, c_hi = bootstrap_ci(ci_v, n_boot=n_boot, rng=rng)

    # Engagement-conditioned leak: bootstrap leak rate on the subset of
    # records that did NOT refuse. Only defined when records carry "refusal".
    out = {
        "n": n,
        "utility_rate": round(u, 4),
        "utility_lo": round(u_lo, 4),
        "utility_hi": round(u_hi, 4),
        "leak_rate": round(l, 4),
        "leak_lo": round(l_lo, 4),
        "leak_hi": round(l_hi, 4),
        "avg_ci_violation": round(c, 4),
        "ci_viol_lo": round(c_lo, 4),
        "ci_viol_hi": round(c_hi, 4),
    }
    if recs and "refusal" in recs[0]:
        engaged_leak = [r["leaked"] for r in recs if not r["refusal"]]
        out["engagement_n"] = len(engaged_leak)
        if engaged_leak:
            el, el_lo, el_hi = bootstrap_ci(engaged_leak, n_boot=n_boot, rng=rng)
            out["engagement_leak_rate"] = round(el, 4)
            out["engagement_leak_lo"] = round(el_lo, 4)
            out["engagement_leak_hi"] = round(el_hi, 4)
        else:
            out["engagement_leak_rate"] = 0.0
            out["engagement_leak_lo"] = 0.0
            out["engagement_leak_hi"] = 0.0
        out["refusal_rate"] = round(sum(r["refusal"] for r in recs) / n, 4)
    return out


def write_csv(rows, fieldnames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# S2A main table CI
# ---------------------------------------------------------------------------

def compute_s2a_ci(records, n_boot, rng, out_dir):
    s2a = [r for r in records if r["defense"] == "none"]

    # Per-model
    rows = []
    for (model,), recs in sorted(group_by(s2a, ["model"]).items()):
        row = {"model": model}
        row.update(compute_stats(recs, n_boot, rng))
        rows.append(row)
    rows.sort(key=lambda r: r["leak_rate"])

    fields = ["model", "n",
              "utility_rate", "utility_lo", "utility_hi",
              "leak_rate", "leak_lo", "leak_hi",
              "avg_ci_violation", "ci_viol_lo", "ci_viol_hi"]
    write_csv(rows, fields, os.path.join(out_dir, "s2a_ci.csv"))

    # Per-model per-mode
    rows_pm = []
    for (model, mode), recs in sorted(group_by(s2a, ["model", "failure_mode"]).items()):
        row = {"model": model, "failure_mode": mode}
        row.update(compute_stats(recs, n_boot, rng))
        rows_pm.append(row)
    rows_pm.sort(key=lambda r: (r["model"], r["failure_mode"]))

    fields_pm = ["model", "failure_mode", "n",
                 "utility_rate", "utility_lo", "utility_hi",
                 "leak_rate", "leak_lo", "leak_hi",
                 "avg_ci_violation", "ci_viol_lo", "ci_viol_hi"]
    write_csv(rows_pm, fields_pm, os.path.join(out_dir, "s2a_per_mode_ci.csv"))


# ---------------------------------------------------------------------------
# Defenses CI
# ---------------------------------------------------------------------------

def compute_defenses_ci(s2a_records, defenses_records, n_boot, rng, out_dir):
    # Combine: "none" arm comes from S2A records (full pool, same 3 models)
    defense_models = {r["model"] for r in defenses_records}
    none_recs = [r for r in s2a_records if r["defense"] == "none" and r["model"] in defense_models]
    all_def = none_recs + defenses_records

    # Per-defense macro (average over models): compute per-model then average CIs
    # Bootstrap at the record level, grouped by defense
    rows_macro = []
    for (defense,), recs in sorted(group_by(all_def, ["defense"]).items()):
        row = {"defense": defense, "n_models": len({r["model"] for r in recs})}
        row.update(compute_stats(recs, n_boot, rng))
        rows_macro.append(row)

    # Add delta vs none
    none_row = next((r for r in rows_macro if r["defense"] == "none"), None)
    for r in rows_macro:
        if none_row:
            r["delta_leak_vs_none"] = round(r["leak_rate"] - none_row["leak_rate"], 4)
            r["delta_utility_vs_none"] = round(r["utility_rate"] - none_row["utility_rate"], 4)
        else:
            r["delta_leak_vs_none"] = ""
            r["delta_utility_vs_none"] = ""

    fields = ["defense", "n_models", "n",
              "utility_rate", "utility_lo", "utility_hi",
              "leak_rate", "leak_lo", "leak_hi",
              "refusal_rate", "engagement_n",
              "engagement_leak_rate", "engagement_leak_lo", "engagement_leak_hi",
              "avg_ci_violation", "ci_viol_lo", "ci_viol_hi",
              "delta_utility_vs_none", "delta_leak_vs_none"]
    write_csv(rows_macro, fields, os.path.join(out_dir, "defenses_ci.csv"))

    # Per-defense per-model
    rows_pm = []
    for (defense, model), recs in sorted(group_by(all_def, ["defense", "model"]).items()):
        row = {"defense": defense, "model": model}
        row.update(compute_stats(recs, n_boot, rng))
        rows_pm.append(row)

    # Add delta vs none per model
    none_by_model = {r["model"]: r for r in rows_pm if r["defense"] == "none"}
    for r in rows_pm:
        baseline = none_by_model.get(r["model"])
        if baseline:
            r["delta_leak_vs_none"] = round(r["leak_rate"] - baseline["leak_rate"], 4)
            r["delta_utility_vs_none"] = round(r["utility_rate"] - baseline["utility_rate"], 4)
        else:
            r["delta_leak_vs_none"] = ""
            r["delta_utility_vs_none"] = ""

    fields_pm = ["defense", "model", "n",
                 "utility_rate", "utility_lo", "utility_hi",
                 "leak_rate", "leak_lo", "leak_hi",
                 "refusal_rate", "engagement_n",
                 "engagement_leak_rate", "engagement_leak_lo", "engagement_leak_hi",
                 "avg_ci_violation", "ci_viol_lo", "ci_viol_hi",
                 "delta_utility_vs_none", "delta_leak_vs_none"]
    write_csv(rows_pm, fields_pm, os.path.join(out_dir, "defenses_per_model_ci.csv"))

    # Per-defense per-mode (macro over models)
    rows_mode = []
    for (defense, mode), recs in sorted(group_by(all_def, ["defense", "failure_mode"]).items()):
        row = {"defense": defense, "failure_mode": mode, "n_models": len({r["model"] for r in recs})}
        row.update(compute_stats(recs, n_boot, rng))
        rows_mode.append(row)

    none_by_mode = {r["failure_mode"]: r for r in rows_mode if r["defense"] == "none"}
    for r in rows_mode:
        baseline = none_by_mode.get(r["failure_mode"])
        if baseline:
            r["delta_leak_vs_none"] = round(r["leak_rate"] - baseline["leak_rate"], 4)
            r["delta_utility_vs_none"] = round(r["utility_rate"] - baseline["utility_rate"], 4)
        else:
            r["delta_leak_vs_none"] = ""
            r["delta_utility_vs_none"] = ""

    fields_mode = ["defense", "failure_mode", "n_models", "n",
                   "utility_rate", "utility_lo", "utility_hi",
                   "leak_rate", "leak_lo", "leak_hi",
                   "refusal_rate", "engagement_n",
                   "engagement_leak_rate", "engagement_leak_lo", "engagement_leak_hi",
                   "avg_ci_violation", "ci_viol_lo", "ci_viol_hi",
                   "delta_utility_vs_none", "delta_leak_vs_none"]
    write_csv(rows_mode, fields_mode, os.path.join(out_dir, "defenses_per_mode_ci.csv"))


# ---------------------------------------------------------------------------
# E2E CI
# ---------------------------------------------------------------------------

def compute_e2e_ci(visual_records, n_boot, rng, out_dir):
    if not visual_records:
        print("  no visual records found, skipping e2e_ci.csv")
        return

    rows = []
    for (model,), recs in sorted(group_by(visual_records, ["model"]).items()):
        row = {"model": model}
        row.update(compute_stats(recs, n_boot, rng))
        rows.append(row)

    fields = ["model", "n",
              "utility_rate", "utility_lo", "utility_hi",
              "leak_rate", "leak_lo", "leak_hi",
              "avg_ci_violation", "ci_viol_lo", "ci_viol_hi"]
    write_csv(rows, fields, os.path.join(out_dir, "e2e_ci.csv"))

    # Per model per mode
    rows_pm = []
    for (model, mode), recs in sorted(group_by(visual_records, ["model", "failure_mode"]).items()):
        row = {"model": model, "failure_mode": mode}
        row.update(compute_stats(recs, n_boot, rng))
        rows_pm.append(row)

    fields_pm = ["model", "failure_mode", "n",
                 "utility_rate", "utility_lo", "utility_hi",
                 "leak_rate", "leak_lo", "leak_hi",
                 "avg_ci_violation", "ci_viol_lo", "ci_viol_hi"]
    write_csv(rows_pm, fields_pm, os.path.join(out_dir, "e2e_per_mode_ci.csv"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="data/results")
    parser.add_argument("--out-dir", default="data/results/aggregated")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading S2A text records...")
    s2a_records = load_text_records(args.results_root)
    print(f"  {len(s2a_records)} records from {len({r['model'] for r in s2a_records})} models")

    print("Loading defenses records...")
    def_records = load_defenses_records(args.results_root)
    print(f"  {len(def_records)} records, defenses: {sorted({r['defense'] for r in def_records})}")

    print("Loading visual/E2E records...")
    vis_records = load_visual_records(args.results_root)
    print(f"  {len(vis_records)} records from {len({r['model'] for r in vis_records})} models")

    print(f"\nBootstrapping with n_boot={args.n_boot}, seed={args.seed}...")

    print("\n[S2A]")
    compute_s2a_ci(s2a_records, args.n_boot, rng, args.out_dir)

    print("\n[Defenses]")
    compute_defenses_ci(s2a_records, def_records, args.n_boot, rng, args.out_dir)

    print("\n[E2E]")
    compute_e2e_ci(vis_records, args.n_boot, rng, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
