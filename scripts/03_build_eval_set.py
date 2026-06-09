#!/usr/bin/env python3
"""Build a stratified evaluation subset from the generated scenario pool.

Splits proportionally by `scenario_family` (the mutation strategy that produced
each scenario), so the eval set stays balanced as new strategies are added.
Optionally also balances by source seed (the prefix of `scenario_id` before
`__`), which is useful when one seed has produced many more keepers than
others.

Outputs:
- a copy of each selected scenario JSON into `--output-dir`
- an optional manifest JSON (`--manifest`) that lists every selection so the
  eval set is reproducible from the manifest alone.

Examples:
  # Pilot: 25 scenarios, balanced across families.
  scripts/03_build_eval_set.py \
      --output-dir data/eval_set_pilot \
      --total 25 \
      --rng-seed 2026 \
      --manifest data/eval_set_pilot/manifest.json

  # Main eval: 200 scenarios, also balanced across source seeds.
  scripts/03_build_eval_set.py \
      --output-dir data/eval_set_main \
      --total 200 \
      --balance-seeds \
      --rng-seed 2026 \
      --manifest data/eval_set_main/manifest.json

  # Ablation subset: deterministic 100-scenario subset of the main eval.
  scripts/03_build_eval_set.py \
      --input-dir data/eval_set_main \
      --output-dir data/eval_set_ablation \
      --total 100 \
      --balance-seeds \
      --rng-seed 2026 \
      --manifest data/eval_set_ablation/manifest.json

  # Defenses subset: 70 scenarios with explicit per-failure-mode quotas.
  scripts/03_build_eval_set.py \
      --input-dir data/generated_merged \
      --output-dir data/eval_set_defenses_70 \
      --per-mode-quotas visual_co_location:18,recipient_misalignment:22,task_ambiguity_overshare:30 \
      --rng-seed 2027 \
      --manifest data/eval_set_defenses_70/manifest.json

  # Paired access-mode probe: 20-scenario subset that must be inside e2e_40.
  scripts/03_build_eval_set.py \
      --input-dir data/generated_merged \
      --output-dir data/eval_set_access_20 \
      --total 20 \
      --subset-of data/eval_set_e2e_40 \
      --rng-seed 2028 \
      --manifest data/eval_set_access_20/manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _source_seed(scenario_id: str) -> str:
    return scenario_id.split("__", 1)[0] if scenario_id else "unknown"


def _scenario_family(scenario: dict) -> str:
    fam = str(scenario.get("scenario_family", "")).strip()
    return fam or "unknown"


def _failure_mode(scenario: dict) -> str:
    mode = str(scenario.get("failure_mode", "")).strip()
    return mode or "unknown"


def _scenario_apps(scenario: dict) -> set[str]:
    """Return the set of app names present in a scenario's initial_states."""
    return {k.replace("open_", "") for k in scenario.get("initial_states", {}).keys()}


def _parse_per_mode_quotas(raw: str) -> dict[str, int]:
    """Parse 'visual_co_location:18,recipient_misalignment:22,...' into a dict."""
    quotas: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition(":")
        if not key or not val:
            raise ValueError(f"Invalid quota spec '{part}'; expected 'mode:count'")
        quotas[key.strip()] = int(val.strip())
    return quotas


def _load_subset_ids(path: Path) -> set[str]:
    """Return scenario_ids from a directory of JSONs or a manifest JSON."""
    if not path.exists():
        raise FileNotFoundError(f"--subset-of path not found: {path}")
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if "selections" in data:
            return {s["scenario_id"] for s in data["selections"]}
        raise ValueError(f"--subset-of file has no 'selections' key: {path}")
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", default="data/generated_merged",
                        help="Directory of *.json scenarios to sample from.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to copy the selected subset.")
    parser.add_argument("--total", type=int, default=200,
                        help="Target subset size (may undershoot if a family has too few entries).")
    parser.add_argument("--rng-seed", type=int, default=2026,
                        help="Seed for reproducible sampling.")
    parser.add_argument("--balance-seeds", action="store_true",
                        help="Within each family, also balance across source seeds (round-robin).")
    parser.add_argument("--manifest", default=None,
                        help="Optional path for a JSON manifest of selections.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without copying files.")
    parser.add_argument(
        "--per-mode-quotas",
        default=None,
        help=(
            "Explicit per-failure-mode quotas, e.g. "
            "'visual_co_location:18,recipient_misalignment:22,task_ambiguity_overshare:30'. "
            "When set, groups by failure_mode and ignores --total."
        ),
    )
    parser.add_argument(
        "--subset-of",
        default=None,
        type=Path,
        help=(
            "Path to a directory of scenario JSONs or a manifest JSON. "
            "Only scenarios whose scenario_id appears in that set are eligible."
        ),
    )
    parser.add_argument(
        "--exclude-apps",
        default=None,
        help=(
            "Comma-separated list of app names to exclude. Any scenario whose "
            "initial_states contains one of these apps is dropped before sampling. "
            "Example: --exclude-apps shop,maps"
        ),
    )
    args = parser.parse_args()

    src = Path(args.input_dir)
    if not src.is_dir():
        print(f"Input dir not found: {src}", file=sys.stderr)
        return 2

    # Resolve --subset-of constraint (eligible scenario_ids).
    subset_ids: set[str] | None = None
    if args.subset_of is not None:
        subset_ids = _load_subset_ids(args.subset_of)
        print(f"--subset-of: restricting to {len(subset_ids)} eligible scenario_ids")

    # Resolve --exclude-apps.
    excluded_apps: set[str] = set()
    if args.exclude_apps:
        excluded_apps = {a.strip() for a in args.exclude_apps.split(",") if a.strip()}
        print(f"--exclude-apps: dropping scenarios that contain any of {sorted(excluded_apps)}")

    # Parse --per-mode-quotas if provided.
    per_mode_quotas: dict[str, int] | None = None
    if args.per_mode_quotas:
        per_mode_quotas = _parse_per_mode_quotas(args.per_mode_quotas)
        print(f"--per-mode-quotas: {per_mode_quotas}")

    files = sorted(src.glob("*.json"))
    # Skip manifests, conversion artifacts, etc.
    files = [f for f in files if not f.name.startswith("manifest")
                                and "manifest" not in f.stem.lower()]
    if not files:
        print(f"No *.json scenarios in {src}", file=sys.stderr)
        return 2

    # When per-mode-quotas is set, group by failure_mode; otherwise by scenario_family.
    group_key = _failure_mode if per_mode_quotas else _scenario_family

    by_family: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    parse_errors = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parse_errors += 1
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            parse_errors += 1
            continue
        # Apply --subset-of filter.
        if subset_ids is not None:
            sid = data.get("scenario_id", "")
            if str(sid) not in subset_ids:
                continue
        # Apply --exclude-apps filter.
        if excluded_apps and (_scenario_apps(data) & excluded_apps):
            continue
        by_family[group_key(data)].append((path, data))

    families = sorted(by_family)
    if not families:
        print("No usable scenarios found.", file=sys.stderr)
        return 2

    key_label = "failure modes" if per_mode_quotas else "families"
    print(f"Pool: {sum(len(v) for v in by_family.values())} scenarios "
          f"across {len(families)} {key_label} "
          f"({parse_errors} unreadable):")
    for fam in families:
        print(f"  {fam}: {len(by_family[fam])}")

    # Compute quotas: explicit per-mode-quotas or equal-split across families.
    if per_mode_quotas:
        quota: dict[str, int] = {}
        for fam in families:
            quota[fam] = per_mode_quotas.get(fam, 0)
            if fam not in per_mode_quotas:
                print(f"  warning: no quota specified for '{fam}'; skipping")
    else:
        per_fam_base = args.total // len(families)
        remainder = args.total - per_fam_base * len(families)
        fam_by_size = sorted(families, key=lambda f: len(by_family[f]), reverse=True)
        quota = {fam: per_fam_base for fam in families}
        for fam in fam_by_size[:remainder]:
            quota[fam] += 1

    rng = random.Random(args.rng_seed)
    selections: list[Path] = []
    selection_meta: list[dict] = []

    for fam in families:
        if quota.get(fam, 0) == 0:
            continue
        pool = list(by_family[fam])
        target = min(quota[fam], len(pool))
        if target < quota[fam]:
            print(f"  warning: '{fam}' has only {len(pool)} scenarios "
                  f"(quota {quota[fam]}); will undershoot")

        if args.balance_seeds:
            by_seed: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
            for path, data in pool:
                by_seed[_source_seed(str(data.get("scenario_id", "")))].append((path, data))
            for entries in by_seed.values():
                rng.shuffle(entries)
            seed_keys = sorted(by_seed)
            cursors = {s: 0 for s in seed_keys}
            picked: list[tuple[Path, dict]] = []
            while len(picked) < target:
                progressed = False
                for s in seed_keys:
                    if len(picked) >= target:
                        break
                    if cursors[s] < len(by_seed[s]):
                        picked.append(by_seed[s][cursors[s]])
                        cursors[s] += 1
                        progressed = True
                if not progressed:
                    break
        else:
            shuffled = list(pool)
            rng.shuffle(shuffled)
            picked = shuffled[:target]

        for path, data in picked:
            selections.append(path)
            selection_meta.append({
                "scenario_id": data.get("scenario_id", path.stem),
                "scenario_family": _scenario_family(data),
                "failure_mode": _failure_mode(data),
                "source_seed": _source_seed(str(data.get("scenario_id", ""))),
                "source_path": str(path),
            })

    fam_breakdown = Counter(m["scenario_family"] for m in selection_meta)
    mode_breakdown = Counter(m["failure_mode"] for m in selection_meta)
    seed_breakdown = Counter(m["source_seed"] for m in selection_meta)
    total_requested = sum(per_mode_quotas.values()) if per_mode_quotas else args.total
    print()
    print(f"Selected {len(selections)} scenarios "
          f"(target {total_requested}; deficit {total_requested - len(selections)})")
    print("  by scenario_family:")
    for fam, count in sorted(fam_breakdown.items()):
        print(f"    {fam}: {count}")
    print("  by failure_mode:")
    for mode, count in sorted(mode_breakdown.items()):
        print(f"    {mode}: {count}")
    print("  by source seed:")
    for seed, count in sorted(seed_breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {seed}: {count}")

    if args.dry_run:
        print("\n(dry-run: nothing copied)")
        return 0

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Don't silently overwrite an existing eval set — fail loudly.
    existing = [p for p in out.glob("*.json") if "manifest" not in p.stem.lower()]
    if existing:
        print(f"\nRefusing to overwrite: {out} already contains {len(existing)} scenario JSON(s). "
              "Choose a fresh --output-dir or delete it first.", file=sys.stderr)
        return 3

    for path in selections:
        shutil.copy2(path, out / path.name)

    print(f"\nCopied {len(selections)} scenarios -> {out}")

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({
                "input_dir": str(src),
                "output_dir": str(out),
                "total_requested": total_requested,
                "total_selected": len(selections),
                "rng_seed": args.rng_seed,
                "balance_seeds": args.balance_seeds,
                "per_mode_quotas": per_mode_quotas,
                "subset_of": str(args.subset_of) if args.subset_of else None,
                "excluded_apps": sorted(excluded_apps) if excluded_apps else [],
                "family_counts": dict(sorted(fam_breakdown.items())),
                "mode_counts": dict(sorted(mode_breakdown.items())),
                "seed_counts": dict(sorted(seed_breakdown.items())),
                "selections": selection_meta,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Manifest -> {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
