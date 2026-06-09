#!/usr/bin/env python3
"""Cross-pool near-duplicate dedup over one or more scenario directories.

USE CASES:

1. **Merge two MCTS passes.** After running MCTS twice with different
   `RNG_SEED_BASE`, both pools are deduped internally but may overlap.
   Pass both dirs and emit the union with cross-pool near-duplicates removed.

2. **Merge MCTS pool with LLM-augmented variants.** After running
   `scripts/12_augment_variants.py`, variants land in a separate dir.
   This script re-checks them against the original pool.

3. **Relax the original dedup thresholds.** The MCTS engine uses
   prompt_sim >= 0.92 AND must_not_share_jaccard >= 0.70 internally. This
   script lets you pick different thresholds (looser or stricter) and
   re-apply them across the merged pool.

WHAT IT IS NOT:

This script CANNOT recover scenarios that were dropped by MCTS-internal
dedup. Those scenarios were never written to disk. To preserve them for
future re-dedup, future MCTS runs need to save pre-dedup scenarios
separately (a small engine change, not done here).

Output: each kept scenario is copied unchanged into `--output-dir`. Stats
go to stdout and to `<output-dir>/dedup_stats.json`.

Usage:

  scripts/11_redo_dedup.py \\
      --input-dir data/generated \\
      --input-dir data/generated_pass2 \\
      --input-dir data/generated_augmented \\
      --output-dir data/generated_merged \\
      --prompt-similarity 0.92 \\
      --jaccard 0.50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Normalization (mirrors mcts/mcts_engine.py exactly so re-dedup matches the
# engine's in-run filter byte-for-byte when thresholds match.)
# --------------------------------------------------------------------------- #

import re


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _scenario_content_fingerprint(scenario: dict[str, Any]) -> str:
    copy = dict(scenario)
    copy.pop("scenario_id", None)
    return json.dumps(copy, sort_keys=True, ensure_ascii=True)


def _scenario_signature(scenario: dict[str, Any]) -> tuple[str, frozenset[str]]:
    task_prompt = _normalize_text(str(scenario.get("task_prompt", "")))
    gt = scenario.get("ground_truth", {})
    mns = gt.get("must_not_share", []) if isinstance(gt, dict) else []
    normalized = frozenset(_normalize_text(str(x)) for x in mns if str(x).strip())
    return task_prompt, normalized


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _is_near_duplicate(
    candidate_sig: tuple[str, frozenset[str]],
    kept_sig: tuple[str, frozenset[str]],
    prompt_threshold: float,
    jaccard_threshold: float,
) -> bool:
    cp, cs = candidate_sig
    kp, ks = kept_sig
    prompt_sim = SequenceMatcher(None, cp, kp).ratio()
    sens_sim = _jaccard(cs, ks)
    return prompt_sim >= prompt_threshold and sens_sim >= jaccard_threshold


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def load_scenarios(dirs: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for d in dirs:
        if not d.is_dir():
            print(f"[warn] skip non-existent dir {d}", file=sys.stderr)
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("manifest"):
                continue
            try:
                with p.open("r", encoding="utf-8") as h:
                    data = json.load(h)
            except (OSError, json.JSONDecodeError) as e:
                print(f"[warn] skip {p}: {e}", file=sys.stderr)
                continue
            if not isinstance(data, dict) or "scenario_id" not in data:
                continue
            out.append((p, data))
    return out


def dedupe(
    scenarios: list[tuple[Path, dict[str, Any]]],
    prompt_threshold: float,
    jaccard_threshold: float,
) -> tuple[list[tuple[Path, dict[str, Any]]], dict[str, int]]:
    # Stage 1: exact content fingerprint.
    seen: set[str] = set()
    exact_unique: list[tuple[Path, dict[str, Any]]] = []
    exact_removed = 0
    for path, scenario in scenarios:
        fp = _scenario_content_fingerprint(scenario)
        if fp in seen:
            exact_removed += 1
            continue
        seen.add(fp)
        exact_unique.append((path, scenario))

    # Stage 2: near-duplicate signature.
    kept: list[tuple[Path, dict[str, Any]]] = []
    kept_sigs: list[tuple[str, frozenset[str]]] = []
    near_removed = 0
    for path, scenario in exact_unique:
        sig = _scenario_signature(scenario)
        if any(
            _is_near_duplicate(sig, ks, prompt_threshold, jaccard_threshold)
            for ks in kept_sigs
        ):
            near_removed += 1
            continue
        kept.append((path, scenario))
        kept_sigs.append(sig)

    return kept, {
        "input_total": len(scenarios),
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "kept_total": len(kept),
        "prompt_threshold": prompt_threshold,
        "jaccard_threshold": jaccard_threshold,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input-dir", "-i", action="append", required=True, type=Path,
        help="Input directory containing scenario JSON files. Repeat for multiple pools.",
    )
    p.add_argument("--output-dir", "-o", required=True, type=Path,
                   help="Where surviving scenarios are copied.")
    p.add_argument(
        "--prompt-similarity", type=float, default=0.92,
        help="Ratcliff/Obershelp threshold on the normalized task prompt (default 0.92, the engine default).",
    )
    p.add_argument(
        "--jaccard", type=float, default=0.50,
        help="Jaccard threshold on normalized must_not_share set (default 0.50; engine default 0.70 is stricter).",
    )
    p.add_argument("--copy", action="store_true",
                   help="Copy survivors to --output-dir (default behavior). Set to false to just print stats.")
    p.add_argument("--no-copy", dest="copy", action="store_false")
    p.set_defaults(copy=True)
    args = p.parse_args()

    scenarios = load_scenarios(args.input_dir)
    print(f"[load] read {len(scenarios)} scenarios across {len(args.input_dir)} dir(s)", file=sys.stderr)
    if not scenarios:
        return 2

    kept, stats = dedupe(scenarios, args.prompt_similarity, args.jaccard)

    print(json.dumps(stats, indent=2))

    if args.copy:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Track scenario_id collisions across input dirs: keep the first one seen,
        # rename later ones with a suffix.
        used_ids: set[str] = set()
        for path, scenario in kept:
            sid = str(scenario.get("scenario_id", path.stem))
            target_id = sid
            suffix = 1
            while target_id in used_ids:
                target_id = f"{sid}__d{suffix}"
                suffix += 1
            used_ids.add(target_id)
            dst = args.output_dir / f"{target_id}.json"
            if target_id != sid:
                # Rewrite the scenario_id inside the JSON to match the filename.
                scenario = {**scenario, "scenario_id": target_id}
                with dst.open("w", encoding="utf-8") as h:
                    json.dump(scenario, h, indent=2)
            else:
                shutil.copy2(path, dst)
        stats_path = args.output_dir / "dedup_stats.json"
        with stats_path.open("w", encoding="utf-8") as h:
            json.dump(stats, h, indent=2)
        print(f"[write] copied {len(kept)} survivors to {args.output_dir}/", file=sys.stderr)
        print(f"[write] stats at {stats_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
