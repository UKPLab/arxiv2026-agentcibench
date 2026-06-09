#!/usr/bin/env bash
# Post-sweep summary: completion check, error scan, aggregate, pretty-print all
# paper-facing results: S2A main, defenses, E2E visual, execution gap.
#
# Usage:
#   scripts/sweep_status.sh
#
# Optional env:
#   EXPECTED_S2A=15   number of S2A models expected (default 15)
#   AGGREGATED_DIR    override aggregated output dir

set -uo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."

EXPECTED_S2A="${EXPECTED_S2A:-15}"
AGGREGATED_DIR="${AGGREGATED_DIR:-data/results/aggregated}"

SEP="================================================================"

echo "$SEP"
echo " Sweep status — $(date '+%Y-%m-%d %H:%M')"
echo "$SEP"

# ── 1. Completion ──────────────────────────────────────────────────────────────
echo
echo "[1/5] Completion"

# S2A text
text_summ=$(find data/results/text -maxdepth 2 -name "summary__*.json" 2>/dev/null | wc -l | tr -d ' ')
text_jsonl=$(find data/results/text -maxdepth 2 -name "benchmark_results__*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
echo "  S2A text/"
echo "    summaries:  $text_summ  (expected $EXPECTED_S2A)"
echo "    jsonls:     $text_jsonl"
[ "$text_summ" -lt "$EXPECTED_S2A" ] && echo "  WARNING: only $text_summ/$EXPECTED_S2A S2A summaries"

# Defenses
tdef_arms=$(find data/results/text_defenses -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
tdef_summ=$(find data/results/text_defenses -name "summary__*.json" 2>/dev/null | wc -l | tr -d ' ')
echo "  Defenses text_defenses/"
echo "    arms:       $tdef_arms  (expected 3: recipient_typed restrictive rubric_informed)"
echo "    summaries:  $tdef_summ  (expected 9 = 3 arms × 3 models)"

# Visual E2E
vmixed_jsonl=$(find data/results/visual_mixed -maxdepth 1 -name "benchmark_results__*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
vmixed_runs=$(find data/results/visual_mixed/runs -name "run_result.json" 2>/dev/null | wc -l | tr -d ' ')
echo "  Visual E2E visual_mixed/"
echo "    jsonls:     $vmixed_jsonl"
echo "    run_result.json files: $vmixed_runs  (expected 100 = 50 scenarios × 2 models)"

# ── 2. Per-model rows ──────────────────────────────────────────────────────────
echo
echo "[2/5] S2A per-model JSONL rows"
text_jsonl_count=$(find data/results/text -maxdepth 2 -name "benchmark_results__*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
if [ "$text_jsonl_count" -gt 0 ]; then
  for f in data/results/text/*/benchmark_results__*.jsonl; do
    model=$(dirname "$f" | xargs basename)
    n=$(wc -l < "$f" | tr -d ' ')
    n_err=$(grep -c '"error"' "$f" 2>/dev/null || true)
    n_err=${n_err:-0}
    printf "  %-44s rows=%4d  errors=%d\n" "$model" "$n" "$n_err"
  done | sort
else
  echo "  (no S2A JSONLs yet)"
fi

# ── 3. Error scan ──────────────────────────────────────────────────────────────
echo
echo "[3/5] Error scan (logs/slurm)"
err_files=( logs/slurm/text-*.err logs/slurm/textdef-*.err logs/slurm/vsmoke-*.err logs/slurm/visual-*.err )
if [ "${#err_files[@]}" -eq 0 ]; then
  echo "  (no slurm .err files)"
else
  any=0
  for pattern in "Traceback" "429" "rate.limit" "model.not.found" "OPENROUTER_API_KEY" "Connection.refused" "browser.*crash" "max.*step"; do
    hits=$(grep -rlE "$pattern" "${err_files[@]}" 2>/dev/null || true)
    if [ -n "$hits" ]; then
      any=1
      n=$(printf '%s\n' "$hits" | wc -l | tr -d ' ')
      echo "  $pattern: $n log file(s)"
      printf '%s\n' "$hits" | sed 's/^/      /'
    fi
  done
  [ "$any" -eq 0 ] && echo "  (no error patterns found)"
fi

# ── 4. Aggregate ───────────────────────────────────────────────────────────────
echo
echo "[4/5] Aggregating ..."
mkdir -p "$AGGREGATED_DIR"
AGG_FLAGS="--dump-pairing"
uv run python scripts/10_aggregate_results.py $AGG_FLAGS --out-dir "$AGGREGATED_DIR" 2>&1 | tail -5

# ── 5. Paper results ───────────────────────────────────────────────────────────
echo
echo "$SEP"
echo " Paper results"
echo "$SEP"

# ── 5a. S2A main (Table 1) ────────────────────────────────────────────────────
echo
echo "── S2A main sweep (Table 1) — sorted by leak_rate ──"
uv run python - "$AGGREGATED_DIR" <<'PY'
import csv, sys
from pathlib import Path
agg = Path(sys.argv[1])
path = agg / "main_table.csv"
if not path.exists():
    print("  (no main_table.csv yet)"); raise SystemExit(0)
rows = [r for r in csv.DictReader(path.open()) if r.get("setting") == "reasoning"]
if not rows:
    print("  (no reasoning records)"); raise SystemExit(0)
rows.sort(key=lambda r: float(r["leak_rate"]))
hdr = f"  {'model':<42} {'n':>4} {'U':>6} {'L':>6} {'Ref':>6} {'L|eng':>7} {'CI':>5}"
print(hdr)
print(f"  {'-'*42} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*5}")
for r in rows:
    print(
        f"  {r['model']:<42} {r['n']:>4} "
        f"{float(r['utility_rate'])*100:>5.1f}% "
        f"{float(r['leak_rate'])*100:>5.1f}% "
        f"{float(r.get('refusal_rate',0))*100:>5.1f}% "
        f"{float(r.get('engagement_leak_rate',0))*100:>6.1f}% "
        f"{float(r['avg_ci_violation']):>5.2f}"
    )
leaks = [float(r["leak_rate"]) for r in rows]
eng   = [float(r.get("engagement_leak_rate",0)) for r in rows]
print()
print(f"  spread {(max(leaks)-min(leaks))*100:.1f} pp  |  mean L {sum(leaks)/len(leaks)*100:.1f}%  |  mean L|eng {sum(eng)/len(eng)*100:.1f}%")
print("  U=utility  L=leak  Ref=refusal  L|eng=leak on engaged scenarios  CI=mean severity")
PY

# ── 5b. Defenses (§5.5) ───────────────────────────────────────────────────────
echo
echo "── Defenses sweep (§5.5) — macro averages over 3 models ──"
uv run python - "$AGGREGATED_DIR" <<'PY'
import csv, sys
from pathlib import Path
agg = Path(sys.argv[1])

macro = agg / "defenses_macro.csv"
macro_mode = agg / "defenses_macro_per_mode.csv"

if not macro.exists():
    print("  (no defenses_macro.csv yet)"); raise SystemExit(0)

rows = list(csv.DictReader(macro.open()))
print(f"  {'defense':<20} {'n':>5} {'U':>6} {'L':>6} {'ΔU':>7} {'ΔL':>7} {'CI':>5}")
print(f"  {'-'*20} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*5}")
for r in rows:
    du = r.get("delta_utility_vs_none","")
    dl = r.get("delta_leak_vs_none","")
    du_s = f"{float(du)*100:>+6.1f}%" if du else "      "
    dl_s = f"{float(dl)*100:>+6.1f}%" if dl else "      "
    print(
        f"  {r['defense']:<20} {r['n']:>5} "
        f"{float(r['utility_rate'])*100:>5.1f}% "
        f"{float(r['leak_rate'])*100:>5.1f}% "
        f"{du_s} {dl_s} "
        f"{float(r['avg_ci_violation']):>5.2f}"
    )

if macro_mode.exists():
    print()
    print("  Per-mode breakdown:")
    rows_m = list(csv.DictReader(macro_mode.open()))
    print(f"  {'defense':<20} {'mode':<30} {'n':>5} {'U':>6} {'L':>6} {'ΔL':>7}")
    print(f"  {'-'*20} {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*7}")
    for r in rows_m:
        dl = r.get("delta_leak_vs_none","")
        dl_s = f"{float(dl)*100:>+6.1f}%" if dl else "      "
        print(
            f"  {r['defense']:<20} {r['failure_mode']:<30} {r['n']:>5} "
            f"{float(r['utility_rate'])*100:>5.1f}% "
            f"{float(r['leak_rate'])*100:>5.1f}% "
            f"{dl_s}"
        )
PY

# ── 5c. E2E Visual + Execution Gap (Table 2) ──────────────────────────────────
echo
echo "── E2E Visual + Execution Gap (Table 2) ──"
uv run python - "$AGGREGATED_DIR" <<'PY'
import csv, sys
from pathlib import Path
agg = Path(sys.argv[1])

eg_path = agg / "execution_gap.csv"
eg_mode_path = agg / "execution_gap_per_mode.csv"
main_path = agg / "main_table.csv"

if not eg_path.exists():
    print("  (no execution_gap.csv — visual_mixed run not yet complete)")
    raise SystemExit(0)

rows = list(csv.DictReader(eg_path.open()))
print(f"  {'model':<42} {'n_paired':>9} {'L_s2a':>7} {'L_e2e':>7} {'n_safe':>7} {'EG':>7}  95% CI")
print(f"  {'-'*42} {'-'*9} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*14}")
for r in rows:
    print(
        f"  {r['model']:<42} {r['n_paired']:>9} "
        f"{float(r['L_s2a'])*100:>6.1f}% "
        f"{float(r['L_e2e'])*100:>6.1f}% "
        f"{r['n_safe_s2a']:>7} "
        f"{float(r['EG'])*100:>6.1f}%  "
        f"[{float(r['EG_lo'])*100:.1f}%, {float(r['EG_hi'])*100:.1f}%]"
    )
egs = [float(r["EG"]) for r in rows]
if egs:
    print(f"\n  mean EG: {sum(egs)/len(egs)*100:.1f}%  (target ~30%)")

if eg_mode_path.exists():
    print()
    print("  Execution gap per failure mode:")
    mrows = list(csv.DictReader(eg_mode_path.open()))
    print(f"  {'model':<42} {'mode':<32} {'n_safe':>7} {'EG':>7}")
    print(f"  {'-'*42} {'-'*32} {'-'*7} {'-'*7}")
    for r in mrows:
        print(
            f"  {r['model']:<42} {r['failure_mode']:<32} "
            f"{r.get('n_safe_s2a','?'):>7} "
            f"{float(r['EG'])*100:>6.1f}%"
        )

# Also show E2E visual rows from main_table for completeness
if main_path.exists():
    vrows = [r for r in csv.DictReader(main_path.open()) if r.get("setting") == "visual"]
    if vrows:
        print()
        print("  E2E visual raw rates (from main_table.csv):")
        print(f"  {'model':<42} {'n':>4} {'U':>6} {'L':>6} {'CI':>5}")
        print(f"  {'-'*42} {'-'*4} {'-'*6} {'-'*6} {'-'*5}")
        for r in sorted(vrows, key=lambda r: float(r["leak_rate"])):
            print(
                f"  {r['model']:<42} {r['n']:>4} "
                f"{float(r['utility_rate'])*100:>5.1f}% "
                f"{float(r['leak_rate'])*100:>5.1f}% "
                f"{float(r['avg_ci_violation']):>5.2f}"
            )
PY

echo
echo "Full CSVs → $AGGREGATED_DIR/"
echo "  main_table.csv  per_mode.csv  defenses_macro.csv  defenses_macro_per_mode.csv"
echo "  execution_gap.csv  execution_gap_per_mode.csv  execution_gap_pairs.csv"
