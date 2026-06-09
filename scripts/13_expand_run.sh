#!/usr/bin/env bash
# Orchestrate the post-first-pass scenario-pool expansion.
#
# Runs three expansion stages back-to-back:
#
#   A) Second MCTS pass with fresh RNG seed base.   (~5.5 h, ~$8)
#      Different stochastic tree -> different child scenarios.
#   B) LLM-augmented variant generation + re-validation.   (~3 h, ~$15-20)
#      For each accepted scenario, generate K variants under structural
#      constraints, then re-validate through the same open-weight reward
#      judge as MCTS.
#   C) Cross-pool dedup with relaxed thresholds.   (seconds, $0)
#      Merge first-pass + second-pass + augmented pools, dedup-by-signature
#      with a looser must_not_share Jaccard than the engine's in-run default.
#
# By default every stage is enabled. Disable any with RUN_<STAGE>=0.
#
# Usage:
#   scripts/13_expand_run.sh
#
# Env knobs:
#   RUN_MCTS_PASS2     default: 1  (Stage A)
#   RUN_AUGMENT        default: 1  (Stage B)
#   RUN_FINAL_DEDUP    default: 1  (Stage C)
#
#   PASS2_RNG_SEED     default: 100  (different from first pass default 42)
#   PASS2_OUTPUT_DIR   default: data/generated_pass2
#
#   AUGMENT_INPUT_DIR    default: data/generated  (first pass)
#   AUGMENT_OUTPUT_DIR   default: data/generated_augmented
#   VARIANTS_PER_SOURCE  default: 3
#   GENERATOR_MODEL      default: $AGENTCI_MUTATOR_MODEL  (deepseek-v4-pro)
#
#   FINAL_OUTPUT_DIR        default: data/generated_merged
#   FINAL_PROMPT_SIMILARITY default: 0.92
#   FINAL_JACCARD           default: 0.50  (looser than engine's 0.70)
#
#   KEEP_THRESHOLD     default: 2.5  (applied in pass2 and augment validation)
#   USE_OPENROUTER     default: 1
#   MAX_CONCURRENCY    default: 1   (for shell-side parallelism; for slurm use sbatch_*)

set -uo pipefail
cd "$(dirname "$0")/.."

RUN_MCTS_PASS2="${RUN_MCTS_PASS2:-1}"
RUN_AUGMENT="${RUN_AUGMENT:-1}"
RUN_FINAL_DEDUP="${RUN_FINAL_DEDUP:-1}"

PASS2_RNG_SEED="${PASS2_RNG_SEED:-100}"
PASS2_OUTPUT_DIR="${PASS2_OUTPUT_DIR:-data/generated_pass2}"

AUGMENT_INPUT_DIR="${AUGMENT_INPUT_DIR:-data/generated}"
AUGMENT_OUTPUT_DIR="${AUGMENT_OUTPUT_DIR:-data/generated_augmented}"
VARIANTS_PER_SOURCE="${VARIANTS_PER_SOURCE:-7}"

FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-data/generated_merged}"
FINAL_PROMPT_SIMILARITY="${FINAL_PROMPT_SIMILARITY:-0.92}"
FINAL_JACCARD="${FINAL_JACCARD:-0.50}"

KEEP_THRESHOLD="${KEEP_THRESHOLD:-2.5}"
USE_OPENROUTER="${USE_OPENROUTER:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

mkdir -p logs
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a logs/expand_run.log; }

print_pool_counts() {
  local label="$1"
  for d in data/generated data/generated_pass2 data/generated_augmented data/generated_merged; do
    [ -d "$d" ] || continue
    local n
    n="$(ls -1 "$d"/*.json 2>/dev/null | grep -vE 'manifest|dedup_stats|augment_summary' | wc -l | tr -d ' ')"
    log "  $label  $d: $n scenarios"
  done
}

log "== expand_run begin =="
print_pool_counts "BEFORE"

########################################################################
# Stage A: second MCTS pass with a different RNG seed base.
########################################################################
if [ "$RUN_MCTS_PASS2" = "1" ]; then
  log "Stage A: second MCTS pass (RNG_SEED_BASE=$PASS2_RNG_SEED -> $PASS2_OUTPUT_DIR)"
  USE_OPENROUTER="$USE_OPENROUTER" \
  RNG_SEED_BASE="$PASS2_RNG_SEED" \
  OUTPUT_DIR="$PASS2_OUTPUT_DIR" \
  RUN_LOG_DIR="data/results/mcts_runs_pass2" \
  KEEP_THRESHOLD="$KEEP_THRESHOLD" \
  MAX_CONCURRENCY="$MAX_CONCURRENCY" \
    scripts/01_generate_scenarios.sh \
    2>&1 | tee -a logs/expand_run_pass2.log
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "Stage A FAILED (some seeds may have errored). Continuing with what landed."
  fi
  print_pool_counts "AFTER stage A"
else
  log "Stage A skipped (RUN_MCTS_PASS2=0)"
fi

########################################################################
# Stage B: LLM-augmented variants + open-weight re-validation.
#
# Operates on the FIRST-PASS pool by default. If you also want to augment
# the second pass, change AUGMENT_INPUT_DIR or run this stage again with
# a different input dir.
########################################################################
if [ "$RUN_AUGMENT" = "1" ]; then
  if [ ! -d "$AUGMENT_INPUT_DIR" ]; then
    log "Stage B skipped: $AUGMENT_INPUT_DIR not found."
  else
    GEN_FLAG=()
    if [ -n "${GENERATOR_MODEL:-}" ]; then
      GEN_FLAG=(--generator-model "$GENERATOR_MODEL")
    fi
    OR_FLAG=()
    [ "$USE_OPENROUTER" = "1" ] && OR_FLAG=(--use-openrouter)

    log "Stage B: augment ($AUGMENT_INPUT_DIR -> $AUGMENT_OUTPUT_DIR, K=$VARIANTS_PER_SOURCE)"
    uv run python scripts/12_augment_variants.py \
      --input-dir "$AUGMENT_INPUT_DIR" \
      --output-dir "$AUGMENT_OUTPUT_DIR" \
      --variants-per-source "$VARIANTS_PER_SOURCE" \
      --keep-threshold "$KEEP_THRESHOLD" \
      "${GEN_FLAG[@]}" \
      "${OR_FLAG[@]}" \
      2>&1 | tee -a logs/expand_run_augment.log
    print_pool_counts "AFTER stage B"
  fi
else
  log "Stage B skipped (RUN_AUGMENT=0)"
fi

########################################################################
# Stage C: cross-pool merge + relaxed dedup.
#
# Reads any of the four pools that exist on disk and writes the deduped
# union to FINAL_OUTPUT_DIR. Use the union as the released benchmark.
########################################################################
if [ "$RUN_FINAL_DEDUP" = "1" ]; then
  POOL_FLAGS=()
  for d in data/generated "$PASS2_OUTPUT_DIR" "$AUGMENT_OUTPUT_DIR"; do
    [ -d "$d" ] && POOL_FLAGS+=(--input-dir "$d")
  done
  if [ "${#POOL_FLAGS[@]}" -eq 0 ]; then
    log "Stage C skipped: no input pools on disk."
  else
    log "Stage C: cross-pool dedup -> $FINAL_OUTPUT_DIR (prompt>=$FINAL_PROMPT_SIMILARITY, jaccard>=$FINAL_JACCARD)"
    uv run python scripts/11_redo_dedup.py \
      "${POOL_FLAGS[@]}" \
      --output-dir "$FINAL_OUTPUT_DIR" \
      --prompt-similarity "$FINAL_PROMPT_SIMILARITY" \
      --jaccard "$FINAL_JACCARD" \
      2>&1 | tee -a logs/expand_run_dedup.log
    print_pool_counts "AFTER stage C"
  fi
else
  log "Stage C skipped (RUN_FINAL_DEDUP=0)"
fi

log "== expand_run done =="
log "Final merged pool (use this as the released benchmark): $FINAL_OUTPUT_DIR"
log "    count: $(ls -1 "$FINAL_OUTPUT_DIR"/*.json 2>/dev/null | grep -vE 'manifest|dedup_stats' | wc -l | tr -d ' ')"
