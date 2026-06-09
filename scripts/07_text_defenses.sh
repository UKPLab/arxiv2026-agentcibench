#!/usr/bin/env bash
# Reasoning-only defenses sweep on a stratified subset.
#
# Replays a stratified subset of generated scenarios under each system-prompt
# defense in config/defenses/ and compares against the `none` baseline. Moved
# off the visual setting (formerly 07_visual_defenses.sh) so the defenses
# table can use the full 4-model panel cheaply, and moved off the full
# reasoning pool because the headline claim is a paired Delta vs. `none`,
# which is well-powered at N=150 stratified (50 per failure mode).
#
# Statistical floor: at N=150 the bootstrap 95% CI half-width on a single
# leakage rate is +/-8 pts; the paired Delta vs. `none` (same scenarios)
# tightens to +/-4 pts, which is enough to resolve the ~14 pt Restrictive
# drop and the ~18 pt Recipient-typed drop on recipient_misalignment.
#
# Cost model: subset_scenarios * |MODELS| * |DEFENSES| reasoning rollouts.
# At the defaults below (150 scenarios * 4 models * 4 defenses = 2400 calls,
# proxy = gpt-5-class, judge = gpt-5-mini) this lands at ~$5-7.
#
# Build the subset before running this script:
#   scripts/03_build_eval_set.py \
#     --output-dir data/eval_set_defenses --total 150 \
#     --rng-seed 2026 --manifest data/eval_set_defenses/manifest.json
#
# Usage:
#   scripts/07_text_defenses.sh [EVAL_DIR] [RESULTS_DIR]
#
# Args:
#   $1  EVAL_DIR      default: data/eval_set_defenses  (stratified 150)
#   $2  RESULTS_ROOT  default: data/results/text_defenses
#
# Env overrides:
#   MODELS         space-separated LiteLLM model strings. Default: a 4-model
#                  panel chosen to cover the closed/open and high/low capability
#                  quadrants visible in Table 1 (paper §5.3).
#   DEFENSES       space-separated defense names matching config/defenses/*.txt
#                  Default: restrictive rubric_informed recipient_typed
#   INCLUDE_NONE   default: 1 (include `none` baseline so the defenses pivot
#                  table has a within-sweep baseline column on the same
#                  stratified subset; set to 0 if you intend to compare
#                  against the no-defense sweep from 02_text_benchmark.sh.
#                  Compare-within-sweep is usually preferable because both
#                  baseline and defense runs see the identical 150 scenarios,
#                  which makes the paired Delta cleaner.)
#   JUDGE_MODEL    default: openai/gpt-5-mini
#   USE_OPENROUTER default: 0
#   MAX_CONCURRENCY default: 1

set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_DIR="${1:-data/eval_set_defenses}"
RESULTS_ROOT="${2:-data/results/text_defenses}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5-mini}"
USE_OPENROUTER="${USE_OPENROUTER:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

# 4-model panel: one premium closed (gpt-5), one premium closed alt (claude),
# one efficient closed (gemini-flash), one open-weight (deepseek). Big enough
# to populate the Pareto plot, small enough to stay under $20 across all
# defenses on the full reasoning pool.
DEFAULT_MODELS="openai/gpt-5 anthropic/claude-sonnet-4.6 google/gemini-2.5-flash deepseek/deepseek-v3.2"
MODELS_RAW="${MODELS:-$DEFAULT_MODELS}"
read -ra MODELS_ARRAY <<< "$MODELS_RAW"

DEFAULT_DEFENSES="restrictive rubric_informed recipient_typed"
DEFENSES_RAW="${DEFENSES:-$DEFAULT_DEFENSES}"
read -ra DEFENSES_ARRAY <<< "$DEFENSES_RAW"

INCLUDE_NONE="${INCLUDE_NONE:-1}"
if [ "$INCLUDE_NONE" = "1" ]; then
  DEFENSES_ARRAY=("none" "${DEFENSES_ARRAY[@]}")
fi

prefix_if_unprefixed() {
  case "$1" in
    openrouter/*) printf '%s' "$1" ;;
    *)            printf 'openrouter/%s' "$1" ;;
  esac
}

if [ "$USE_OPENROUTER" = "1" ]; then
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "USE_OPENROUTER=1 but OPENROUTER_API_KEY is not set." >&2
    exit 1
  fi
  for i in "${!MODELS_ARRAY[@]}"; do
    MODELS_ARRAY[$i]="$(prefix_if_unprefixed "${MODELS_ARRAY[$i]}")"
  done
  JUDGE_MODEL="$(prefix_if_unprefixed "$JUDGE_MODEL")"
fi

if [ ! -d "$EVAL_DIR" ]; then
  echo "Eval dir not found: $EVAL_DIR" >&2
  echo "Build one first:" >&2
  echo "  scripts/03_build_eval_set.py \\" >&2
  echo "    --output-dir $EVAL_DIR --total 150 \\" >&2
  echo "    --rng-seed 2026 --manifest $EVAL_DIR/manifest.json" >&2
  exit 2
fi

scenario_count="$(ls -1 "$EVAL_DIR"/*.json 2>/dev/null | grep -v manifest | wc -l | tr -d ' ')"

export AGENTCI_ALLOW_HEURISTIC_FALLBACK="${AGENTCI_ALLOW_HEURISTIC_FALLBACK:-0}"
export AGENTCI_LITELLM_TIMEOUT_SECONDS="${AGENTCI_LITELLM_TIMEOUT_SECONDS:-120}"
export AGENTCI_LITELLM_RETRIES="${AGENTCI_LITELLM_RETRIES:-6}"

total_runs=$((scenario_count * ${#MODELS_ARRAY[@]} * ${#DEFENSES_ARRAY[@]}))

echo "==> Reasoning defenses sweep"
echo "    eval set      : $EVAL_DIR ($scenario_count scenarios)"
echo "    models        : ${MODELS_ARRAY[*]}"
echo "    defenses      : ${DEFENSES_ARRAY[*]}"
echo "    judge         : $JUDGE_MODEL"
echo "    concurrency   : $MAX_CONCURRENCY"
echo "    estimated runs: $total_runs"
read -r -p "Proceed? [y/N] " ans
case "$ans" in
  y|Y|yes|YES) ;;
  *) echo "Aborted."; exit 1 ;;
esac

slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9._-]+#-#g; s#^-+|-+$##g'
}

run_one_cell() {
  local defense="$1" model="$2"
  local model_slug
  model_slug="$(slug "$model")"
  local out_dir="$RESULTS_ROOT/$defense/$model_slug"
  mkdir -p "$out_dir"
  echo "==> defense=$defense  model=$model -> $out_dir"
  uv run python -m eval.run_benchmark \
    --generated-dir "$EVAL_DIR" \
    --results-dir "$out_dir" \
    --proxy-model "$model" \
    --judge-model "$JUDGE_MODEL" \
    --defense "$defense" \
    --no-progress
}

failures=()
for defense in "${DEFENSES_ARRAY[@]}"; do
  if [ "$MAX_CONCURRENCY" -le 1 ]; then
    for model in "${MODELS_ARRAY[@]}"; do
      if ! run_one_cell "$defense" "$model"; then
        echo "  FAILED: $defense/$model" >&2
        failures+=("$defense/$model")
      fi
    done
  else
    pids=(); tags=(); active=0
    for model in "${MODELS_ARRAY[@]}"; do
      if [ "$active" -ge "$MAX_CONCURRENCY" ]; then
        wait -n || true
        active=$((active - 1))
      fi
      run_one_cell "$defense" "$model" &
      pids+=("$!")
      tags+=("$defense/$model")
      active=$((active + 1))
    done
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        failures+=("${tags[$i]}")
      fi
    done
  fi
done

echo
echo "==> Reasoning defenses sweep done."
echo "    results : $RESULTS_ROOT/<defense>/<model_slug>/summary__*.json"
echo "    aggregate next: uv run python scripts/10_aggregate_results.py"
if [ "${#failures[@]}" -gt 0 ]; then
  echo "    failed cells: ${failures[*]}" >&2
  exit 2
fi
