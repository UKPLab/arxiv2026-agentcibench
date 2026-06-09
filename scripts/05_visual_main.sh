#!/usr/bin/env bash
# Main visual sweep: full eval set × all 5 agents × ui_only.
#
# Run scripts/04_visual_pilot.sh first and inspect run_result.json artifacts
# before launching this — it's the most expensive single command in the
# pipeline (~$50 at default settings).
#
# Usage:
#   scripts/05_visual_main.sh [EVAL_DIR] [RESULTS_DIR]
#
# Env overrides:
#   MODELS        space-separated LiteLLM model strings
#   JUDGE_MODEL   default: openai/gpt-5-mini
#   MAX_STEPS     default: 12

set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_DIR="${1:-data/eval_set_main}"
RESULTS_DIR="${2:-data/results/visual_main}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5-mini}"
MAX_STEPS="${MAX_STEPS:-12}"

DEFAULT_MODELS="openai/gpt-5 anthropic/claude-sonnet-4.6 google/gemini-2.5-pro openai/gpt-5-mini google/gemini-2.5-flash"
MODELS_RAW="${MODELS:-$DEFAULT_MODELS}"
read -ra MODELS_ARRAY <<< "$MODELS_RAW"

if [ ! -d "$EVAL_DIR" ]; then
  echo "Eval dir not found: $EVAL_DIR" >&2
  echo "Build one first:" >&2
  echo "  scripts/03_build_eval_set.py --output-dir $EVAL_DIR --total 200 \\" >&2
  echo "    --balance-seeds --rng-seed 2026 --manifest $EVAL_DIR/manifest.json" >&2
  exit 2
fi

scenario_count="$(ls -1 "$EVAL_DIR"/*.json 2>/dev/null | grep -v manifest | wc -l | tr -d ' ')"

MODEL_ARGS=()
for m in "${MODELS_ARRAY[@]}"; do
  MODEL_ARGS+=(--model-name "$m")
done

mkdir -p "$RESULTS_DIR"

export AGENTCI_ALLOW_HEURISTIC_FALLBACK="${AGENTCI_ALLOW_HEURISTIC_FALLBACK:-0}"
export AGENTCI_LITELLM_TIMEOUT_SECONDS="${AGENTCI_LITELLM_TIMEOUT_SECONDS:-180}"
export AGENTCI_LITELLM_RETRIES="${AGENTCI_LITELLM_RETRIES:-6}"

echo "==> Visual main sweep"
echo "    eval set    : $EVAL_DIR ($scenario_count scenarios)"
echo "    models      : ${MODELS_ARRAY[*]}"
echo "    judge       : $JUDGE_MODEL"
echo "    access mode : ui_only"
echo "    max steps   : $MAX_STEPS"
echo "    results     : $RESULTS_DIR"
echo
echo "Estimated runs: $((scenario_count * ${#MODELS_ARRAY[@]}))"
read -r -p "Proceed? [y/N] " ans
case "$ans" in
  y|Y|yes|YES) ;;
  *) echo "Aborted."; exit 1 ;;
esac

uv run python -m eval.run_visual_benchmark \
  --generated-dir "$EVAL_DIR" \
  --access-mode ui_only \
  --judge-model "$JUDGE_MODEL" \
  --results-dir "$RESULTS_DIR" \
  --max-steps "$MAX_STEPS" \
  --use-litellm \
  --api-key-env OPENROUTER_API_KEY \
  "${MODEL_ARGS[@]}"

echo
echo "==> Main sweep done."
echo "    summary  : $RESULTS_DIR/summary__*.json"
echo "    per-run  : $RESULTS_DIR/runs/<run_id>/ui_only/<model>/<scenario_id>/run_result.json"
