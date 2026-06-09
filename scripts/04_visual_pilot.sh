#!/usr/bin/env bash
# Tiny visual pilot: 25-scenario × 5-agent run in `ui_only` mode to validate
# the BrowserGym harness, judge, and per-scenario artifact layout BEFORE
# spending real money on the main sweep.
#
# Cost: ~$5-8 depending on which agents you keep.
#
# Usage:
#   scripts/04_visual_pilot.sh [EVAL_DIR] [RESULTS_DIR]
#
# Env overrides:
#   MODELS        space-separated LiteLLM model strings
#   JUDGE_MODEL   default: openai/gpt-5-mini
#   MAX_STEPS     default: 12

set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_DIR="${1:-data/eval_set_pilot}"
RESULTS_DIR="${2:-data/results/visual_pilot}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5-mini}"
MAX_STEPS="${MAX_STEPS:-12}"

# Default frontier 5. Override with MODELS="..." to swap.
DEFAULT_MODELS="openai/gpt-5 anthropic/claude-sonnet-4.6 google/gemini-2.5-pro openai/gpt-5-mini google/gemini-2.5-flash"
MODELS_RAW="${MODELS:-$DEFAULT_MODELS}"
read -ra MODELS_ARRAY <<< "$MODELS_RAW"

if [ ! -d "$EVAL_DIR" ]; then
  echo "Eval dir not found: $EVAL_DIR" >&2
  echo "Build one first:" >&2
  echo "  scripts/03_build_eval_set.py --output-dir $EVAL_DIR --total 25 \\" >&2
  echo "    --rng-seed 2026 --manifest $EVAL_DIR/manifest.json" >&2
  exit 2
fi

scenario_count="$(ls -1 "$EVAL_DIR"/*.json 2>/dev/null | grep -v manifest | wc -l | tr -d ' ')"

MODEL_ARGS=()
for m in "${MODELS_ARRAY[@]}"; do
  MODEL_ARGS+=(--model-name "$m")
done

mkdir -p "$RESULTS_DIR"

export AGENTCI_ALLOW_HEURISTIC_FALLBACK="${AGENTCI_ALLOW_HEURISTIC_FALLBACK:-0}"

echo "==> Visual pilot"
echo "    eval set    : $EVAL_DIR ($scenario_count scenarios)"
echo "    models      : ${MODELS_ARRAY[*]}"
echo "    judge       : $JUDGE_MODEL"
echo "    access mode : ui_only"
echo "    max steps   : $MAX_STEPS"
echo "    results     : $RESULTS_DIR"
echo

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
echo "==> Pilot done. Inspect:"
echo "    $RESULTS_DIR/summary__*.json"
echo "    $RESULTS_DIR/runs/<run_id>/ui_only/<model>/<scenario_id>/run_result.json"
