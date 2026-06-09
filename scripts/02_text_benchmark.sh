#!/usr/bin/env bash
# Reasoning (text-only) headline sweep over the generated scenario pool.
#
# This is now a paper-headline run, not a post-MCTS filter: it sweeps the full
# 16-model panel from Table 1 and writes per-model results into separate
# subdirs so scripts/10_aggregate_results.py can pick them up.
#
# Cost: ~$80-130 for the 16-model panel over ~500 scenarios (driven by
# Opus-4.7 and GPT-5; nano-tier rows are <$2 each). Use MODELS="..." to
# trim to a single cheap proxy when you just need a post-MCTS quality gate.
#
# Usage:
#   scripts/02_text_benchmark.sh                       # full 16-model panel
#   MODELS="openai/gpt-5-nano" scripts/02_text_benchmark.sh  # cheap filter
#
# Args:
#   $1  GENERATED_DIR    default: data/generated
#   $2  RESULTS_ROOT     default: data/results/text  (per-model subdirs inside)
#
# Env overrides:
#   MODELS         space-separated LiteLLM model strings (default: 16-model panel)
#   JUDGE_MODEL    default: google/gemma-4-31b-it  (set once, never change mid-sweep)
#   USE_OPENROUTER default: 0   (1 prepends `openrouter/` to every model)
#   MAX_CONCURRENCY default: 1  (run N models in parallel; respect rate limits)

set -euo pipefail
cd "$(dirname "$0")/.."

GENERATED_DIR="${1:-data/generated}"
RESULTS_ROOT="${2:-data/results/text}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-4-31b-it}"
USE_OPENROUTER="${USE_OPENROUTER:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

# 16-model panel matching Table 1 of the paper. Override with MODELS="..." to
# sweep a subset. Names follow LiteLLM provider routing; USE_OPENROUTER=1
# prefixes them with `openrouter/` at runtime so the same list works either
# through direct provider keys or through OpenRouter.
DEFAULT_MODELS=(
  # Closed-source
  "openai/gpt-5.4"
  "openai/gpt-5.4-mini"
  "x-ai/grok-4.3"
  "qwen/qwen3.6-max-preview"
  "google/gemini-3.1-pro-preview"
  "google/gemini-3-flash-preview"
  "anthropic/claude-opus-4.7"
  "anthropic/claude-sonnet-4.6"
  # Open-weight
  "moonshotai/kimi-k2.6"
  "minimax/minimax-m2.7"
  "qwen/qwen3.6-35b-a3b"
  "google/gemma-4-26b-a4b-it"
  "z-ai/glm-5.1"
  "openai/gpt-oss-120b"
  "deepseek/deepseek-v4-pro"
)
MODELS_RAW="${MODELS:-${DEFAULT_MODELS[*]}}"
read -ra MODELS_ARRAY <<< "$MODELS_RAW"

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

export AGENTCI_ALLOW_HEURISTIC_FALLBACK="${AGENTCI_ALLOW_HEURISTIC_FALLBACK:-0}"
export AGENTCI_LITELLM_TIMEOUT_SECONDS="${AGENTCI_LITELLM_TIMEOUT_SECONDS:-120}"
export AGENTCI_LITELLM_RETRIES="${AGENTCI_LITELLM_RETRIES:-6}"
# Allow up to 5-min waits so a Retry-After:300 doesn't exhaust all retries immediately.
export AGENTCI_LITELLM_BACKOFF_MAX_SECONDS="${AGENTCI_LITELLM_BACKOFF_MAX_SECONDS:-360}"

mkdir -p "$RESULTS_ROOT"

scenario_count="$(ls -1 "$GENERATED_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')"

echo "==> Reasoning (text-only) sweep"
echo "    scenarios   : $GENERATED_DIR ($scenario_count)"
echo "    models      : ${#MODELS_ARRAY[@]}"
for m in "${MODELS_ARRAY[@]}"; do echo "                  - $m"; done
echo "    judge       : $JUDGE_MODEL"
echo "    results     : $RESULTS_ROOT/<model_slug>/"
echo "    concurrency : $MAX_CONCURRENCY"
echo

# Slug-ify a model string for use as a directory name.
slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9._-]+#-#g; s#^-+|-+$##g'
}

run_one_model() {
  local model="$1"
  local model_slug
  model_slug="$(slug "$model")"
  local out_dir="$RESULTS_ROOT/$model_slug"
  mkdir -p "$out_dir"
  echo "==> $model -> $out_dir"
  uv run python -m eval.run_benchmark \
    --generated-dir "$GENERATED_DIR" \
    --results-dir "$out_dir" \
    --proxy-model "$model" \
    --judge-model "$JUDGE_MODEL" \
    --no-progress
}

failures=()
if [ "$MAX_CONCURRENCY" -le 1 ]; then
  for model in "${MODELS_ARRAY[@]}"; do
    if ! run_one_model "$model"; then
      echo "  FAILED: $model" >&2
      failures+=("$model")
    fi
  done
else
  # Bounded-parallel fan-out. Each background job is one model.
  pids=()
  models_for_pids=()
  active=0
  for model in "${MODELS_ARRAY[@]}"; do
    if [ "$active" -ge "$MAX_CONCURRENCY" ]; then
      # wait for any one to finish
      if ! wait -n; then
        :  # collect failures after the join
      fi
      active=$((active - 1))
    fi
    run_one_model "$model" &
    pids+=("$!")
    models_for_pids+=("$model")
    active=$((active + 1))
  done
  # Drain remaining jobs and collect failures.
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failures+=("${models_for_pids[$i]}")
    fi
  done
fi

echo
echo "==> Reasoning sweep complete."
echo "    per-model summaries: $RESULTS_ROOT/<model_slug>/summary__*.json"
echo "    aggregate next:      uv run python scripts/10_aggregate_results.py"
if [ "${#failures[@]}" -gt 0 ]; then
  echo "    failed models: ${failures[*]}" >&2
  exit 2
fi
