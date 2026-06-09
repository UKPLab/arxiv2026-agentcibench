#!/usr/bin/env bash
# MCTS generation across every seed in data/seeds/.
#
# Designed to be re-runnable: each seed gets a deterministic per-seed RNG so
# adding new seed JSONs later just runs MCTS for those new files (existing
# scenario files are not overwritten unless their scenario_id collides).
#
# Cost: ~$2-3 per seed at default settings (mutator gpt-5, judge gpt-5-mini,
# 3 cheap target proxies). Tune ITERATIONS / NODE_EXPANSION_LIMIT to scale.
#
# Yield: with 19 seeds, defaults below produce ~430-500 accepted scenarios
# (NODE_EXPANSION_LIMIT * seeds, minus near-duplicate filter and threshold drops).
#
# Usage:
#   scripts/01_generate_scenarios.sh                       # all seeds
#   scripts/01_generate_scenarios.sh path/to/seed.json ... # specific files
#
# Env overrides:
#   SEEDS_DIR             default: data/seeds
#   OUTPUT_DIR            default: data/generated
#   RUN_LOG_DIR           default: data/results/mcts_runs
#   ITERATIONS            default: 35
#   NODE_EXPANSION_LIMIT  default: 28
#   RNG_SEED_BASE         default: 42  (per-seed offset added)
#   USE_OPENROUTER        default: 0   (set to 1 to prepend `openrouter/` to all
#                                       model strings so LiteLLM routes through
#                                       OpenRouter instead of the direct provider)
#   MAX_CONCURRENCY       default: 1   (number of seeds processed in parallel;
#                                       set to 3 or 4 to fan out across seeds
#                                       without hammering the API. Each seed
#                                       fires ~1-2 RPM internally, so concurrency
#                                       N peaks at ~2N RPM aggregate per model.)
#   AGENTCI_MUTATOR_MODEL default: deepseek/deepseek-v4-pro
#   AGENTCI_JUDGE_MODEL   default: google/gemma-4-31b-it
#   TARGET_A/B/C          default: qwen/qwen3.6-35b-a3b,
#                                  moonshotai/kimi-k2.5,
#                                  minimax/minimax-m2.5

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS_DIR="${SEEDS_DIR:-data/seeds}"
OUTPUT_DIR="${OUTPUT_DIR:-data/generated}"
RUN_LOG_DIR="${RUN_LOG_DIR:-data/results/mcts_runs}"
ITERATIONS="${ITERATIONS:-35}"
NODE_EXPANSION_LIMIT="${NODE_EXPANSION_LIMIT:-28}"
RNG_SEED_BASE="${RNG_SEED_BASE:-42}"
USE_OPENROUTER="${USE_OPENROUTER:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
# Optional explicit keep_threshold; empty = let the engine auto-derive from
# target count and aggregation (default 3.0 for 3 targets at mean aggregation).
KEEP_THRESHOLD="${KEEP_THRESHOLD:-}"

# Open-weight stack for the MCTS engine: mutator + judge + three diverse
# target proxies. Closed-weight frontier agents are never used at generation
# time, which is what the paper's "open-weight proxies, closed-weight
# evaluation" transfer claim requires.
export AGENTCI_MUTATOR_MODEL="${AGENTCI_MUTATOR_MODEL:-deepseek/deepseek-v4-pro}"
export AGENTCI_JUDGE_MODEL="${AGENTCI_JUDGE_MODEL:-google/gemma-4-31b-it}"
export AGENTCI_ALLOW_HEURISTIC_FALLBACK="${AGENTCI_ALLOW_HEURISTIC_FALLBACK:-0}"
export AGENTCI_LITELLM_TIMEOUT_SECONDS="${AGENTCI_LITELLM_TIMEOUT_SECONDS:-120}"
export AGENTCI_LITELLM_RETRIES="${AGENTCI_LITELLM_RETRIES:-6}"

# Three diverse open-weight target proxies for MCTS rollouts.
TARGET_A="${TARGET_A:-qwen/qwen3.6-35b-a3b}"
TARGET_B="${TARGET_B:-moonshotai/kimi-k2.5}"
TARGET_C="${TARGET_C:-minimax/minimax-m2.5}"

# When USE_OPENROUTER=1, prepend `openrouter/` to every model name so LiteLLM
# routes the call through OpenRouter (requires OPENROUTER_API_KEY exported).
if [ "$USE_OPENROUTER" = "1" ]; then
  prefix_if_unprefixed() {
    case "$1" in
      openrouter/*) printf '%s' "$1" ;;
      *)            printf 'openrouter/%s' "$1" ;;
    esac
  }
  AGENTCI_MUTATOR_MODEL="$(prefix_if_unprefixed "$AGENTCI_MUTATOR_MODEL")"
  AGENTCI_JUDGE_MODEL="$(prefix_if_unprefixed "$AGENTCI_JUDGE_MODEL")"
  TARGET_A="$(prefix_if_unprefixed "$TARGET_A")"
  TARGET_B="$(prefix_if_unprefixed "$TARGET_B")"
  TARGET_C="$(prefix_if_unprefixed "$TARGET_C")"
  export AGENTCI_MUTATOR_MODEL AGENTCI_JUDGE_MODEL
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "USE_OPENROUTER=1 but OPENROUTER_API_KEY is not set." >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR" "$RUN_LOG_DIR"

# Seed list: explicit args win, else glob the directory.
if [ "$#" -gt 0 ]; then
  SEED_FILES=("$@")
else
  shopt -s nullglob
  SEED_FILES=("$SEEDS_DIR"/*.json)
  shopt -u nullglob
fi

if [ "${#SEED_FILES[@]}" -eq 0 ]; then
  echo "No seed JSON files found in $SEEDS_DIR" >&2
  exit 1
fi

echo "==> MCTS generation"
echo "    seeds            : ${#SEED_FILES[@]}"
echo "    iterations/seed  : $ITERATIONS"
echo "    expansion limit  : $NODE_EXPANSION_LIMIT"
echo "    targets          : $TARGET_A | $TARGET_B | $TARGET_C"
echo "    mutator/judge    : $AGENTCI_MUTATOR_MODEL / $AGENTCI_JUDGE_MODEL"
echo "    output           : $OUTPUT_DIR"
echo

KEEP_THRESHOLD_ARGS=()
if [ -n "$KEEP_THRESHOLD" ]; then
  KEEP_THRESHOLD_ARGS=(--keep-threshold "$KEEP_THRESHOLD")
fi

i=0
failures=()
for seed_path in "${SEED_FILES[@]}"; do
  if [ ! -f "$seed_path" ]; then
    echo "  skip (missing): $seed_path" >&2
    continue
  fi
  seed_name="$(basename "$seed_path" .json)"
  rng_seed=$((RNG_SEED_BASE + i))
  i=$((i + 1))
  echo "==> [$i/${#SEED_FILES[@]}] $seed_name (rng-seed=$rng_seed)"
  if uv run python -m mcts.mcts_engine \
      --seed "$seed_path" \
      --iterations "$ITERATIONS" \
      --node-expansion-limit "$NODE_EXPANSION_LIMIT" \
      --rng-seed "$rng_seed" \
      --output-dir "$OUTPUT_DIR" \
      --run-log-dir "$RUN_LOG_DIR" \
      --target-model "$TARGET_A" \
      --target-model "$TARGET_B" \
      --target-model "$TARGET_C" \
      --target-aggregation mean \
      --threshold-aware-search \
      --diversity-weight 0.5 \
      "${KEEP_THRESHOLD_ARGS[@]}"; then
    :
  else
    echo "  FAILED: $seed_name (continuing with remaining seeds)" >&2
    failures+=("$seed_name")
  fi
done

total="$(ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')"
echo
echo "==> Generation complete"
echo "    scenarios in $OUTPUT_DIR : $total"
if [ "${#failures[@]}" -gt 0 ]; then
  echo "    failed seeds: ${failures[*]}"
  exit 2
fi
