#!/usr/bin/env bash
# Smoke test: confirms the streamlined pipeline runs end-to-end on one seed
# with cheap models and a tiny iteration count. Run this before any expensive
# generation/eval sweep to catch breakage early. Costs <$1.
#
# Usage:
#   scripts/00_smoke_test.sh
#
# Env overrides:
#   SEED=path/to/seed.json   (default: first seed in data/seeds/)
#   ITERATIONS=4             (default: 4)

set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-$(ls -1 data/seeds/*.json | head -n 1)}"
ITERATIONS="${ITERATIONS:-4}"
USE_OPENROUTER="${USE_OPENROUTER:-0}"
OUT_DIR="$(mktemp -d -t agentci_smoke_XXXXXX)/generated"
LOG_DIR="$(dirname "$OUT_DIR")/mcts_runs"

# Open-weight stack for the smoke test (matches scripts/01_generate_scenarios.sh
# defaults). Smoke uses 2 of the 3 production targets for speed.
export AGENTCI_MUTATOR_MODEL="${AGENTCI_MUTATOR_MODEL:-deepseek/deepseek-v4-pro}"
export AGENTCI_JUDGE_MODEL="${AGENTCI_JUDGE_MODEL:-google/gemma-4-31b-it}"
export AGENTCI_ALLOW_HEURISTIC_FALLBACK="${AGENTCI_ALLOW_HEURISTIC_FALLBACK:-0}"

TARGET_A="${TARGET_A:-qwen/qwen3.6-35b-a3b}"
TARGET_B="${TARGET_B:-moonshotai/kimi-k2.5}"

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
  export AGENTCI_MUTATOR_MODEL AGENTCI_JUDGE_MODEL
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "USE_OPENROUTER=1 but OPENROUTER_API_KEY is not set." >&2
    exit 1
  fi
fi

echo "==> Smoke test"
echo "    seed       : $SEED"
echo "    iterations : $ITERATIONS"
echo "    output     : $OUT_DIR"

uv run python -m mcts.mcts_engine \
  --seed "$SEED" \
  --iterations "$ITERATIONS" \
  --node-expansion-limit 3 \
  --rng-seed 7 \
  --output-dir "$OUT_DIR" \
  --run-log-dir "$LOG_DIR" \
  --target-model "$TARGET_A" \
  --target-model "$TARGET_B" \
  --target-aggregation mean \
  --diversity-weight 0.5

count="$(ls -1 "$OUT_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "==> Smoke produced $count scenario(s) in $OUT_DIR"
[ "$count" -gt 0 ] && echo "OK" || { echo "FAIL: no scenarios kept"; exit 1; }
