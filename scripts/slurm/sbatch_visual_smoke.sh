#!/usr/bin/env bash
# Visual benchmark smoke test on slurm. Runs ONE scenario end-to-end with one
# model in ui_only mode to confirm BrowserGym + headless Chromium + the agent
# config + judge model all work on the cluster. Costs <$0.20.
#
# Submit with (no --array; single job):
#   sbatch scripts/slurm/sbatch_visual_smoke.sh
#
# Override the scenario or model:
#   sbatch --export=ALL,SMOKE_SCENARIO=data/eval_set_e2e_30/<file>.json,SMOKE_MODEL=openai/gpt-5-mini \
#     scripts/slurm/sbatch_visual_smoke.sh
#
#SBATCH --job-name=agentci-vsmoke
#SBATCH --output=logs/slurm/vsmoke-%j.out
#SBATCH --error=logs/slurm/vsmoke-%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=36:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm data/results/visual_smoke

# Pick a default scenario: first JSON in data/generated_merged/ (frozen pool).
SMOKE_SCENARIO="${SMOKE_SCENARIO:-$(ls -1 data/generated_merged/*.json 2>/dev/null | grep -v dedup_stats | grep -v manifest | head -n 1)}"
SMOKE_MODEL="${SMOKE_MODEL:-anthropic/claude-sonnet-4.6}"
SMOKE_ACCESS_MODE="${SMOKE_ACCESS_MODE:-mixed}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-6}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-4-31b-it}"
PORT="${PORT:-13099}"
RESULTS_DIR="${RESULTS_DIR:-data/results/visual_smoke}"

if [ -z "${SMOKE_SCENARIO:-}" ] || [ ! -f "$SMOKE_SCENARIO" ]; then
  echo "ERROR: SMOKE_SCENARIO not set or file not found: '$SMOKE_SCENARIO'" >&2
  exit 1
fi

echo "==> Visual smoke test"
echo "    scenario   : $SMOKE_SCENARIO"
echo "    model      : $SMOKE_MODEL"
echo "    access     : $SMOKE_ACCESS_MODE"
echo "    max_steps  : $SMOKE_MAX_STEPS"
echo "    judge      : $JUDGE_MODEL"
echo "    port       : $PORT"
echo "    results    : $RESULTS_DIR"
echo

export AGENTCI_LITELLM_TIMEOUT_SECONDS="${AGENTCI_LITELLM_TIMEOUT_SECONDS:-180}"
export AGENTCI_LITELLM_RETRIES="${AGENTCI_LITELLM_RETRIES:-3}"

# Playwright's ldd-based dependency check respects LD_LIBRARY_PATH.
# Compute nodes lack libatk-bridge, libdrm, libgbm, libasound; the .so files
# are bundled under libs/playwright-deps/ (extracted from Ubuntu 20.04 debs).
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
export LD_LIBRARY_PATH="${REPO_ROOT}/libs/playwright-deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Resolve OpenRouter prefix if needed.
if [ "${USE_OPENROUTER:-1}" = "1" ]; then
  case "$SMOKE_MODEL" in openrouter/*) ;; *) SMOKE_MODEL="openrouter/$SMOKE_MODEL" ;; esac
  case "$JUDGE_MODEL" in openrouter/*) ;; *) JUDGE_MODEL="openrouter/$JUDGE_MODEL" ;; esac
fi

uv run playwright install chromium

uv run python -m eval.run_visual_benchmark \
  --scenario "$SMOKE_SCENARIO" \
  --access-mode "$SMOKE_ACCESS_MODE" \
  --judge-model "$JUDGE_MODEL" \
  --results-dir "$RESULTS_DIR" \
  --max-steps "$SMOKE_MAX_STEPS" \
  --use-litellm \
  --api-key-env OPENROUTER_API_KEY \
  --port "$PORT" \
  --model-name "$SMOKE_MODEL"

rc=$?
echo
if [ $rc -eq 0 ]; then
  echo "==> PASS: visual smoke completed"
  echo "Inspect run artifacts:"
  find "$RESULTS_DIR" -name "run_result.json" -newer "$0" 2>/dev/null | tail -5
else
  echo "==> FAIL (exit=$rc): see logs/slurm/vsmoke-${SLURM_JOB_ID:-?}.err"
fi
exit $rc
