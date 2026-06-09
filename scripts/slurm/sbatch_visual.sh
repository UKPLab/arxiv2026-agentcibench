#!/usr/bin/env bash
# Slurm array job: one visual benchmark cell per model.
#
# Visual rollouts run BrowserGym + headless Chromium inside the slurm task,
# so each task needs (a) a writable HOME (uv/playwright cache) and (b) its
# own port for the OpenApps server. We assign port = 13000 + task_id so
# concurrent array tasks don't collide on the loopback.
#
# E2E main (2 models, mixed access, N=50 scenarios):
#   sbatch --array=0-1 \
#     --export=ALL,EVAL_DIR=data/eval_set_e2e_50,RESULTS_ROOT=data/results/visual_mixed \
#     scripts/slurm/sbatch_visual.sh
#
# Override model list via env:
#   OVERRIDE_MODELS="modelA modelB"  (space-separated)
#
#SBATCH --job-name=agentci-visual
#SBATCH --output=logs/slurm/visual-%A_%a.out
#SBATCH --error=logs/slurm/visual-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=36:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm

# E2E main panel (2 models, mixed access).  Override via OVERRIDE_MODELS.
DEFAULT_MAIN_MODELS=(
  "anthropic/claude-opus-4.7"
  "anthropic/claude-sonnet-4.6"
)

# Apply env override if set.
read -ra MAIN_MODELS <<< "${OVERRIDE_MODELS:-${DEFAULT_MAIN_MODELS[*]}}"

MODEL="${MAIN_MODELS[$SLURM_ARRAY_TASK_ID]}"
ACCESS_MODE="${ACCESS_MODE:-mixed}"
RESULTS_ROOT_DEFAULT="data/results/visual_mixed"

EVAL_DIR="${EVAL_DIR:-data/eval_set_e2e_50}"
RESULTS_ROOT="${RESULTS_ROOT:-$RESULTS_ROOT_DEFAULT}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-4-31b-it}"
MAX_STEPS="${MAX_STEPS:-20}"
PORT=$(( 13000 + SLURM_ARRAY_TASK_ID ))

# Resolve OpenRouter routing inline so each cell is self-contained.
if [ "${USE_OPENROUTER:-1}" = "1" ]; then
  case "$MODEL"      in openrouter/*) ;; *) MODEL="openrouter/$MODEL"           ;; esac
  case "$JUDGE_MODEL" in openrouter/*) ;; *) JUDGE_MODEL="openrouter/$JUDGE_MODEL" ;; esac
  API_KEY_ENV="OPENROUTER_API_KEY"
else
  API_KEY_ENV="${API_KEY_ENV:-OPENROUTER_API_KEY}"
fi

mkdir -p "$RESULTS_ROOT"
echo "[$(date -Iseconds)] model=$MODEL access=$ACCESS_MODE port=$PORT eval=$EVAL_DIR"

# Compute nodes lack libatk-bridge, libdrm, libgbm, libasound; bundled .so files
# live under libs/playwright-deps/ (same fix as sbatch_visual_smoke.sh).
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
export LD_LIBRARY_PATH="${REPO_ROOT}/libs/playwright-deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

uv run playwright install chromium

export AGENTCI_LITELLM_TIMEOUT_SECONDS="${AGENTCI_LITELLM_TIMEOUT_SECONDS:-180}"
export AGENTCI_LITELLM_RETRIES="${AGENTCI_LITELLM_RETRIES:-6}"

uv run python -m eval.run_visual_benchmark \
  --generated-dir "$EVAL_DIR" \
  --access-mode "$ACCESS_MODE" \
  --judge-model "$JUDGE_MODEL" \
  --results-dir "$RESULTS_ROOT" \
  --max-steps "$MAX_STEPS" \
  --use-litellm \
  --api-key-env "$API_KEY_ENV" \
  --port "$PORT" \
  --model-name "$MODEL"
