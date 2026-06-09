#!/usr/bin/env bash
# Slurm array job: one reasoning sweep job per model in the 16-model panel.
# Each job runs eval.run_benchmark for ONE model over the full generated pool
# and writes its summary into data/results/text/<model_slug>/.
#
# Submit with:
#   sbatch --array=0-15 scripts/slurm/sbatch_text_sweep.sh
#
# All work is API-bound (LiteLLM/OpenRouter); 1 CPU + 4 GB RAM per task.
# Wall-clock per model: 30-90 min for the cheap tier, 2-4 h for Opus/GPT-5
# on a ~500 scenario pool.
#
#SBATCH --job-name=agentci-text
#SBATCH --output=logs/slurm/text-%A_%a.out
#SBATCH --error=logs/slurm/text-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=36:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm

# Default panel (matches scripts/02_text_benchmark.sh DEFAULT_MODELS).
# Override via OVERRIDE_MODELS to ADD new models without editing this file:
#   sbatch --array=0-1 \
#     --export=ALL,OVERRIDE_MODELS="modelA modelB",USE_OPENROUTER=1 \
#     scripts/slurm/sbatch_text_sweep.sh
# Each model writes to data/results/text/<slug>/ — existing model dirs are
# untouched, so re-running the aggregator picks up the new rows.
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
read -ra MODELS <<< "${OVERRIDE_MODELS:-${DEFAULT_MODELS[*]}}"

if [ -z "${MODELS[$SLURM_ARRAY_TASK_ID]:-}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for ${#MODELS[@]} models" >&2
  exit 1
fi
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
GENERATED_DIR="${GENERATED_DIR:-data/generated_merged}"
RESULTS_ROOT="${RESULTS_ROOT:-data/results/text}"
echo "[$(date -Iseconds)] model=$MODEL  task=$SLURM_ARRAY_TASK_ID  dir=$GENERATED_DIR  results=$RESULTS_ROOT"

USE_OPENROUTER=1 MODELS="$MODEL" MAX_CONCURRENCY=1 \
  scripts/02_text_benchmark.sh "$GENERATED_DIR" "$RESULTS_ROOT"
