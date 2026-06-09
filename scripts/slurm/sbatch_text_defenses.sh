#!/usr/bin/env bash
# Slurm array job: one cell per (defense, model) of the reasoning defenses
# sweep. 3 defenses × 3 models = 9 tasks (array 0-8).
# "none" is NOT re-run here; synthesise it from the S2A sweep filtered to the
# defenses subset via scripts/10_aggregate_results.py --defenses-subset.
#
# Build the 50-scenario subset first (~17 per failure mode):
#   uv run python scripts/03_build_eval_set.py \
#     --input-dir data/generated_merged \
#     --output-dir data/eval_set_defenses_50 \
#     --per-mode-quotas visual_co_location:17,recipient_misalignment:17,task_ambiguity_overshare:16 \
#     --rng-seed 2027 --manifest data/eval_set_defenses_50/manifest.json
#
# Submit with:
#   sbatch --array=0-8 --export=ALL,EVAL_DIR=data/eval_set_defenses_50 \
#     scripts/slurm/sbatch_text_defenses.sh
#
#SBATCH --job-name=agentci-textdef
#SBATCH --output=logs/slurm/textdef-%A_%a.out
#SBATCH --error=logs/slurm/textdef-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=36:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm

DEFENSES=( "restrictive" "rubric_informed" "recipient_typed" )
# 3 models × 3 defenses = 9 array tasks (0-8). Edit this list to pick your panel.
# Chosen to span the leak-rate range in the S2A ranking:
#   - Opus 4.7  : 13.7% L  (safest ceiling — does any defense improve further?)
#   - GPT-5.4   : 18.8% L, 42% refusal (high-refusal — defenses may shift refusal vs. leak tradeoff)
#   - DeepSeek-v4-pro: 82.9% L (most room to move; tests defense floor for unsafe models)
MODELS=(
  "anthropic/claude-opus-4.7"
  "openai/gpt-5.4"
  "deepseek/deepseek-v4-pro"
)

# Row-major: task = defense_idx * len(MODELS) + model_idx
N_MODELS=${#MODELS[@]}
DEF_IDX=$(( SLURM_ARRAY_TASK_ID / N_MODELS ))
MOD_IDX=$(( SLURM_ARRAY_TASK_ID % N_MODELS ))
DEFENSE="${DEFENSES[$DEF_IDX]}"
MODEL="${MODELS[$MOD_IDX]}"

# Slug the model for the per-cell output dir (must match scripts/07's slug()).
slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9._-]+#-#g; s#^-+|-+$##g'
}
MODEL_SLUG="$(slug "$MODEL")"

EVAL_DIR="${EVAL_DIR:-data/eval_set_defenses_50}"
OUT_DIR="data/results/text_defenses/$DEFENSE/$MODEL_SLUG"
mkdir -p "$OUT_DIR"

# Resolve OpenRouter routing inline so each cell is self-contained.
if [ "${USE_OPENROUTER:-1}" = "1" ]; then
  case "$MODEL" in openrouter/*) ;; *) MODEL="openrouter/$MODEL" ;; esac
  JUDGE="${JUDGE_MODEL:-openrouter/google/gemma-4-31b-it}"
else
  JUDGE="${JUDGE_MODEL:-google/gemma-4-31b-it}"
fi

echo "[$(date -Iseconds)] defense=$DEFENSE  model=$MODEL  -> $OUT_DIR"

uv run python -m eval.run_benchmark \
  --generated-dir "$EVAL_DIR" \
  --results-dir "$OUT_DIR" \
  --proxy-model "$MODEL" \
  --judge-model "$JUDGE" \
  --defense "$DEFENSE" \
  --no-progress
