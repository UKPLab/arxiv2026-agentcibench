#!/usr/bin/env bash
# Slurm array job: shard the variant-augmentation work across N tasks.
#
# Each task processes every Nth source scenario via the --shard I/N flag,
# so 8-way parallelism cuts wall-clock from ~12h sequential to ~1.5h.
# Each shard writes its survivors into the SAME output dir; variant IDs are
# unique by construction (augvar_<source_id>_<idx>) so no collisions.
#
# Submit with:
#   sbatch --array=0-7%8 scripts/slurm/sbatch_augment.sh
#
# Knobs (export as env or set via --export=):
#   INPUT_DIR            default: data/generated
#   OUTPUT_DIR           default: data/generated_augmented
#   VARIANTS_PER_SOURCE  default: 5
#   KEEP_THRESHOLD       default: 2.5
#   GENERATOR_MODEL      default: (uses AGENTCI_MUTATOR_MODEL = deepseek-v4-pro)
#   SHARDS               default: 8  (must match the --array=0-(N-1)%N you submit)
#   USE_OPENROUTER       default: 1
#
#SBATCH --job-name=agentci-aug
#SBATCH --output=logs/slurm/aug-%A_%a.out
#SBATCH --error=logs/slurm/aug-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=36:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm

INPUT_DIR="${INPUT_DIR:-data/generated}"
OUTPUT_DIR="${OUTPUT_DIR:-data/generated_augmented}"
VARIANTS_PER_SOURCE="${VARIANTS_PER_SOURCE:-7}"
KEEP_THRESHOLD="${KEEP_THRESHOLD:-2.5}"
SHARDS="${SHARDS:-8}"
USE_OPENROUTER="${USE_OPENROUTER:-1}"

export AGENTCI_MUTATOR_MODEL="${AGENTCI_MUTATOR_MODEL:-deepseek/deepseek-v4-pro}"
export AGENTCI_JUDGE_MODEL="${AGENTCI_JUDGE_MODEL:-google/gemma-4-31b-it}"
export AGENTCI_ALLOW_HEURISTIC_FALLBACK="${AGENTCI_ALLOW_HEURISTIC_FALLBACK:-0}"
export AGENTCI_LITELLM_TIMEOUT_SECONDS="${AGENTCI_LITELLM_TIMEOUT_SECONDS:-120}"
export AGENTCI_LITELLM_RETRIES="${AGENTCI_LITELLM_RETRIES:-6}"

# Three diverse open-weight target proxies for MCTS rollouts.
TARGET_A="${TARGET_A:-qwen/qwen3.6-35b-a3b}"
TARGET_B="${TARGET_B:-moonshotai/kimi-k2.5}"
TARGET_C="${TARGET_C:-minimax/minimax-m2.5}"

GEN_FLAG=()
if [ -n "${GENERATOR_MODEL:-}" ]; then
  GEN_FLAG=(--generator-model "$GENERATOR_MODEL")
fi
OR_FLAG=()
[ "$USE_OPENROUTER" = "1" ] && OR_FLAG=(--use-openrouter)

mkdir -p "$OUTPUT_DIR"

echo "[$(date -Iseconds)] augment shard ${SLURM_ARRAY_TASK_ID}/$SHARDS"
echo "  input=$INPUT_DIR"
echo "  output=$OUTPUT_DIR"
echo "  variants_per_source=$VARIANTS_PER_SOURCE  keep_threshold=$KEEP_THRESHOLD"
echo "  generator_model=${GENERATOR_MODEL:-(default=AGENTCI_MUTATOR_MODEL)}"

uv run python scripts/12_augment_variants.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --variants-per-source "$VARIANTS_PER_SOURCE" \
  --keep-threshold "$KEEP_THRESHOLD" \
  --shard "${SLURM_ARRAY_TASK_ID}/${SHARDS}" \
  "${GEN_FLAG[@]}" \
  "${OR_FLAG[@]}"
