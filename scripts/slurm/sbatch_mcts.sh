#!/usr/bin/env bash
# Slurm array job: one MCTS run per seed in $SEEDS_DIR (default
# data/seeds/).
#
# Submit with:
#   sbatch --array=0-$(($(ls $SEEDS_DIR/*.json | wc -l)-1)) \
#       --export=ALL,SEEDS_DIR=data/seeds,... scripts/slurm/sbatch_mcts.sh
#
# Knobs: same env vars as scripts/01_generate_scenarios.sh (USE_OPENROUTER,
# ITERATIONS, NODE_EXPANSION_LIMIT, RNG_SEED_BASE, AGENTCI_MUTATOR_MODEL,
# AGENTCI_JUDGE_MODEL, TARGET_A/B/C) plus:
#   SEEDS_DIR  default: data/seeds  (folder to glob *.json from)
# All work is API-bound; one CPU + 4 GB RAM per task is plenty.
#
#SBATCH --job-name=agentci-mcts
#SBATCH --output=logs/slurm/mcts-%A_%a.out
#SBATCH --error=logs/slurm/mcts-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=36:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm data/generated data/results/mcts_runs

SEEDS_DIR="${SEEDS_DIR:-data/seeds}"
shopt -s nullglob
SEEDS=( "$SEEDS_DIR"/*.json )
shopt -u nullglob
if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "ERROR: no *.json seeds found in '$SEEDS_DIR'" >&2
  exit 1
fi
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "${SEED:-}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for ${#SEEDS[@]} seeds" >&2
  exit 1
fi
echo "[$(date -Iseconds)] seed=$SEED  task=$SLURM_ARRAY_TASK_ID"

USE_OPENROUTER="${USE_OPENROUTER:-1}" \
ITERATIONS="${ITERATIONS:-35}" \
NODE_EXPANSION_LIMIT="${NODE_EXPANSION_LIMIT:-28}" \
RNG_SEED_BASE="${RNG_SEED_BASE:-42}" \
  scripts/01_generate_scenarios.sh "$SEED"
