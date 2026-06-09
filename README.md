# AgentCIBench Artifact

This directory is the anonymized code and data artifact for the
submission "Capable but Careless: Do Computer-Use Agents Follow Contextual
Integrity?" It contains the code needed to regenerate scenarios, run the
reasoning benchmark, run the live OpenApps visual benchmark, run prompt-defense
ablations, and aggregate results.

## Contents

```text
config/                    Hydra app, task, agent, and defense configs
data/generated_merged/      Final 117-scenario AgentCIBench benchmark pool
data/eval_set_e2e_50/       50-scenario subset used for live UI evaluation
data/seeds/                 Hand-written seed scenarios for generation
envs/                       Scenario-to-OpenApps visual benchmark bridge
eval/                       Reasoning and visual benchmark runners
mcts/                       Scenario generation engine and CI scoring helpers
scripts/                    End-to-end experiment scripts and aggregation tools
src/open_apps/              Local multi-app web environment used by visual runs
tests/                      Focused regression tests for the included code
```

The final benchmark pool contains 117 JSON scenarios:

| Failure mode | Count |
| --- | ---: |
| task_ambiguity_overshare | 75 |
| recipient_misalignment | 24 |
| visual_co_location | 18 |

The pool is derived from 36 seed files, including 28 seeds represented in the
final benchmark pool, and spans messenger, todo,
calendar, maps, code editor, and shopping states. The live UI subset
(`data/eval_set_e2e_50/`) contains 50 scenarios and avoids the shopping app so
that reviewers can run the visual benchmark without the optional webshop index.

## Setup

Requirements:

- Python 3.11
- `uv`
- Playwright Chromium for live visual runs
- API keys for the model providers you evaluate

Install dependencies:

```bash
uv sync
uv run playwright install chromium
```

Most model calls go through LiteLLM. Set either direct provider keys or an
OpenRouter key:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...
```

When using OpenRouter, pass `USE_OPENROUTER=1` to the shell scripts or use
`openrouter/<provider>/<model>` model strings directly.

## Quick Verification

Run the local checks that do not require paid model calls:

```bash
uv run python -m compileall src envs eval mcts prompts.py
uv run pytest tests/test_mcts_phase_a.py tests/test_prompts.py \
  tests/test_proxy_agent.py tests/test_reward_judge.py \
  tests/test_visual_benchmark.py
```

Inspect the benchmark pool:

```bash
uv run python scripts/03_build_eval_set.py \
  --input-dir data/generated_merged \
  --output-dir /tmp/agentcibench_probe \
  --total 12 \
  --rng-seed 2026 \
  --manifest /tmp/agentcibench_probe/manifest.json
```

## Run the Reasoning Benchmark

The reasoning benchmark evaluates a model's final disclosure decision from the
scenario JSON state. It does not launch a browser.

Single-model smoke run on the final 117-scenario pool:

```bash
uv run python -m eval.run_benchmark \
  --generated-dir data/generated_merged \
  --results-dir data/results/text_smoke/<model_slug> \
  --proxy-model openrouter/openai/gpt-5.4-mini \
  --judge-model openrouter/google/gemma-4-31b-it \
  --no-progress
```

Full panel runner:

```bash
USE_OPENROUTER=1 \
MODELS="openai/gpt-5.4 anthropic/claude-sonnet-4.6 deepseek/deepseek-v4-pro" \
scripts/02_text_benchmark.sh data/generated_merged data/results/text
```

The default model list in `scripts/02_text_benchmark.sh` matches the paper's
main sweep. Override `MODELS` to run a smaller or different model panel.

## Run the Live UI Benchmark

The visual benchmark launches the local OpenApps web environment, lets an agent
act in the rendered UI, captures the final app state, and scores outbound
content against each scenario's ground truth.

Single scenario:

```bash
uv run python -m eval.run_visual_benchmark \
  --scenario data/eval_set_e2e_50/seed_manager_summary_todo_001__ambiguity_trap__28044567.json \
  --results-dir data/results/visual_smoke \
  --runtime-root data/runtime_openapps \
  --agent-model openrouter/anthropic/claude-sonnet-4.6 \
  --judge-model openrouter/google/gemma-4-31b-it \
  --access-mode mixed \
  --use-litellm \
  --api-key-env OPENROUTER_API_KEY \
  --max-steps 35
```

Batch runner:

```bash
USE_OPENROUTER=1 \
scripts/05_visual_main.sh data/eval_set_e2e_50 data/results/visual_e2e
```

Visual runs are more expensive and slower than reasoning runs. The output
directory will contain per-scenario run artifacts and a `benchmark_results__*.jsonl`.

## Run Defense Ablations

Defense prompts are in `config/defenses/`.

```bash
uv run python scripts/03_build_eval_set.py \
  --input-dir data/generated_merged \
  --output-dir data/eval_set_defenses \
  --total 70 \
  --rng-seed 2026 \
  --manifest data/eval_set_defenses/manifest.json

USE_OPENROUTER=1 \
MODELS="openai/gpt-5.4-mini deepseek/deepseek-v4-pro" \
scripts/07_text_defenses.sh data/eval_set_defenses data/results/text_defenses
```

## Regenerate Scenarios

The final pool is already included. To generate new scenarios from the included
seeds:

```bash
USE_OPENROUTER=1 \
OUTPUT_DIR=data/generated_new \
RUN_LOG_DIR=data/results/mcts_runs_new \
ITERATIONS=35 \
NODE_EXPANSION_LIMIT=28 \
scripts/01_generate_scenarios.sh
```

To merge and deduplicate multiple generated directories:

```bash
uv run python scripts/11_redo_dedup.py \
  --input-dir data/generated_new \
  --input-dir data/generated_merged \
  --output-dir data/generated_deduped \
  --stats-path data/dedup_stats_new.json
```

## Aggregate Results

After running benchmarks, regenerate compact tables:

```bash
uv run python scripts/10_aggregate_results.py \
  --results-root data/results \
  --out-dir data/results/aggregated \
  --scenario-dirs data/generated_merged data/eval_set_e2e_50

uv run python scripts/11_bootstrap_ci.py \
  --results-root data/results \
  --out-dir data/results/aggregated
```

No result files are included in this artifact. Raw reasoning outputs, visual
outputs, and aggregate tables can be regenerated with the commands above.
