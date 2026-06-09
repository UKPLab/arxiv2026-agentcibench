# scripts/

End-to-end runners for the AgentCIBench artifact. The root `README.md`
contains the canonical reproduction commands; this file is a compact script
reference.

| # | Script | Purpose |
|---|---|---|
| 00 | `00_smoke_test.sh` | One seed with a short MCTS run to confirm the generation pipeline works. |
| 01 | `01_generate_scenarios.sh` | MCTS scenario generation over `data/seeds/`. |
| 02 | `02_text_benchmark.sh` | Reasoning/text-only sweep over a scenario directory. |
| 03 | `03_build_eval_set.py` | Stratified subset builder by scenario family and source seed. |
| 04 | `04_visual_pilot.sh` | Small live OpenApps visual run for harness validation. |
| 05 | `05_visual_main.sh` | Live OpenApps visual benchmark runner. |
| 07 | `07_text_defenses.sh` | Reasoning-only defense sweep over `config/defenses/`. |
| 10 | `10_aggregate_results.py` | Aggregate generated benchmark outputs into tables. |
| 11 | `11_bootstrap_ci.py` | Bootstrap confidence intervals for aggregate result tables. |
| 11 | `11_redo_dedup.py` | Merge and deduplicate generated scenario directories. |
| 12 | `12_augment_variants.py` | Optional LLM-based scenario variant generation. |
| 12 | `12_plot_results.py` | Optional figure generation from aggregate CSVs. |
| 13 | `13_expand_run.sh` | Optional expansion wrapper for additional generation passes. |
| 13 | `13_power_analysis.py` | Optional power-analysis table generation. |
| -- | `slurm/sbatch_*.sh` | Slurm array wrappers for generation, reasoning, defenses, and visual runs. |

## Minimal Flow

```bash
USE_OPENROUTER=1 scripts/00_smoke_test.sh
USE_OPENROUTER=1 scripts/01_generate_scenarios.sh
USE_OPENROUTER=1 MAX_CONCURRENCY=4 scripts/02_text_benchmark.sh data/generated_merged data/results/text

uv run python scripts/03_build_eval_set.py \
  --input-dir data/generated_merged \
  --output-dir data/eval_set_defenses \
  --total 70 \
  --rng-seed 2026 \
  --manifest data/eval_set_defenses/manifest.json

USE_OPENROUTER=1 scripts/07_text_defenses.sh data/eval_set_defenses data/results/text_defenses
uv run python scripts/10_aggregate_results.py
```

## Adding a Mutation Strategy

When a new family is added to `mcts/mutators.py` and `prompts.py`,
`03_build_eval_set.py` discovers it by scanning the generated pool. To add a
strategy:

1. Append to `LOCAL_MUTATION_STRATEGIES` in `mcts/mutators.py`.
2. Add a heuristic fallback and register it in `_MUTATION_FNS`.
3. Add the LLM mutation prompt in `prompts.py:MUTATION_STRATEGY_PROMPTS`.
4. Add `MUTATION_STRATEGY_TO_FAILURE_MODE` metadata.
5. Re-run `scripts/01_generate_scenarios.sh`.

## Defenses

`scripts/07_text_defenses.sh` calls `eval/run_benchmark.py --defense
{none,restrictive,rubric_informed,recipient_typed}`. Defense prompt files live
in `config/defenses/`.

```bash
DEFENSES="restrictive rubric_informed" \
MODELS="openai/gpt-5" \
USE_OPENROUTER=1 \
scripts/07_text_defenses.sh
```

## Environment

All scripts assume `uv sync` has been run. Model calls use LiteLLM provider
strings. Set direct provider keys or route through OpenRouter:

```bash
export OPENROUTER_API_KEY=...
USE_OPENROUTER=1 scripts/00_smoke_test.sh
USE_OPENROUTER=1 scripts/01_generate_scenarios.sh
USE_OPENROUTER=1 scripts/02_text_benchmark.sh data/generated_merged data/results/text
```
