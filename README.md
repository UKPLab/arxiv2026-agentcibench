# AgentCIBench

**Capable but Careless: Do Computer-Use Agents Follow Contextual Integrity?**

[![Paper](https://img.shields.io/badge/arXiv-pending-b31b1b.svg)](https://arxiv.org/abs/PENDING)
[![Data on HF](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow)](https://huggingface.co/datasets/UKPLab/AgentCIBench)
[![License: Apache 2.0](https://img.shields.io/badge/code-Apache_2.0-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/data-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

<video src="agentcibench-trajectory-video.mp4" controls width="100%" title="AgentCIBench trajectory overview"></video>

AgentCIBench is an evaluation harness that measures whether computer-use
agents (CUAs) respect **contextual integrity (CI)** when operating across
personal applications. It converts everyday cross-app requests into
executable, deterministically scored scenarios that target three failure
modes: **visual co-location**, **task-ambiguity overshare**, and
**recipient misalignment**.

We evaluate 15 frontier agents and find that 11 leak on more than 50% of
scenarios, with an average leakage of 67.9% — and the same failures
persist when agents act end-to-end in the rendered OpenApps UI.

- 📄 **Paper:** [arXiv:PENDING](https://arxiv.org/abs/PENDING)
- 🤗 **Dataset:** [huggingface.co/datasets/UKPLab/AgentCIBench](https://huggingface.co/datasets/UKPLab/AgentCIBench)
- 🌐 **Leaderboard / project page:** [ukplab.github.io/arxiv2026-agentcibench](https://ukplab.github.io/arxiv2026-agentcibench)

## Contents

```text
config/                Hydra app, task, agent, and defense configs
data/                  Local copy of the benchmark (also mirrored on Hugging Face)
envs/                  Scenario-to-OpenApps visual benchmark bridge
eval/                  Reasoning and visual benchmark runners
mcts/                  Scenario generation engine and CI scoring helpers
scripts/               End-to-end experiment scripts and aggregation tools
src/open_apps/         Local multi-app web environment used by visual runs
tests/                 Focused regression tests
docker/                Optional containerised runtime
```

## Quickstart

### Option A — local install (Python 3.11 + `uv`)

```bash
git clone https://github.com/UKPLab/arxiv2026-agentcibench.git
cd arxiv2026-agentcibench
uv sync
uv run playwright install chromium
```

Set provider keys (LiteLLM routes most calls):

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...  # or use direct provider keys
```

### Option B — Docker (recommended for reproducibility)

```bash
docker build -t agentcibench -f docker/Dockerfile .
docker run --rm -it \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v "$PWD/data:/app/data" \
  agentcibench bash
```

See [`docker/README.md`](docker/README.md) for `docker compose`, GPU notes,
and the headed-browser variant used for visual debugging.

### Pull data from Hugging Face (optional)

The repo ships a local copy of the scenarios under `data/`. To pull the
canonical version:

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('UKPLab/AgentCIBench')
ds['test'].to_json('data/generated_merged.jsonl')
ds['test_e2e'].to_json('data/eval_set_e2e_50.jsonl')
"
```

## Running the benchmark

### Reasoning (state-grounded) benchmark

```bash
uv run python -m eval.run_benchmark \
  --generated-dir data/generated_merged \
  --results-dir data/results/text_smoke/<model_slug> \
  --proxy-model openrouter/openai/gpt-5.4-mini \
  --judge-model openrouter/google/gemma-4-31b-it
```

Full sweep (matches the paper):

```bash
USE_OPENROUTER=1 \
MODELS="openai/gpt-5.4 anthropic/claude-sonnet-4.6 deepseek/deepseek-v4-pro" \
scripts/02_text_benchmark.sh data/generated_merged data/results/text
```

### Live UI benchmark

```bash
USE_OPENROUTER=1 scripts/05_visual_main.sh data/eval_set_e2e_50 data/results/visual_e2e
```

### Defense ablations

```bash
USE_OPENROUTER=1 \
MODELS="openai/gpt-5.4-mini deepseek/deepseek-v4-pro" \
scripts/07_text_defenses.sh data/eval_set_defenses data/results/text_defenses
```

### Regenerate scenarios

```bash
USE_OPENROUTER=1 OUTPUT_DIR=data/generated_new \
RUN_LOG_DIR=data/results/mcts_runs_new \
ITERATIONS=35 NODE_EXPANSION_LIMIT=28 \
scripts/01_generate_scenarios.sh
```

## Verifying the install (no paid API calls)

```bash
uv run python -m compileall src envs eval mcts prompts.py
uv run pytest tests/test_mcts_phase_a.py tests/test_prompts.py \
  tests/test_proxy_agent.py tests/test_reward_judge.py \
  tests/test_visual_benchmark.py
```

## Submitting to the leaderboard

We host a leaderboard at [ukplab.github.io/arxiv2026-agentcibench](https://ukplab.github.io/arxiv2026-agentcibench).
To submit your model, open a PR adding a row to `leaderboard/models.json` with a
link to the per-scenario JSONL output produced by `eval.run_benchmark`.

## Citation

```bibtex
@article{goel2026agentcibench,
  title   = {Capable but Careless: Do Computer-Use Agents Follow Contextual Integrity?},
  author  = {Goel, Anmol and others},
  journal = {arXiv preprint arXiv:PENDING},
  year    = {2026}
}
```

## Licensing

- **Code**: Apache License 2.0 (`LICENSE`)
- **Data and scenario pool**: CC BY 4.0 (see Hugging Face dataset card)
- **OpenApps environment assets**: included synthetic content, released under
  CC BY 4.0 alongside the data

## Responsible use

AgentCIBench targets privacy-failure behaviour by design. The released
scenarios are intended for pre-deployment evaluation, regression testing,
and mitigation research, not for soliciting harmful outputs. See the
Ethical Considerations section of the paper for the full discussion.

## Contributing

Issues and PRs are welcome. See `CONTRIBUTING.md` and the
[Code of Conduct](CODE_OF_CONDUCT.md). For security disclosures, please
email `anmol.goel@tu-darmstadt.de` rather than opening a public issue.

## Contact

- Anmol Goel — `anmol.goel@tu-darmstadt.de`
