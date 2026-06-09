# Contributing to AgentCIBench

Thanks for your interest in the benchmark. Contributions of all sizes are
welcome --- new scenarios, defense prompts, leaderboard submissions, bug
reports, and documentation fixes.

## Ground rules

- Be respectful. See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- The benchmark targets privacy-failure behaviour by design. Do not submit
  scenarios or prompts that rely on real personal data, real identifiers,
  or content that could be used to harm a specific person.
- New scenarios must be CC BY 4.0 licensable and use synthetic content only.

## Development setup

```bash
uv sync
uv run playwright install chromium
```

Run the offline test suite before opening a PR:

```bash
uv run pytest -q \
  tests/test_mcts_phase_a.py tests/test_prompts.py \
  tests/test_proxy_agent.py tests/test_reward_judge.py \
  tests/test_visual_benchmark.py
uv run ruff check src envs eval mcts prompts.py
```

## Submitting scenarios

1. Add the seed JSON under `data/seeds/` following the schema documented in
   `mcts/README.md`.
2. Include `must_share` and `must_not_share` lists that are exhaustive for
   the scenario state.
3. Open a PR describing which failure mode the seed targets and which
   CI parameter it stresses.

## Submitting leaderboard runs

1. Run `eval.run_benchmark` on `data/generated_merged` with the canonical
   judge model.
2. Open a PR adding your model row to `leaderboard/models.json` with a link
   to the per-scenario JSONL.
3. We will re-run a sampled subset to verify the headline numbers before
   merging.

## Security disclosures

Please email `anmol.goel@tu-darmstadt.de` rather than opening a public issue
for vulnerabilities or for scenarios that inadvertently leak real personal
information.
