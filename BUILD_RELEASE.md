# Building the public GitHub release

This folder (`agentcibench_release/`) contains the **release-only files**
that differ from the anonymised `submission/` artifact: a de-anonymised
README, citation metadata, Docker setup, CI workflow, contribution and
licensing files, and `.gitignore`.

To assemble the public repository:

```bash
# 1. Start from a clean target directory
mkdir -p ~/code/agentcibench-public
cd ~/code/agentcibench-public
git init -b main

# 2. Copy the runnable artifact (code, configs, tests, sample data)
rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  /Users/anmolg/Desktop/OpenApps/submission/ ./

# 3. Overlay the release files (this folder) — these take precedence over
#    any files in submission/ with the same name (README.md, LICENSE)
rsync -a /Users/anmolg/Desktop/OpenApps/agentcibench_release/ ./

# 4. Sanity check: no API keys, no anonymised URLs, no logs
grep -RIn --exclude-dir=.git -E '(sk-[A-Za-z0-9_-]{20,}|anonymous|ANONYMIZED)' . || echo "clean"
gitleaks detect --source . --no-banner

# 5. Smoke test
uv sync
uv run python -m compileall src envs eval mcts prompts.py
uv run pytest tests/test_mcts_phase_a.py tests/test_prompts.py \
  tests/test_proxy_agent.py tests/test_reward_judge.py \
  tests/test_visual_benchmark.py

# 6. First commit, push, tag
git add .
git commit -m "Initial public release: AgentCIBench v1.0.0"
git remote add origin git@github.com:agentcibench/agentcibench.git
git push -u origin main
git tag -a v1.0.0 -m "AgentCIBench v1.0.0 (arXiv release)"
git push origin v1.0.0
```

## Checklist before pushing

- [ ] All authors listed in `CITATION.cff` and paper match.
- [ ] arXiv ID and HF dataset URL filled into `README.md` and the paper.
- [ ] `LICENSE` (Apache 2.0 for code, CC BY 4.0 note for data) reviewed.
- [ ] `data/` does not contain reviewer-specific artefacts or `dedup_stats`
      files referencing internal infra.
- [ ] `gitleaks` / `trufflehog` returns clean.
- [ ] `.env` and any local config files are gitignored.
- [ ] CI workflow passes on a throwaway fork before announcement.
- [ ] Docker image builds locally (`docker build -t agentcibench
      -f docker/Dockerfile .`).

## After the first release

- Cut a GitHub Release tagged `v1.0.0` with the arXiv abstract and the HF
  dataset link in the release notes.
- Enable GitHub Pages on `main` → `/AgentCIBench/website/` (or copy that
  folder to `docs/`) so the leaderboard is publicly browsable.
- Open an issue template for leaderboard submissions.
