# Containerised AgentCIBench

This directory provides an optional Docker setup for running the benchmark in
an isolated environment. The image bundles Python 3.11, `uv`, Playwright
Chromium, and all project dependencies pinned to `uv.lock`.

> Docker is **not required**. The native `uv sync` flow described in the
> repository README is fully supported. Use Docker when you need bit-for-bit
> reproducibility (e.g., for a leaderboard submission) or when running on a
> headless machine where installing browser system libraries is inconvenient.

## Files

- `Dockerfile` — single-stage image (~2.5 GB) with Playwright Chromium.
- `docker-compose.yml` — convenience wrapper that mounts `data/` and
  `config/` as volumes and reads API keys from a repo-root `.env`.

## Build

From the repository root (not from `docker/`):

```bash
docker build -t agentcibench -f docker/Dockerfile .
```

## Run

Interactive shell with mounted data:

```bash
docker run --rm -it \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v "$PWD/data:/app/data" \
  agentcibench bash
```

Reasoning-benchmark smoke test (headless):

```bash
docker run --rm \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v "$PWD/data:/app/data" \
  agentcibench \
  uv run python -m eval.run_benchmark \
    --generated-dir data/generated_merged \
    --results-dir data/results/text_smoke/docker \
    --proxy-model openrouter/openai/gpt-5.4-mini \
    --judge-model openrouter/google/gemma-4-31b-it
```

Visual benchmark (Playwright Chromium is preinstalled, so no extra setup):

```bash
docker run --rm \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v "$PWD/data:/app/data" \
  agentcibench \
  uv run python -m eval.run_visual_benchmark \
    --scenario data/eval_set_e2e_50/<scenario>.json \
    --results-dir data/results/visual_docker \
    --runtime-root data/runtime_openapps \
    --agent-model openrouter/anthropic/claude-sonnet-4.6 \
    --judge-model openrouter/google/gemma-4-31b-it \
    --access-mode mixed \
    --use-litellm --api-key-env OPENROUTER_API_KEY \
    --max-steps 35
```

## docker compose

A `.env` file at the repo root (gitignored) holds your API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

Then:

```bash
docker compose -f docker/docker-compose.yml run --rm agentci bash
```

Results persist in the named volume `agentci-results`; remove with
`docker volume rm agentcibench_agentci-results`.

## GPU / open-weight models

The default image is CPU-only. For local open-weight inference (vLLM, TGI,
etc.) we recommend running those services in a separate GPU container and
pointing AgentCIBench at them via an OpenAI-compatible endpoint:

```bash
docker run --rm \
  -e OPENAI_API_KEY="dummy" \
  -e OPENAI_BASE_URL="http://host.docker.internal:8000/v1" \
  -v "$PWD/data:/app/data" \
  agentcibench \
  uv run python -m eval.run_benchmark ...
```

## Troubleshooting

- **`playwright: command not found`** — rebuild with no cache:
  `docker build --no-cache -t agentcibench -f docker/Dockerfile .`
- **Permission errors on mounted volumes (Linux)** — the image runs as
  uid 1000; either chown the host directory or run with `--user $(id -u)`.
- **Headed browser for debugging** — the bundled Chromium runs headless by
  default. To debug visually, run with `--network host`, install an X server
  on the host, and forward `DISPLAY`.
