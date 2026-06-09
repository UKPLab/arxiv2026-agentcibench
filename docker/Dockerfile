# syntax=docker/dockerfile:1.6
#
# AgentCIBench reproducible runtime.
#
# Bundles Python 3.11, uv, Playwright Chromium, and the project source.
# Model API keys are passed in at run-time via environment variables; no keys
# are baked into the image.
#
# Build (from repo root):
#   docker build -t agentcibench -f docker/Dockerfile .
#
# Run an interactive shell with the local data volume mounted:
#   docker run --rm -it \
#     -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
#     -v "$PWD/data:/app/data" \
#     agentcibench bash
#
# Run the reasoning benchmark headlessly:
#   docker run --rm \
#     -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
#     -v "$PWD/data:/app/data" \
#     agentcibench \
#     uv run python -m eval.run_benchmark \
#       --generated-dir data/generated_merged \
#       --results-dir data/results/text_smoke/docker \
#       --proxy-model openrouter/openai/gpt-5.4-mini \
#       --judge-model openrouter/google/gemma-4-31b-it

FROM python:3.11-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libnss3 \
        libnspr4 \
        libdbus-1-3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.5.11

WORKDIR /app

# Dependency layer — copied first so dependency changes alone don't bust the
# source-code layer cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Project source.
COPY . .
RUN uv sync --frozen

# Playwright Chromium for the visual benchmark.
RUN uv run playwright install --with-deps chromium

# A non-root user keeps mounted volumes sane on Linux hosts.
RUN useradd --create-home --uid 1000 agentci \
    && chown -R agentci:agentci /app /opt/playwright
USER agentci

CMD ["bash"]
