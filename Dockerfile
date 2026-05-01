# Builder
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
ENV SENTENCE_TRANSFORMERS_HOME=/models
RUN --mount=type=cache,target=/root/.cache/huggingface \
    uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Runtime
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv ./venv
COPY --from=builder /models /models
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME="/models"

COPY src ./src
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860', timeout=5)"

CMD ["python", "-m", "movie_recommender.app"]
