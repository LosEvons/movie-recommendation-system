# Movie Recommendation System

A small semantic-search movie recommender built with Sentence Transformers, ChromaDB, and Gradio.

## Quick start with Docker Hub

1. Start the stack:

```powershell
docker compose -f docker-compose.prod.yml up -d
```

2. Open the UI:

- http://localhost:7860

## Local development

```powershell
docker compose up --build -d
```

Then run ingestion:

```powershell
docker compose run --rm app python -m movie_recommender.ingest
```

## Docker image

Published image:

- `nausteri/movie-recommendation-system:latest`

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `CHROMA_HOST` | `chromadb` | ChromaDB hostname |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `CHROMA_RETRIES` | `10` | Connection attempts before giving up |
| `CHROMA_RETRY_DELAY` | `2` | Seconds between retries |
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence Transformer model — must match between ingest and app |
| `TMDB_LIMIT` | `5000` | Rows to ingest; use `100` for fast dev cycles |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Gradio bind address |
| `GRADIO_SERVER_PORT` | `7860` | Gradio port |
| `GRADIO_SHARE` | `false` | Set to `true` to get a public Gradio link |

> **Note:** Ingestion uses [kagglehub](https://github.com/Kaggle/kagglehub) which can download the TMDB dataset anonymously. If you hit authentication errors, set `KAGGLE_API_TOKEN` to a token from your Kaggle account settings.
>
> **Note:** `MODEL_NAME` is baked into the Docker image at build time as `all-MiniLM-L6-v2`. Overriding it at runtime requires rebuilding the image, otherwise embeddings will be incompatible.

## What to expect

- If the database is empty, the UI shows a friendly message telling you to run ingest first.
- Recommendations are returned as a simple list of movie titles.
