# Local Deployment

## Stack Overview

```text
Browser
  |
  v
Streamlit UI (:8501)
  |
  v
FastAPI API (:8000)
  | \
  |  +--> Redis (:6379) for sessions, pending actions, and checkpoints
  |
  +----> Qdrant (:6333) for retrieval/index storage

data_source/raw
  |
  v
scripts/ingest_data.py
  |
  v
data_source/processed + Qdrant collection
```

## Supported Local Modes

### 1. Host-run app + Docker infra

Use this when you want the simplest day-to-day developer loop.

```bash
cp .env.example .env
docker compose up -d qdrant redis
python scripts/dev.py ingest-data
python scripts/dev.py run-api
python scripts/dev.py run-ui
```

### 2. Full Docker stack

Use this when you want the whole stack to come up the same way on every machine.

```bash
cp .env.example .env
docker compose up -d qdrant redis
docker compose run --rm --build api python scripts/ingest_data.py
docker compose up --build
```

After that:

- API: `http://127.0.0.1:8000`
- UI: `http://127.0.0.1:8501`
- Qdrant: `http://127.0.0.1:6333`
- Redis: `127.0.0.1:6379`

## Environment Expectations

Start from `.env.example`.

The most important values for local deployment are:

- `QDRANT_URL`
  - Host-run app: keep `http://localhost:6333`
  - Docker API service: compose overrides this to `http://qdrant:6333`
- `REDIS_URL`
  - Host-run app: keep `redis://localhost:6379/0`
  - Docker API service: compose overrides this to `redis://redis:6379/0`
- `INTERNAL_SUPPORT_API_BASE_URL`
  - Host-run UI: keep `http://127.0.0.1:8000`
  - Docker UI service: compose overrides this to `http://api:8000`
- `LOCAL_GIT_DEFAULT_REPO_PATH` and `LOCAL_GIT_ALLOWED_ROOTS`
  - Compose overrides these to `/app` inside the API container to avoid host-specific paths

## Predictable Startup Notes

- `docker compose up -d qdrant redis` is enough for the existing host-run bootstrap.
- The API container uses `/ready` as its healthcheck, so it stays unhealthy until ingestion has created the Qdrant collection.
- The UI container can start before the API is fully ready, but it will not answer successfully until the API is healthy.
- `./data_source` is bind-mounted into the API container so processed files and raw source content stay on your machine instead of inside the image.
- Hugging Face cache is stored in a named Docker volume to avoid redownloading models every container restart.
