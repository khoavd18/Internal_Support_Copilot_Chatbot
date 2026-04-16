# Troubleshooting

## Startup Failures

### API exits immediately with startup validation errors

Symptom:

- API process stops on startup
- logs contain `Startup configuration validation failed`

What to run:

```bash
python scripts/dev.py run-api
docker compose logs --tail=100 api
```

Common causes:

- Redis-backed persistence is enabled but `REDIS_URL` is missing
- `QDRANT_MODE=server` but `QDRANT_URL` is empty
- `QDRANT_COLLECTION_NAME`, `QDRANT_VECTOR_NAME`, or `QDRANT_SPARSE_VECTOR_NAME` is empty
- `AUTH_DEFAULT_ROLE` is invalid
- GitHub or local git actions are enabled without their required settings

Fix:

- start from `.env.example`
- only enable optional features after setting their required values
- rerun the API and then check `/health`

### `/ready` reports persistence not ready

What to run:

```bash
docker compose ps redis
docker compose logs --tail=100 redis
curl -i http://127.0.0.1:8000/ready
```

Common causes:

- `SESSION_STORE_BACKEND=redis`, `ACTION_STORE_BACKEND=redis`, or `GRAPH_CHECKPOINTER_BACKEND=redis` but Redis is down
- `REDIS_URL` points to the wrong host or port

Fix:

```bash
docker compose up -d redis
```

## Missing Env Vars Or Misconfigured Optional Features

### GitHub write actions fail before execution

Common required settings when `GITHUB_ACTIONS_ENABLED=true`:

- `GITHUB_APP_ID` or `GITHUB_CLIENT_ID`
- `GITHUB_PRIVATE_KEY_PATH`
- `GITHUB_ALLOWED_REPOS` and/or `GITHUB_ALLOWED_ORGS`

Useful check:

```bash
curl http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

### Local git actions fail before execution

Common required settings when `LOCAL_GIT_ACTIONS_ENABLED=true`:

- `LOCAL_GIT_DEFAULT_REPO_PATH`
- `LOCAL_GIT_ALLOWED_ROOTS`

The default repo path must exist and stay inside the allowed roots.

## Qdrant Issues

### `/ready` returns `503`

What to run:

```bash
docker compose ps qdrant
docker compose logs --tail=100 qdrant
curl -i http://127.0.0.1:8000/ready
curl http://127.0.0.1:6333/collections
```

Common causes:

- Qdrant is not running
- `QDRANT_URL` points to the wrong host
- ingestion has not been run yet
- the configured collection does not exist

Fix:

```bash
docker compose up -d qdrant
python scripts/dev.py ingest-data
```

### Qdrant is up but the collection is missing

Symptom:

- readiness reports `Collection '<name>' was not found in Qdrant`

Fix:

```bash
python scripts/dev.py ingest-data
```

Then recheck:

```bash
curl -i http://127.0.0.1:8000/ready
```

## Ingestion Failures

### `ingest-data` stops with missing raw data

Exact failures:

- `Missing data_source/raw`
- `No source files found under data_source/raw`

Fix:

- create `data_source/raw`
- add source files before rerunning ingestion

Retry:

```bash
python scripts/dev.py ingest-data
```

### Ingestion runs but retrieval still has no data

What to check:

```bash
type data_source\\processed\\prepare_stats.json
```

or on bash:

```bash
cat data_source/processed/prepare_stats.json
```

If `total_documents` is `0`, the raw source layout is wrong or the source files were skipped during preparation.

### Ingestion fails while rebuilding Qdrant

Likely cause:

- Qdrant is not reachable at `QDRANT_URL`

Fix:

```bash
docker compose up -d qdrant
python scripts/dev.py ingest-data
```

## Action Execution Failures

### `403` or permission denied on write actions

Write-capable requests require:

- `X-User-ID`
- `X-User-Role: operator`

Example:

```bash
curl -X POST http://127.0.0.1:8000/multi-agent/actions/commit \
  -H "Content-Type: application/json" \
  -H "X-User-ID: operator-1" \
  -H "X-User-Role: operator" \
  -d '{"message":"test","confirmed":true}'
```

### GitHub action fails with allowlist or auth errors

Common causes:

- `GITHUB_ACTIONS_ENABLED=false`
- repo is not in `GITHUB_ALLOWED_REPOS`
- org is not in `GITHUB_ALLOWED_ORGS`
- GitHub App credentials are incomplete

Checks:

```bash
curl http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

### Local git commit action fails

Common causes:

- `LOCAL_GIT_ACTIONS_ENABLED=false`
- repo path is outside `LOCAL_GIT_ALLOWED_ROOTS`
- commit message is empty
- no staged changes exist
- requested path is outside the repo root

Quick checks:

```bash
git status --short
git rev-parse --show-toplevel
```

### Repeated write request does not execute again

This is usually expected.

The action layer keeps durable action records with statuses such as:

- `pending`
- `confirmed`
- `running`
- `succeeded`
- `failed`
- `cancelled`

If a request reuses the same `idempotency_key`, the system may replay the prior result or refuse to rerun a failed/in-progress action.

## Common Test Failures

### `ModuleNotFoundError` or missing tools such as `ruff`

Install the dev environment:

```bash
python scripts/dev.py install
```

### You only want the normal fast test suite

Run:

```bash
python scripts/dev.py run-tests -- -m "not integration" -q
```

### A test depends on changed local config

Start from `.env.example` and avoid enabling optional write features unless the test specifically needs them.

### You see SWIG or dependency deprecation warnings

Those warnings are noisy but are not the same as test failures unless pytest is configured to treat warnings as errors.
