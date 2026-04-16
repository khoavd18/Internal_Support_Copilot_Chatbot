# Operator Runbook

## Normal Startup

Use this path for the usual local operator workflow.

1. Create `.env` from `.env.example`.
2. Start infrastructure:

```bash
docker compose up -d qdrant redis
```

3. Ingest source data:

```bash
python scripts/dev.py ingest-data
```

4. Start the API:

```bash
python scripts/dev.py run-api
```

5. Start the UI in a second terminal:

```bash
python scripts/dev.py run-ui
```

## Readiness Checks

Check API health:

```bash
curl http://127.0.0.1:8000/health
```

Check readiness:

```bash
curl -i http://127.0.0.1:8000/ready
```

Check infrastructure containers:

```bash
docker compose ps
docker compose logs --tail=100 qdrant redis
```

Useful readiness signals:

- `ready: true` means required dependencies are available.
- `checks.qdrant.collection_exists: true` means ingestion has built the collection.
- `checks.persistence.ok: true` means the configured session/action/checkpoint backend is reachable.

## Data Ingestion

Before ingestion, put source content under `data_source/raw`.

Run ingestion:

```bash
python scripts/dev.py ingest-data
```

Useful outputs:

- `data_source/processed/documents.jsonl`
- `data_source/processed/tickets.jsonl`
- `data_source/processed/prepare_stats.json`

If ingestion succeeds but retrieval is still not ready, rerun:

```bash
curl -i http://127.0.0.1:8000/ready
```

## Write Actions

Write actions are disabled by default. They require:

- operator headers: `X-User-ID` and `X-User-Role: operator`
- feature flags and credentials in `.env`
- confirmation unless disabled by config

Example direct issue creation:

```bash
curl -X POST http://127.0.0.1:8000/multi-agent/actions/create-issue \
  -H "Content-Type: application/json" \
  -H "X-User-ID: operator-1" \
  -H "X-User-Role: operator" \
  -d '{
    "repo_full_name": "owner/repo",
    "title": "Bug title",
    "body": "Steps to reproduce",
    "confirmed": true,
    "idempotency_key": "issue-owner-repo-bug-001"
  }'
```

If a repeated request returns an existing action result, that is expected. Write actions are idempotent by key and status.

## Logs

Host-run API logs are written to the terminal running `python scripts/dev.py run-api`.

Container logs:

```bash
docker compose logs -f api
docker compose logs -f ui
docker compose logs -f qdrant
docker compose logs -f redis
```

## Safe Restart

Restart only infrastructure:

```bash
docker compose restart qdrant redis
```

Restart the full containerized stack:

```bash
docker compose down
docker compose up --build
```

For host-run mode, stop the API and UI with `Ctrl+C`, then start them again with the same `scripts/dev.py` commands.
