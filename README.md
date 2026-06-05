# Internal Support Copilot

Enterprise-oriented internal support copilot that combines local RAG, agent routing, and a LangGraph-based multi-agent orchestration layer for GitHub Docs, GitLab Handbook, and GitHub Issues.

## Why This Project Is Worth Exploring

- Multi-layer runtime design: plain RAG, single-agent orchestration, and LangGraph supervisor mode.
- Practical internal-support use case instead of a toy chatbot wrapper.
- Retrieval transparency with sources, debug metadata, and guardrail decisions.
- Controlled write actions for GitHub repository creation and local git commits.
- Test coverage for retrieval helpers, multi-agent synthesis, action wrappers, and API wiring.

## What This Project Does

- Retrieves and synthesizes answers from internal knowledge sources.
- Supports three backend modes: plain RAG, single-agent, and multi-agent supervisor flow.
- Exposes a FastAPI service for application integration.
- Provides a Streamlit UI for local exploration and debugging.
- Includes guarded write actions for GitHub issue creation, GitHub repository creation, and local git commits.

## Architecture At A Glance

- `src/api`: FastAPI entrypoints and request/response orchestration.
- `src/agent`: single-agent logic, routing, memory, and action wrappers.
- `src/agent/graph`: supervisor graph and subgraphs for multi-agent retrieval.
- `src/rag`: ingestion, chunking, indexing, retrieval, and answer generation.
- `src/integrations`: GitHub App and local git integrations.
- `src/ui`: Streamlit chat interface.
- `scripts`: data preparation, retrieval evaluation, and environment checks.
- `tests`: focused unit tests for retrieval, agent behavior, actions, and API wiring.

Detailed notes live in [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md).
Routing notes live in [docs/ROUTING.md](./docs/ROUTING.md).
Action notes live in [docs/ACTIONS.md](./docs/ACTIONS.md).
Deployment notes live in [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md).
Operator runbook lives in [docs/RUNBOOK.md](./docs/RUNBOOK.md).
Troubleshooting notes live in [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md).
Persistence notes live in [docs/PERSISTENCE.md](./docs/PERSISTENCE.md).
Logging notes live in [docs/LOGGING.md](./docs/LOGGING.md).
Observability notes live in [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md).
Authorization notes live in [docs/AUTHORIZATION.md](./docs/AUTHORIZATION.md).
Secret-handling notes live in [docs/SECURITY.md](./docs/SECURITY.md).

## Repository Layout

```text
.
|-- src/
|   |-- api/
|   |-- agent/
|   |-- integrations/
|   |-- rag/
|   |-- ui/
|-- scripts/
|-- tests/
|-- data_source/
|-- eval/
|-- .github/workflows/
|-- .env.example
|-- Dockerfile
|-- docker-compose.yml
|-- Makefile
|-- pyproject.toml
|-- pytest.ini
|-- requirements.txt
```

## Dependency Files

- `pyproject.toml`: package metadata plus mirrored runtime/dev dependency declarations.
- `requirements.txt`: runtime install list.
- `requirements-dev.txt`: runtime dependencies plus test/lint tooling.
- `scripts/dev.py`: canonical cross-platform wrapper for install, test, run, ingest, and evaluation commands.
- `Makefile`: optional shortcuts for systems with `make`; it delegates to `scripts/dev.py` or standard Python module commands.
- No lock file is currently committed; the files above are kept in sync for local bootstrap.

## Bootstrap

Use Python `3.10` or `3.11`. The examples below use one supported version explicitly; replace it with the supported minor version installed on your machine.

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/dev.py install
Copy-Item .env.example .env
docker compose up -d qdrant redis
```

Add your raw source material under `data_source/raw`, then ingest and run:

```powershell
python scripts/dev.py ingest-data
python scripts/dev.py run-api
python scripts/dev.py run-ui
```

Run tests any time with:

```powershell
python scripts/dev.py run-tests
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python scripts/dev.py install
cp .env.example .env
docker compose up -d qdrant redis
```

Add your raw source material under `data_source/raw`, then ingest and run:

```bash
python scripts/dev.py ingest-data
python scripts/dev.py run-api
python scripts/dev.py run-ui
```

Run tests any time with:

```bash
python scripts/dev.py run-tests
```

### Helper Commands

- `python scripts/dev.py install`
- `python scripts/dev.py benchmark-answers`
- `python scripts/dev.py benchmark-retrieval`
- `python scripts/dev.py ingest-data`
- `python scripts/dev.py run-api`
- `python scripts/dev.py run-ui`
- `python scripts/dev.py run-tests`

You can pass extra flags to the wrapped command after `--`.
Example: `python scripts/dev.py run-api -- --host 0.0.0.0 --port 8000`

Optional Makefile shortcuts are available on systems with `make`:

```bash
make install
make docker-infra
make ingest
make run-api
make run-ui
make test-unit
make eval-retrieval
make eval-answers
```

### Enterprise Support Sample Data

The synthetic enterprise support dataset lives under `data/sample_enterprise_support/`.
It is separate from the existing GitHub Docs/GitLab/GitHub Issues ingestion pipeline.

Validate and preview the generated RAG documents without writing to Qdrant:

```bash
python scripts/ingest_enterprise_support_data.py --dry-run
```

Ingest the enterprise support documents into a separate Qdrant collection:

```bash
python scripts/ingest_enterprise_support_data.py --collection-name enterprise_support_copilot_qdrant
```

To query that collection through the API, start the API with `QDRANT_COLLECTION_NAME=enterprise_support_copilot_qdrant`. To intentionally mix enterprise support documents into the default collection, pass the same collection name used by your normal `QDRANT_COLLECTION_NAME`.

### Enterprise GraphRAG Fusion

The enterprise support path now includes a lightweight in-memory Knowledge Graph and a simple GraphRAG fusion layer. The fusion layer tries vector retrieval first, retrieves related graph context from the synthetic enterprise support KG, deduplicates evidence by stable entity IDs, and marks each context item as `vector`, `graph`, or `both`.

The first `/enterprise/ask` version does not call an LLM. It returns a placeholder answer plus the fused evidence so retrieval behavior can be inspected safely:

```bash
curl -X POST http://127.0.0.1:8000/enterprise/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why is the API timeout risky for Northstar?",
    "top_k": 5,
    "graph_depth": 2
  }'
```

### Enterprise Risk Scoring

The synthetic enterprise support path also includes a lightweight customer risk scorer. Because `scikit-learn` is not currently part of the project dependencies, this first version uses a deterministic heuristic anomaly baseline over recent tickets and risk events instead of `IsolationForest`.

```bash
curl -X POST http://127.0.0.1:8000/risk/customer-score \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "cust_009"}'
```

## Local Deployment

Two practical local modes are supported:

- Host-run app + Docker infra: `docker compose up -d qdrant redis`, then run `python scripts/dev.py ingest-data`, `python scripts/dev.py run-api`, and `python scripts/dev.py run-ui`
- Full Docker stack: `docker compose up -d qdrant redis`, then `docker compose run --rm --build api python scripts/ingest_data.py`, then `docker compose up --build`

Container defaults:

- API: `http://127.0.0.1:8000`
- UI: `http://127.0.0.1:8501`
- Qdrant: `http://127.0.0.1:6333`
- Redis: `127.0.0.1:6379`

The API and UI images are built from the repo `Dockerfile`, and `docker-compose.yml` overrides container-only addresses such as `QDRANT_URL=http://qdrant:6333`, `REDIS_URL=redis://redis:6379/0`, and `INTERNAL_SUPPORT_API_BASE_URL=http://api:8000`.

A concise stack diagram and the full startup notes live in [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md).

## Startup Assumptions

- Python `>=3.10,<3.12` is supported. The CI workflow exercises Python `3.10` and `3.11`; the Docker image uses Python `3.11`.
- The default `.env` assumes Qdrant is available at `http://localhost:6333`, which is what `docker compose up -d qdrant redis` starts.
- The containerized API service overrides `QDRANT_URL` and `REDIS_URL` to use Docker service names, so you do not need a separate container-only env file.
- The default `.env.example` also assumes Redis is available at `redis://localhost:6379/0` for persistent chat/session and checkpoint storage.
- The containerized UI service overrides `INTERNAL_SUPPORT_API_BASE_URL` to `http://api:8000`.
- Fresh clones include a processed snapshot under `data_source/processed` for demo/evaluation use. Raw source material is expected under `data_source/raw` and is kept out of version control; to rebuild the processed snapshot, add raw sources and run `python scripts/dev.py ingest-data`.
- `ingest-data` prepares JSONL files and rebuilds the Qdrant collection, so Qdrant must be running before you call it.
- The API validates environment configuration at startup and will fail fast if enabled features are missing required settings.
- The first real API/UI run may download Hugging Face models. `.env.example` defaults to `LLM_QUANTIZATION=none` for CPU-only machines; set `LLM_QUANTIZATION=4bit` or `8bit` only when CUDA and bitsandbytes are available.
- GitHub and local git write actions are disabled by default and require extra environment configuration before use.
- Authorization is enabled by default. Anonymous requests are treated as `viewer`, while write-capable requests require `X-User-ID` and `X-User-Role: operator`.

## Persistence

- Chat history, pending confirmation actions, write-action records, and supervisor graph checkpoints now go through pluggable persistence abstractions.
- Redis is the first persistent backend and is included in `docker-compose.yml`.
- If Redis is unavailable and you want the previous local-only behavior, set `SESSION_STORE_BACKEND=memory`, `ACTION_STORE_BACKEND=memory`, and `GRAPH_CHECKPOINTER_BACKEND=memory`.
- Setup and migration notes live in [docs/PERSISTENCE.md](./docs/PERSISTENCE.md).

## Example API Usage

### Customer summary

```bash
curl -X POST http://127.0.0.1:8000/crm/customer-summary \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "cust_001"}'
```

### Ticket triage

```bash
curl -X POST http://127.0.0.1:8000/support/ticket-triage \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "tkt_001"}'
```

### Suggested support reply

```bash
curl -X POST http://127.0.0.1:8000/support/suggest-reply \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "tkt_001"}'
```

### SLA check

```bash
curl -X POST http://127.0.0.1:8000/support/sla-check \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "tkt_026"}'
```

### Customer risk score

```bash
curl -X POST http://127.0.0.1:8000/risk/customer-score \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "cust_009"}'
```

### Ask the multi-agent backend

```bash
curl -X POST http://127.0.0.1:8000/multi-agent/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I sign in with a passkey?",
    "mode": "auto",
    "debug": true
  }'
```

### Trigger a guarded action

```bash
curl -X POST http://127.0.0.1:8000/multi-agent/actions/create-issue \
  -H "Content-Type: application/json" \
  -H "X-User-ID: operator-1" \
  -H "X-User-Role: operator" \
  -d '{
    "repo_full_name": "your-org/demo-repo",
    "title": "Bug login",
    "body": "Steps to reproduce",
    "confirmed": true,
    "idempotency_key": "issue-bug-login-001"
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/multi-agent/actions/create-repo \
  -H "Content-Type: application/json" \
  -H "X-User-ID: operator-1" \
  -H "X-User-Role: operator" \
  -d '{
    "org": "your-org",
    "name": "demo-repo",
    "private": true,
    "confirmed": true,
    "idempotency_key": "repo-demo-repo-001"
  }'
```

### Use Chat To Propose And Confirm Actions

`/agent/ask` and `/multi-agent/ask` can now detect write actions from natural-language requests.

- Example request: `create issue repo:your-org/demo-repo title:"Bug login" body:"Steps to reproduce"`
- The same request needs operator headers such as `X-User-ID: operator-1` and `X-User-Role: operator`.
- First response: returns an action proposal when confirmation is required.
- Confirm in the same chat session by sending `yes` / `xac nhan`, or resend the original request with `confirmed=true`.
- Repeating the same write request in the same session now reuses the persisted action record instead of executing the side effect twice.

## Core Endpoints

- `GET /health`
- `GET /ready`
- `POST /crm/customer-summary`
- `POST /support/ticket-triage`
- `POST /support/suggest-reply`
- `POST /support/sla-check`
- `POST /risk/customer-score`
- `POST /enterprise/ask`
- `POST /ask`
- `POST /agent/ask`
- `POST /multi-agent/ask`
- `POST /multi-agent/actions/create-issue`
- `POST /multi-agent/actions/create-repo`
- `POST /multi-agent/actions/commit`

## Health Checks

Use `/health` for a broad status snapshot that includes startup validation and dependency diagnostics:

```bash
curl http://127.0.0.1:8000/health
```

Use `/ready` for readiness probes. It returns `200` only when required dependencies are ready, and `503` with diagnostics when they are not:

```bash
curl -i http://127.0.0.1:8000/ready
```

Typical readiness failures include:

- Qdrant is not reachable at the configured `QDRANT_URL`.
- The configured Qdrant collection does not exist yet because ingestion has not been run.
- Redis-backed persistence is enabled but Redis is not reachable.
- Optional GitHub or local git actions are enabled but their configuration is incomplete.

## Structured Logging

- Logs are structured and default to JSON lines with `LOG_FORMAT=json`.
- Switch to a compact human-readable format with `LOG_FORMAT=text`.
- Control verbosity with `LOG_LEVEL`, which defaults to `INFO` in `.env.example`.
- Correlation fields are added when available: `request_id`, `session_id`, `user_id`, `action_id`, and `agent_name`.
- The API accepts `X-Request-ID` and optional `X-User-ID` headers so upstream services can propagate correlation identifiers.
- Server logs include stack traces for unexpected failures, while API responses keep error messages clean and non-sensitive.
- Log messages, exception text, health diagnostics, and common URL credential formats are redacted before they are emitted.

Example:

```json
{"timestamp":"2026-04-14T08:12:00Z","level":"INFO","logger":"src.api.main","event":"http.request.completed","request_id":"req-123","session_id":"chat-42","message":"HTTP request completed","status_code":200,"duration_ms":14.8}
```

Field definitions and redaction rules live in [docs/LOGGING.md](./docs/LOGGING.md).

## Observability

- Metrics and tracing hooks live in [src/core/observability.py](./src/core/observability.py).
- The default backend is lightweight in-memory instrumentation with no external exporter dependency.
- Measured latencies currently include request handling, retrieval, reranking, LLM calls, and write-action execution.
- Write actions also emit success/failure counters by action type.
- The design is backend-agnostic so Prometheus or OpenTelemetry adapters can be added later without touching the application call sites.

Key environment variables:

- `OBSERVABILITY_BACKEND=memory` or `noop`
- `OBSERVABILITY_TRACE_HISTORY_LIMIT=200`

Metric names and instrumentation points are documented in [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md).

## Authorization

- Read-only endpoints are available to `viewer` requests, including anonymous requests when `AUTH_ALLOW_ANONYMOUS_READS=true`.
- Direct write endpoints and chat-driven write intents require `operator`.
- The default lightweight provider reads `X-User-ID` and `X-User-Role`, but the permission boundary is isolated in [src/core/auth.py](./src/core/auth.py) so an SSO-backed provider can replace it later.
- The current audit of read-only, mixed-mode, and write-capable routes lives in [docs/AUTHORIZATION.md](./docs/AUTHORIZATION.md).

## Testing

Run the focused test suite:

```bash
python scripts/dev.py run-tests
```

Run the CI-equivalent unit subset:

```bash
python -m pytest -m "not integration" -q
```

Run the retrieval benchmark:

```bash
python scripts/dev.py benchmark-retrieval
```

Use `eval/queries/retrieval_benchmark.jsonl` for labeled cases, or pass a different benchmark file after `--`.
The benchmark writes a per-query run file under `eval/runs/`, a machine-readable JSON summary under `eval/runs/`, and the latest human-readable report to `eval/reports/latest_retrieval_summary.txt`.
Format details live in [eval/README.md](./eval/README.md).

Run the answer-quality benchmark:

```bash
python scripts/dev.py benchmark-answers
```

It reuses the same benchmark case file, generates answers through the local pipeline, and scores practical dimensions such as correctness, groundedness, citation relevance, and completeness.
Outputs are written to timestamped files under `eval/runs/` and `eval/reports/`.

Run lint locally:

```bash
ruff check .
ruff format --check .
```

Run a narrower regression set used during recent cleanup:

```bash
python scripts/dev.py run-tests -- tests/test_agent_tools.py tests/test_multi_agent_synthesize.py tests/test_github_client.py tests/test_agent_actions.py tests/test_api_actions.py tests/test_chat_actions.py tests/test_api_chat_actions.py -q
```

## CI

- GitHub Actions runs three fast PR checks: `Lint`, `Import And Format`, and `Unit Tests`.
- `Lint` and `Import And Format` use `ruff`, but only on changed Python files in the PR. This keeps normal review cycles fast while the repo still has some older style backlog outside the active diff.
- `Unit Tests` runs `python -m pytest -m "not integration" -q` on Python `3.10` and `3.11`.
- The default PR pipeline does not start live Qdrant, Redis, or GitHub integrations. Current checked-in tests stay isolated with fakes, monkeypatching, or `fakeredis`.
- If you add tests that need live external services, mark them with `@pytest.mark.integration` so they stay clearly separated from the default unit-test workflow.

Local equivalents:

```bash
python -m ruff check --select E,F,UP,B --ignore E501,B008 <changed-python-files>
python -m ruff check --select I <changed-python-files>
python -m ruff format --check <changed-python-files>
python -m pytest -m "not integration" -q
```

## Operational Notes

- Keep secrets in `.env`; never commit real credentials.
- Prefer storing the GitHub App PEM outside the repo and referencing it with `GITHUB_PRIVATE_KEY_PATH`.
- If Redis or Qdrant use credentials, keep them only in local environment variables.
- Existing `data_source/processed` files are tracked as a processed snapshot. Regenerated or additional processed artifacts should only be committed deliberately.
- GitHub write actions are protected by explicit feature flags and confirmation fields.
- Local git commit actions are restricted by allowed root configuration.
- Repeated write requests are guarded by persisted idempotency records with statuses such as `pending`, `running`, `succeeded`, and `failed`.
- Logs and `/health` or `/ready` outputs now redact known secrets, but they should still be treated as internal operational data.

## Repo Maturity Signals

- MIT licensed for public personal-project sharing.
- GitHub Actions CI added under `.github/workflows/ci.yml`.
- Contributor guidance lives in `CONTRIBUTING.md`.
- Tooling standards are captured in `pyproject.toml` and `.editorconfig`.

## Engineering Standards

- Prefer small, testable modules over broad service classes.
- Keep agent orchestration and external side effects separated.
- Document new entrypoints and env flags when extending the platform.
- Add tests for routing, tools, and integration wrappers before shipping changes.

## Fresh Clone Checklist

- Create and activate a Python `3.10` or `3.11` virtual environment.
- Run `python scripts/dev.py install`.
- Copy `.env.example` to `.env`.
- Start Qdrant and Redis with `docker compose up -d qdrant redis`.
- Add source content under `data_source/raw`.
- Run `python scripts/dev.py ingest-data`.
- Start the API with `python scripts/dev.py run-api`.
- Start the UI with `python scripts/dev.py run-ui`.
- Verify the repo with `python scripts/dev.py run-tests`.
