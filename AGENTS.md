# AGENTS.md

Guidance for Codex and future coding agents working in this repository.

## Project Goal

This repository is an AI Engineer portfolio project: an Internal Support Copilot built with FastAPI, Streamlit, RAG, Qdrant, Redis, LangGraph multi-agent orchestration, guarded write actions, Docker, evaluation scripts, and tests.

The long-term direction is an Enterprise Support Intelligence Copilot that can reason across synthetic enterprise support data, CRM-like customer records, support tickets, knowledge base documents, engineering and GitHub issue evidence, incident and risk data, and future Knowledge Graph / GraphRAG and anomaly/risk scoring components.

## Repository Layout

- `src/api/`: FastAPI application, health/readiness endpoints, and request handling.
- `src/ui/`: Streamlit chat UI.
- `src/agent/`: routing, memory, guarded actions, tools, and single-agent workflows.
- `src/agent/graph/`: LangGraph supervisor and multi-agent subgraphs.
- `src/rag/`: ingestion, chunking, indexing, retrieval, reranking, and generation.
- `src/integrations/`: GitHub App and local git integration clients.
- `src/persistence/`: Redis and in-memory persistence for sessions, actions, and graph checkpoints.
- `src/core/`: settings, auth, security, logging, observability, and runtime checks.
- `scripts/`: local development, ingestion, download, and evaluation helpers.
- `tests/`: unit and focused regression tests.
- `eval/`: benchmark queries, run outputs, and evaluation documentation.
- `docs/`: architecture, deployment, authorization, observability, routing, actions, persistence, security, and troubleshooting notes.
- `data_source/`: synthetic/raw/processed data used for local demos and evaluation.

## Setup Commands

Use Python `3.10` or `3.11`.

Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/dev.py install
Copy-Item .env.example .env
docker compose up -d qdrant redis
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python scripts/dev.py install
cp .env.example .env
docker compose up -d qdrant redis
```

Common commands:

```bash
python scripts/dev.py ingest-data
python scripts/dev.py run-api
python scripts/dev.py run-ui
python scripts/dev.py benchmark-retrieval
python scripts/dev.py benchmark-answers
```

Optional Makefile shortcuts may exist for systems with `make`, but `scripts/dev.py` is the canonical cross-platform entrypoint.

## Test And Lint Commands

Run the default test suite:

```bash
python scripts/dev.py run-tests
```

Run the CI-equivalent unit subset:

```bash
python -m pytest -m "not integration" -q
```

Run lint/format checks:

```bash
python -m ruff check .
python -m ruff format --check .
```

If full-repo lint fails on pre-existing backlog, do not silently reformat the whole repository. Prefer focused checks on changed Python files unless the user explicitly asks for a broader cleanup.

## Coding Conventions

- Keep changes small, explicit, and reviewable.
- Follow existing module boundaries and local patterns.
- Prefer typed, testable functions over broad service rewrites.
- Keep external side effects behind integration or action layers.
- Keep auth, guardrails, action confirmation, and idempotency behavior intact.
- Use structured parsers and existing helpers instead of ad hoc string handling when practical.
- Add or update tests for new behavior.
- Update README or docs when user-facing behavior, setup, commands, environment variables, or architecture notes change.
- Avoid committing generated caches, local artifacts, model caches, or environment-specific files.

## Data Safety Rules

- Never use private, proprietary, or real customer data.
- Use synthetic data only for customer profiles, tickets, incidents, CRM-like records, support transcripts, and evaluation cases.
- Do not commit secrets, tokens, API keys, private keys, `.env`, or local credential files.
- Prefer `.env.example` placeholders and documentation over real values.
- Keep GitHub App private keys outside the repository and reference them by path.
- Treat logs, health output, benchmark outputs, and generated data as potentially sensitive unless proven synthetic.

## Engineering Constraints

- Do not rewrite the architecture unless explicitly asked.
- Do not remove the existing GitHub ingestion pipeline.
- Do not remove existing features or tests to make a change easier.
- Keep FastAPI, Streamlit, RAG, Qdrant, Redis, LangGraph, guarded actions, Docker, evaluation scripts, and tests working unless the task explicitly changes them.
- Keep new dependencies minimal and justified.
- Preserve backward-compatible environment defaults where possible.
- Mark tests that require live external services with `@pytest.mark.integration`.
- Validate new data loaders with synthetic fixtures and focused tests.
- Keep future Knowledge Graph / GraphRAG, anomaly scoring, and risk scoring work additive and well isolated until the architecture is intentionally expanded.

## Definition Of Done

- The requested behavior is implemented without unrelated rewrites.
- Tests pass where possible, or failures are documented with exact reproduction commands and causes.
- New data loaders or ingestion paths are validated with synthetic data.
- README and docs are updated when behavior, setup, commands, or environment variables change.
- No `__pycache__`, `*.pyc`, `.pytest_cache`, `.ruff_cache`, model cache, local secret, or generated throwaway artifact is committed.
- Git status is reviewed so only intentional files are changed.
