# Observability

This project includes a small vendor-neutral observability layer in [src/core/observability.py](../src/core/observability.py).

## Goals

- Keep instrumentation points consistent and lightweight
- Avoid hard-coding Prometheus or OpenTelemetry details into application code
- Make future exporter integration possible by swapping the backend implementation

## Current Backend

- Default backend: `memory`
- Optional backend: `noop`
- Configuration:
  - `OBSERVABILITY_BACKEND=memory|noop`
  - `OBSERVABILITY_TRACE_HISTORY_LIMIT=200`

The in-memory backend records:

- Counter totals
- Histogram summaries
- A bounded list of completed spans for debugging and tests

It does not expose a `/metrics` endpoint yet and does not export traces to an external collector.

## Metric Names

- `http.server.request.duration_ms`
  - Measured in [src/api/main.py](../src/api/main.py)
  - Attributes: `method`, `path`, `status_code`, `status`

- `rag.retrieval.duration_ms`
  - Measured in [src/rag/retrieval/retriever.py](../src/rag/retrieval/retriever.py)
  - Attributes: `top_k`, `rebuild`, `has_filter`, `status`

- `rag.rerank.duration_ms`
  - Measured in [src/rag/retrieval/retriever.py](../src/rag/retrieval/retriever.py)
  - Attributes: `stage`, `strategy`, `status`
  - Current strategies: `heuristic`, `cross_encoder`

- `llm.call.duration_ms`
  - Measured in [src/pipeline.py](../src/pipeline.py)
  - Attributes: `backend`, `entrypoint`, `status`

- `action.execution.duration_ms`
  - Measured in [src/agent/actions.py](../src/agent/actions.py)
  - Attributes: `action_name`, `status`

- `action.execution.total`
  - Measured in [src/agent/actions.py](../src/agent/actions.py)
  - Attributes: `action_name`, `status`
  - Current statuses: `succeeded`, `failed`

## Tracing Hooks

Each timed operation also creates a span with a stable operation name, for example:

- `http.request`
- `rag.retrieve`
- `rag.rerank`
- `llm.call`
- `action.execute`

The span abstraction captures:

- start/end time
- duration
- success/failure status
- parent/child relationships
- correlation context when available, such as `request_id`, `session_id`, `user_id`, `action_id`, and `agent_name`

## Design Notes

- Metric attributes intentionally avoid high-cardinality payloads such as raw questions, request bodies, tokens, or document content.
- Span attributes may include small numeric or categorical diagnostics such as `top_k`, `docs_in`, `docs_out`, or `status_code`.
- To add Prometheus support later, implement an `ObservabilityBackend` that maps counters and histograms to Prometheus primitives.
- To add OpenTelemetry later, implement an `ObservabilityBackend` that maps spans and metric calls to OTel SDK APIs.
