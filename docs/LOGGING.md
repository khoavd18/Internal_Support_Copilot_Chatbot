# Logging

This project uses structured application logs across the API, retrieval flow, routing, tool execution, and write actions.

## Format

- Default format: JSON lines
- Optional format: plain text key/value pairs
- Environment variables:
  - `LOG_LEVEL`: standard Python log level such as `DEBUG`, `INFO`, `WARNING`, or `ERROR`
  - `LOG_FORMAT`: `json` or `text`

Example JSON log:

```json
{"timestamp":"2026-04-14T08:12:00Z","level":"INFO","logger":"src.api.main","event":"http.request.completed","request_id":"req-123","session_id":"chat-42","message":"HTTP request completed","status_code":200,"duration_ms":14.8}
```

## Correlation Fields

The logging layer attaches these fields when they are available in the current execution context:

- `request_id`: request correlation id, usually from `X-Request-ID`
- `session_id`: chat or workflow session id
- `user_id`: optional caller id from `X-User-ID`
- `action_id`: persisted write-action record id
- `agent_name`: logical agent or execution area such as `internal_support_agent` or `supervisor`

## Logged Events

Common event families include:

- `http.request.*`: request lifecycle at the API boundary
- `health.*`: health and readiness probes
- `chat.*`: plain, agent, and multi-agent request handling
- `pipeline.*`: retrieval and answer generation pipeline
- `tool.*`: knowledge-base tool execution
- `retrieval.*`: vector search and reranking flow
- `route.*`: routing decisions
- `action.*`: write-action preparation, confirmation, execution, replay, and failure
- `github.*` / `local_git.*`: external write-capable integrations

## Error Handling

- Unexpected server-side failures are logged with stack traces via `logger.exception(...)`.
- User-facing API responses return clean failure messages and do not include raw tracebacks.
- Readiness and startup validation diagnostics are exposed through `/health` and `/ready` without leaking secrets.

## Redaction Rules

The formatter redacts values for sensitive-looking keys such as:

- `token`
- `secret`
- `password`
- `authorization`
- `private_key`
- `api_key`
- `jwt`

It also sanitizes free-form log messages and exception text by:

- masking configured secret values such as `QDRANT_API_KEY` and `GITHUB_PRIVATE_KEY_PATH`
- masking URL credentials such as `redis://:password@host:6379/0`
- masking common token-style assignments such as `token=...`, `Authorization: Bearer ...`, and JSON fields like `"api_key": "..."`

Large strings and collections are truncated to keep volume reasonable.

## Operational Notes

- Prefer passing `X-Request-ID` from ingress, reverse proxies, or API clients for cross-service traceability.
- Use `LOG_LEVEL=DEBUG` only for targeted debugging sessions because retrieval and action flows produce additional diagnostic events.
- Structured logs are intended for stdout/stderr collection by local shells, containers, or a log shipper.
- Redaction reduces accidental exposure, but logs and health payloads should still be treated as internal operational data rather than something to paste publicly.
