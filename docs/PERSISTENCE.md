# Persistence Setup

This project now supports pluggable persistence for chat history, pending confirmation
state, write-action records/idempotency state, and LangGraph supervisor checkpoints.

## What Changed

- Chat history and pending action state are routed through a `SessionStateStore` abstraction.
- Write-action records and idempotency tracking are routed through an `ActionStore` abstraction.
- The supervisor graph checkpointer is routed through a checkpointer factory.
- Redis is the first persistent backend for both of those paths.

## Recommended Local Setup

Start the bundled infrastructure:

```bash
docker compose up -d
```

That starts:

- Qdrant on `localhost:6333`
- Redis on `localhost:6379`

## Environment Variables

These settings live in `.env`:

```dotenv
SESSION_STORE_BACKEND=redis
ACTION_STORE_BACKEND=redis
GRAPH_CHECKPOINTER_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
REDIS_KEY_PREFIX=internal_support_copilot
SESSION_HISTORY_MAX_MESSAGES=6
SESSION_STATE_TTL_SECONDS=0
ACTION_STATE_TTL_SECONDS=0
GRAPH_CHECKPOINT_TTL_SECONDS=0
```

Notes:

- `SESSION_STATE_TTL_SECONDS=0` means session history and pending confirmations do not expire automatically.
- `ACTION_STATE_TTL_SECONDS=0` means action records and idempotency state do not expire automatically.
- `GRAPH_CHECKPOINT_TTL_SECONDS=0` means graph checkpoints do not expire automatically.
- If you need the old behavior for local debugging, set both backends to `memory`.

## Migration Notes

There is no automatic migration path from the old in-memory session store.

Why:

- The previous implementation kept state only inside the running Python process.
- Once that process exited, there was no durable source of truth to migrate from.

Practical rollout:

1. Start Redis.
2. Set `SESSION_STORE_BACKEND=redis`.
3. Set `ACTION_STORE_BACKEND=redis`.
4. Set `GRAPH_CHECKPOINTER_BACKEND=redis`.
5. Restart the API service.
6. New sessions, action records, and supervisor checkpoints will persist across restarts.

## Redis Key Layout

The current implementation uses a simple namespaced layout:

- `...:session:<session_id>:history`
- `...:session:<session_id>:pending_action`
- `...:action:<action_id>`
- `...:action-idempotency:<idempotency_key>`
- `...:graph:thread:<thread_id>:...`

All keys are prefixed with `REDIS_KEY_PREFIX`.

## When PostgreSQL Is The Better Choice

Redis is a good fit for short-lived conversational state and confirmation continuity.

PostgreSQL is the better next step for graph checkpoint history if you need:

- stronger durability guarantees
- long-term retention
- easier operational backups
- audit/reporting queries over historical checkpoint metadata

In other words: Redis works well for low-friction runtime continuity, while PostgreSQL is usually the better home for durable workflow history.
