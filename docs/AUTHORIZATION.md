# Authorization

This project now has a minimal role-based authorization layer in [src/core/auth.py](../src/core/auth.py).

## Design

- Current provider: header-based identity resolution
- Current roles:
  - `viewer`: read-only access
  - `operator`: read access plus dangerous write actions
- Default behavior:
  - anonymous requests are treated as `viewer`
  - write-capable requests require `X-User-ID` and `X-User-Role: operator`

The API and action routes depend on `AuthContext` and policy lookup helpers instead of raw headers directly. That keeps the permission boundary clean and makes it easier to swap the identity provider later for SSO, JWT claims, or a trusted reverse-proxy identity layer.

## Headers

- `X-User-ID`
- `X-User-Role`

Config is controlled by:

- `AUTH_ENABLED`
- `AUTH_USER_HEADER`
- `AUTH_ROLE_HEADER`
- `AUTH_DEFAULT_ROLE`
- `AUTH_ALLOW_ANONYMOUS_READS`
- `AUTH_REQUIRE_USER_ID_FOR_OPERATOR`

## Endpoint Audit

### Read-only

- `GET /health`
- `GET /ready`
- `POST /ask`

### Mixed-mode

- `POST /agent/ask`
- `POST /multi-agent/ask`

These endpoints are read-only for normal retrieval questions, but they become operator-only when the chat request is recognized as a write action or as a confirmation/cancellation step for a pending write action.

### Direct write endpoints

- `POST /multi-agent/actions/create-repo`
- `POST /multi-agent/actions/create-issue`
- `POST /multi-agent/actions/commit`

## Write-capable Internal Actions

Current policies cover these action names:

- `create_repo`
- `create_issue`
- `commit`
- `create_issue_comment`
- `dispatch_workflow`
- `push`

Only the first three are exposed through public API routes today. The others are included in the policy table now so future endpoints can reuse the same authorization boundary instead of introducing ad hoc checks.

## Examples

Read-only request:

```bash
curl http://127.0.0.1:8000/health
```

Operator request:

```bash
curl -X POST http://127.0.0.1:8000/multi-agent/actions/create-issue \
  -H "Content-Type: application/json" \
  -H "X-User-ID: operator-1" \
  -H "X-User-Role: operator" \
  -d '{
    "repo_full_name": "your-org/demo-repo",
    "title": "Bug login",
    "body": "Steps to reproduce",
    "confirmed": true
  }'
```

## Extensibility

The abstraction points intended for future SSO integration are:

- `AuthContext`
- `IdentityProvider`
- `resolve_request_auth_context(...)`
- `require_action_permission(...)`

To integrate SSO later, implement a new `IdentityProvider` that resolves the same `AuthContext` from SSO claims or trusted headers, then register it with `configure_identity_provider(...)`.
