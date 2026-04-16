# Secret Handling

This repository keeps secret handling lightweight and explicit:

- Runtime configuration still comes from environment variables and `.env`.
- Secret-bearing values are read through shared config helpers in `src/core/security.py`.
- Structured logging and health/readiness diagnostics redact known secret values, URL credentials, and common token-style fields.

## Secret-Bearing Settings

Current high-sensitivity settings include:

- `QDRANT_API_KEY`
- `REDIS_URL` when it embeds credentials
- `GITHUB_PRIVATE_KEY_PATH`

Related identifiers such as `GITHUB_APP_ID`, `GITHUB_CLIENT_ID`, and `GITHUB_INSTALLATION_ID` are configuration values, but they should still be handled with normal operational care.

## Secure Local Development

- Keep real credentials only in your local `.env`.
- Do not commit `.env`, PEM files, API keys, or password-bearing URLs.
- Prefer storing the GitHub App PEM outside the repository and referencing it with `GITHUB_PRIVATE_KEY_PATH`.
- If Redis or Qdrant require authentication, put those credentials only in local environment variables.
- Treat `/health`, `/ready`, and server logs as internal operational surfaces even though they now redact secrets.
- If a credential is pasted into a shell, log, or ticket by mistake, rotate it promptly.

## Scope

This hardening focuses on preventing accidental exposure through logs, readiness payloads, and API error responses. It does not replace secret managers, vaults, or enterprise key rotation workflows.
