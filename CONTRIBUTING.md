# Contributing

Thanks for your interest in improving Internal Support Copilot.

## Local Setup

Use Python `3.10` or `3.11`; the project metadata intentionally excludes Python `3.12+` until the ML dependency stack is validated there.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/dev.py install
Copy-Item .env.example .env
```

## Development Expectations

- Keep changes scoped and easy to review.
- Prefer small, testable functions over broad, implicit behavior.
- Keep integration side effects behind dedicated client modules.
- Update docs when you add new endpoints, environment variables, or workflows.

## Validation

Run these before opening a pull request:

```powershell
python -m ruff check .
python -m ruff format --check .
python scripts/dev.py run-tests
```

## Pull Request Notes

- Describe the user-facing impact.
- Mention any new configuration flags.
- Call out any intentional follow-up work or known limitations.
