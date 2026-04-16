# Contributing

Thanks for your interest in improving Internal Support Copilot.

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
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
ruff check .
pytest -q
```

## Pull Request Notes

- Describe the user-facing impact.
- Mention any new configuration flags.
- Call out any intentional follow-up work or known limitations.
