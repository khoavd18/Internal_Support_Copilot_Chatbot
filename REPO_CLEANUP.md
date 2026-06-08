# Repository Cleanup Summary

This cleanup focused on repository hygiene only. No application logic was changed.

## What Was Cleaned

- Expanded `.gitignore` and `.dockerignore` to better exclude Python caches, virtual environments, local secrets, logs, coverage outputs, editor junk, model caches, and packaging artifacts.
- Removed generated Python bytecode that should stay local, including `__pycache__/` directories and `*.pyc` files.
- Kept application features and the existing processed snapshot intact; regenerated or additional `data_source/processed` outputs should only be committed deliberately.
- Updated `README.md` and `CONTRIBUTING.md` so setup, Python-version, test, and evaluation guidance matches the current repo.

## Why

- Generated files and caches create noisy diffs and make the repository harder to review.
- Local secrets and machine-specific artifacts should never be versioned.
- Accurate setup documentation helps a clean clone behave predictably.
