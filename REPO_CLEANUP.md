# Repository Cleanup Summary

This cleanup focused on repository hygiene only. No application logic was changed.

## What Was Cleaned

- Expanded `.gitignore` to better exclude Python caches, virtual environments, local secrets, logs, coverage outputs, editor junk, and packaging artifacts.
- Removed generated repository artifacts that should stay local, including:
  - `data_source/processed/`
  - `eval/reports/*.txt`
  - `pytest-cache-files-*`
  - `__pycache__/` directories
  - local archive artifacts such as `Internal_Support_Copilot.zip`
- Updated `README.md` so the setup flow matches the cleaned repo state and explains that processed data must be generated locally.

## Why

- Generated files and caches create noisy diffs and make the repository harder to review.
- Local secrets and machine-specific artifacts should never be versioned.
- Accurate setup documentation helps a clean clone behave predictably.
