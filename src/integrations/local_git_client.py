from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from src.core.settings import LOCAL_GIT_ALLOWED_ROOTS, LOCAL_GIT_DEFAULT_REPO_PATH

logger = logging.getLogger(__name__)


class LocalGitClientError(RuntimeError):
    """Raised when a local git operation fails."""


GitRunner = Callable[..., subprocess.CompletedProcess[str]]


class LocalGitClient:
    def __init__(
        self,
        *,
        default_repo_path: Optional[Path | str] = None,
        allowed_roots: Optional[Iterable[Path | str]] = None,
        runner: Optional[GitRunner] = None,
    ) -> None:
        self.default_repo_path = Path(default_repo_path or LOCAL_GIT_DEFAULT_REPO_PATH).expanduser().resolve()
        source_roots = allowed_roots or LOCAL_GIT_ALLOWED_ROOTS
        self.allowed_roots = [Path(item).expanduser().resolve() for item in source_roots]
        self._runner = runner or subprocess.run

    def _ensure_allowed_path(self, candidate: Path) -> Path:
        resolved = candidate.expanduser().resolve()
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue

        allowed = ", ".join(str(root) for root in self.allowed_roots)
        raise LocalGitClientError(f"Path '{resolved}' is outside allowed roots: {allowed}")

    def _run_git(
        self,
        repo_path: Path,
        args: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = ["git", *args]
        logger.info(
            "Running git command",
            extra={
                "event": "local_git.command.started",
                "repo_path": str(repo_path),
                "git_args": list(args),
            },
        )
        result = self._runner(
            command,
            cwd=str(repo_path),
            text=True,
            capture_output=True,
        )

        if check and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            message = stderr or stdout or f"git {' '.join(args)} failed"
            logger.warning(
                "Git command failed",
                extra={
                    "event": "local_git.command.failed",
                    "repo_path": str(repo_path),
                    "git_args": list(args),
                    "returncode": result.returncode,
                },
            )
            raise LocalGitClientError(message)

        logger.info(
            "Git command completed",
            extra={
                "event": "local_git.command.completed",
                "repo_path": str(repo_path),
                "git_args": list(args),
                "returncode": result.returncode,
            },
        )
        return result

    def resolve_repo_path(self, repo_path: Optional[Path | str] = None) -> Path:
        requested = Path(repo_path or self.default_repo_path)
        allowed_path = self._ensure_allowed_path(requested)
        top_level = self._run_git(allowed_path, ["rev-parse", "--show-toplevel"]).stdout.strip()
        if not top_level:
            raise LocalGitClientError("Unable to resolve the git repository root")
        return self._ensure_allowed_path(Path(top_level))

    def _normalize_paths(self, repo_root: Path, paths: Iterable[str]) -> list[str]:
        normalized_paths: list[str] = []
        seen = set()

        for raw_path in paths:
            path_value = str(raw_path or "").strip()
            if not path_value:
                continue

            candidate = Path(path_value)
            resolved = candidate if candidate.is_absolute() else repo_root / candidate
            resolved = self._ensure_allowed_path(resolved)

            try:
                relative = resolved.relative_to(repo_root)
            except ValueError as exc:
                raise LocalGitClientError(
                    f"File '{resolved}' is outside git repo '{repo_root}'"
                ) from exc

            relative_value = relative.as_posix()
            if relative_value not in seen:
                seen.add(relative_value)
                normalized_paths.append(relative_value)

        return normalized_paths

    def commit(
        self,
        *,
        message: str,
        repo_path: Optional[Path | str] = None,
        paths: Optional[Iterable[str]] = None,
        stage_all: bool = False,
        include_untracked: bool = False,
    ) -> Dict[str, Any]:
        commit_message = str(message or "").strip()
        if not commit_message:
            raise LocalGitClientError("Commit message must not be empty")

        repo_root = self.resolve_repo_path(repo_path)
        normalized_paths = self._normalize_paths(repo_root, paths or [])
        logger.info(
            "Local git commit requested",
            extra={
                "event": "local_git.commit.started",
                "repo_path": str(repo_root),
                "paths_count": len(normalized_paths),
                "stage_all": stage_all,
                "include_untracked": include_untracked,
                "message_length": len(commit_message),
            },
        )

        if normalized_paths:
            self._run_git(repo_root, ["add", "--", *normalized_paths])
        elif stage_all:
            add_args = ["add", "--all"] if include_untracked else ["add", "-u"]
            self._run_git(repo_root, add_args)

        staged_paths = [
            line.strip()
            for line in self._run_git(repo_root, ["diff", "--cached", "--name-only"]).stdout.splitlines()
            if line.strip()
        ]
        if not staged_paths:
            raise LocalGitClientError("No staged changes were found for commit creation")

        status_before = [
            line.rstrip()
            for line in self._run_git(repo_root, ["status", "--short"]).stdout.splitlines()
            if line.strip()
        ]

        self._run_git(repo_root, ["commit", "-m", commit_message])
        commit_sha = self._run_git(repo_root, ["rev-parse", "HEAD"]).stdout.strip()
        branch = self._run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        logger.info(
            "Local git commit created",
            extra={
                "event": "local_git.commit.completed",
                "repo_path": str(repo_root),
                "branch": branch,
                "commit_sha": commit_sha,
                "staged_paths_count": len(staged_paths),
            },
        )

        return {
            "repo_path": str(repo_root),
            "branch": branch,
            "commit_sha": commit_sha,
            "message": commit_message,
            "staged_paths": staged_paths,
            "status_before": status_before,
            "stage_all": stage_all,
            "include_untracked": include_untracked,
        }
