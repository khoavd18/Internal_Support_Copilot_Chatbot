from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

import requests

from src.core.security import sanitize_error_text
from src.core.settings import (
    GITHUB_ACTIONS_ENABLED,
    GITHUB_ALLOWED_ORGS,
    GITHUB_ALLOWED_REPOS,
    GITHUB_API_BASE,
    GITHUB_API_VERSION,
    GITHUB_HTTP_TIMEOUT,
    GITHUB_INSTALLATION_ID,
)
from src.integrations.github_app_auth import GitHubAppAuth

logger = logging.getLogger(__name__)


class GitHubClientError(RuntimeError):
    """Raised when a GitHub API operation fails."""


class GitHubClient:
    def __init__(
        self,
        auth: Optional[GitHubAppAuth] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: Optional[int] = None,
        allowed_repos: Optional[Iterable[str]] = None,
        allowed_orgs: Optional[Iterable[str]] = None,
    ) -> None:
        self.auth = auth or GitHubAppAuth()
        self.api_base = (api_base or GITHUB_API_BASE).rstrip("/")
        self.api_version = str(api_version or GITHUB_API_VERSION).strip()
        self.timeout = int(timeout or GITHUB_HTTP_TIMEOUT)
        self.allowed_repos = {
            str(item).strip().lower()
            for item in (allowed_repos or GITHUB_ALLOWED_REPOS)
            if str(item).strip()
        }
        self.allowed_orgs = {
            str(item).strip().lower()
            for item in (allowed_orgs or GITHUB_ALLOWED_ORGS)
            if str(item).strip()
        }
        self.default_installation_id = str(GITHUB_INSTALLATION_ID or "").strip()
        self._session = requests.Session()

    @staticmethod
    def normalize_repo(repo_value: str) -> str:
        value = str(repo_value or "").strip()

        if value.startswith("https://github.com/"):
            value = value.replace("https://github.com/", "", 1).strip("/")

        if value.endswith(".git"):
            value = value[:-4]

        return value

    @classmethod
    def split_repo(cls, repo_full_name: str) -> tuple[str, str]:
        normalized = cls.normalize_repo(repo_full_name)
        if "/" not in normalized:
            raise GitHubClientError("Repository must use the format owner/repo")
        owner, repo = normalized.split("/", 1)
        owner = owner.strip()
        repo = repo.strip()
        if not owner or not repo:
            raise GitHubClientError("Repository must use the format owner/repo")
        return owner, repo

    def ensure_repo_allowed(self, repo_full_name: str) -> str:
        normalized = self.normalize_repo(repo_full_name).lower()
        if not normalized:
            raise GitHubClientError("Repository value must not be empty")

        if not self.allowed_repos:
            raise GitHubClientError(
                "GITHUB_ALLOWED_REPOS is empty. Configure an allowlist before enabling GitHub actions."
            )

        if normalized not in self.allowed_repos:
            raise GitHubClientError(f"Repository '{repo_full_name}' is not listed in GITHUB_ALLOWED_REPOS")

        return normalized

    def ensure_org_allowed(self, org: str) -> str:
        normalized = str(org or "").strip().lower()
        if not normalized:
            raise GitHubClientError("Organization value must not be empty")

        if not self.allowed_orgs:
            raise GitHubClientError(
                "GITHUB_ALLOWED_ORGS is empty. Configure an allowlist before enabling repository creation."
            )

        if normalized not in self.allowed_orgs:
            raise GitHubClientError(f"Organization '{org}' is not listed in GITHUB_ALLOWED_ORGS")

        return normalized

    def _base_headers(self, bearer_token: str) -> Dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {bearer_token}",
            "X-GitHub-Api-Version": self.api_version,
            "User-Agent": "internal-support-copilot-github-app",
        }

    @staticmethod
    def _safe_response_body(response: requests.Response) -> str:
        body = sanitize_error_text(response.text, max_length=240)
        return body or "<empty>"

    def get_repository_installation(self, repo_full_name: str) -> Dict[str, Any]:
        owner, repo = self.split_repo(repo_full_name)
        return self.auth.get_repo_installation(owner, repo)

    def get_organization_installation(self, org: str) -> Dict[str, Any]:
        normalized_org = self.ensure_org_allowed(org)
        return self.auth.get_org_installation(normalized_org)

    def resolve_installation_id(self, repo_full_name: str) -> str:
        if self.default_installation_id:
            return self.default_installation_id

        installation = self.get_repository_installation(repo_full_name)
        installation_id = installation.get("id")
        if not installation_id:
            raise GitHubClientError(f"Unable to resolve installation id for repository '{repo_full_name}'")
        return str(installation_id)

    def resolve_org_installation_id(self, org: str) -> str:
        if self.default_installation_id:
            return self.default_installation_id

        installation = self.get_organization_installation(org)
        installation_id = installation.get("id")
        if not installation_id:
            raise GitHubClientError(f"Unable to resolve installation id for organization '{org}'")
        return str(installation_id)

    def _request_installation(
        self,
        method: str,
        path: str,
        repo_full_name: str,
        json_body: Optional[Dict[str, Any]] = None,
        expected_statuses: tuple[int, ...] = (200, 201, 202, 204),
    ) -> requests.Response:
        if not GITHUB_ACTIONS_ENABLED:
            raise GitHubClientError("GITHUB_ACTIONS_ENABLED is disabled.")

        normalized_repo = self.ensure_repo_allowed(repo_full_name)
        installation_id = self.resolve_installation_id(normalized_repo)
        token = self.auth.get_installation_token(installation_id)
        url = f"{self.api_base}{path}"

        logger.info(
            "GitHub installation request started",
            extra={
                "event": "github.request.started",
                "method": method.upper(),
                "path": path,
                "repo": normalized_repo,
            },
        )
        response = self._session.request(
            method=method.upper(),
            url=url,
            headers=self._base_headers(token),
            json=json_body,
            timeout=self.timeout,
        )

        if response.status_code not in expected_statuses:
            logger.warning(
                "GitHub installation request failed",
                extra={
                    "event": "github.request.failed",
                    "method": method.upper(),
                    "path": path,
                    "repo": normalized_repo,
                    "status_code": response.status_code,
                },
            )
            raise GitHubClientError(
                f"GitHub API call failed. method={method.upper()} path={path} "
                f"status={response.status_code} body={self._safe_response_body(response)}"
            )

        logger.info(
            "GitHub installation request completed",
            extra={
                "event": "github.request.completed",
                "method": method.upper(),
                "path": path,
                "repo": normalized_repo,
                "status_code": response.status_code,
            },
        )
        return response

    def _request_installation_by_org(
        self,
        method: str,
        path: str,
        org: str,
        json_body: Optional[Dict[str, Any]] = None,
        expected_statuses: tuple[int, ...] = (200, 201, 202, 204),
    ) -> requests.Response:
        if not GITHUB_ACTIONS_ENABLED:
            raise GitHubClientError("GITHUB_ACTIONS_ENABLED is disabled.")

        normalized_org = self.ensure_org_allowed(org)
        installation_id = self.resolve_org_installation_id(normalized_org)
        token = self.auth.get_installation_token(installation_id)
        url = f"{self.api_base}{path}"

        logger.info(
            "GitHub organization request started",
            extra={
                "event": "github.request.started",
                "method": method.upper(),
                "path": path,
                "org": normalized_org,
            },
        )
        response = self._session.request(
            method=method.upper(),
            url=url,
            headers=self._base_headers(token),
            json=json_body,
            timeout=self.timeout,
        )

        if response.status_code not in expected_statuses:
            logger.warning(
                "GitHub organization request failed",
                extra={
                    "event": "github.request.failed",
                    "method": method.upper(),
                    "path": path,
                    "org": normalized_org,
                    "status_code": response.status_code,
                },
            )
            raise GitHubClientError(
                f"GitHub API call failed. method={method.upper()} path={path} "
                f"status={response.status_code} body={self._safe_response_body(response)}"
            )

        logger.info(
            "GitHub organization request completed",
            extra={
                "event": "github.request.completed",
                "method": method.upper(),
                "path": path,
                "org": normalized_org,
                "status_code": response.status_code,
            },
        )
        return response

    def create_issue(
        self,
        repo_full_name: str,
        title: str,
        body: str,
        labels: Optional[list[str]] = None,
        assignees: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        owner, repo = self.split_repo(repo_full_name)
        payload: Dict[str, Any] = {
            "title": title.strip(),
            "body": body.strip(),
        }
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees

        response = self._request_installation(
            method="POST",
            path=f"/repos/{owner}/{repo}/issues",
            repo_full_name=repo_full_name,
            json_body=payload,
            expected_statuses=(201,),
        )
        data = response.json()
        logger.info(
            "GitHub issue created",
            extra={
                "event": "github.issue.created",
                "repo": repo_full_name,
                "issue_number": data.get("number"),
                "labels_count": len(labels or []),
                "assignees_count": len(assignees or []),
            },
        )

        return {
            "issue_number": data.get("number"),
            "title": data.get("title"),
            "html_url": data.get("html_url"),
            "state": data.get("state"),
            "id": data.get("id"),
        }

    def create_issue_comment(
        self,
        repo_full_name: str,
        issue_number: int,
        body: str,
    ) -> Dict[str, Any]:
        owner, repo = self.split_repo(repo_full_name)

        response = self._request_installation(
            method="POST",
            path=f"/repos/{owner}/{repo}/issues/{int(issue_number)}/comments",
            repo_full_name=repo_full_name,
            json_body={"body": body.strip()},
            expected_statuses=(201,),
        )
        data = response.json()
        logger.info(
            "GitHub issue comment created",
            extra={
                "event": "github.issue_comment.created",
                "repo": repo_full_name,
                "issue_number": int(issue_number),
                "comment_id": data.get("id"),
            },
        )

        return {
            "comment_id": data.get("id"),
            "html_url": data.get("html_url"),
            "issue_url": data.get("issue_url"),
            "created_at": data.get("created_at"),
        }

    def dispatch_workflow(
        self,
        repo_full_name: str,
        workflow_id: str,
        ref: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        owner, repo = self.split_repo(repo_full_name)
        payload = {
            "ref": ref.strip(),
            "inputs": inputs or {},
        }

        response = self._request_installation(
            method="POST",
            path=f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
            repo_full_name=repo_full_name,
            json_body=payload,
            expected_statuses=(200, 204),
        )

        result: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "ref": ref,
            "accepted": response.status_code in {200, 204},
            "inputs": inputs or {},
            "status_code": response.status_code,
        }

        if response.status_code == 200:
            try:
                data = response.json()
            except Exception:
                data = {}
            result["workflow_run_id"] = data.get("workflow_run_id")
            result["run_url"] = data.get("run_url")
            result["html_url"] = data.get("html_url")

        logger.info(
            "GitHub workflow dispatched",
            extra={
                "event": "github.workflow.dispatched",
                "repo": repo_full_name,
                "workflow_id": workflow_id,
                "ref": ref,
                "status_code": response.status_code,
            },
        )
        return result

    def create_organization_repository(
        self,
        org: str,
        name: str,
        *,
        description: str = "",
        private: bool = True,
        auto_init: bool = False,
    ) -> Dict[str, Any]:
        normalized_org = self.ensure_org_allowed(org)
        repo_name = str(name or "").strip()
        if not repo_name:
            raise GitHubClientError("Repository name must not be empty")

        payload: Dict[str, Any] = {
            "name": repo_name,
            "description": str(description or "").strip(),
            "private": bool(private),
            "auto_init": bool(auto_init),
        }

        response = self._request_installation_by_org(
            method="POST",
            path=f"/orgs/{normalized_org}/repos",
            org=normalized_org,
            json_body=payload,
            expected_statuses=(201,),
        )
        data = response.json()
        logger.info(
            "GitHub repository created",
            extra={
                "event": "github.repo.created",
                "org": normalized_org,
                "repo_name": repo_name,
                "private": bool(private),
            },
        )

        return {
            "id": data.get("id"),
            "name": data.get("name"),
            "full_name": data.get("full_name"),
            "private": data.get("private"),
            "html_url": data.get("html_url"),
            "default_branch": data.get("default_branch"),
        }

    def healthcheck(self, repo_full_name: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "actions_enabled": GITHUB_ACTIONS_ENABLED,
            "allowed_repos": sorted(self.allowed_repos),
            "allowed_orgs": sorted(self.allowed_orgs),
            "auth": self.auth.auth_health(),
        }

        if repo_full_name:
            try:
                normalized_repo = self.ensure_repo_allowed(repo_full_name)
                installation = self.get_repository_installation(normalized_repo)
                payload["repo"] = normalized_repo
                payload["installation_id"] = installation.get("id")
                payload["installation_target_type"] = installation.get("target_type")
                payload["installation_repository_selection"] = installation.get("repository_selection")
            except Exception as exc:
                payload["repo"] = repo_full_name
                payload["repo_check_error"] = sanitize_error_text(exc, max_length=240)

        return payload
