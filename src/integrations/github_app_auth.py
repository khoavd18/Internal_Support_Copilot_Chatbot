from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import jwt
import requests

from src.core.security import sanitize_error_text
from src.core.settings import (
    GITHUB_API_BASE,
    GITHUB_API_VERSION,
    GITHUB_APP_ID,
    GITHUB_CLIENT_ID,
    GITHUB_HTTP_TIMEOUT,
    GITHUB_PRIVATE_KEY_PATH,
)


class GitHubAppAuthError(RuntimeError):
    """Raised when GitHub App authentication fails."""


class GitHubAppAuth:
    def __init__(
        self,
        app_id: Optional[str] = None,
        client_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.app_id = str(app_id or GITHUB_APP_ID).strip()
        self.client_id = str(client_id or GITHUB_CLIENT_ID).strip()
        self.private_key_path = str(private_key_path or GITHUB_PRIVATE_KEY_PATH).strip()
        self.api_base = (api_base or GITHUB_API_BASE).rstrip("/")
        self.api_version = str(api_version or GITHUB_API_VERSION).strip()
        self.timeout = int(timeout or GITHUB_HTTP_TIMEOUT)

        self._session = requests.Session()
        self._lock = threading.Lock()
        self._installation_tokens: Dict[str, str] = {}
        self._installation_expiries: Dict[str, datetime] = {}

    def is_configured(self) -> bool:
        return bool((self.client_id or self.app_id) and self.private_key_path)

    def _validate_base_config(self) -> None:
        missing = []

        if not (self.client_id or self.app_id):
            missing.append("GITHUB_CLIENT_ID or GITHUB_APP_ID")

        if not self.private_key_path:
            missing.append("GITHUB_PRIVATE_KEY_PATH")

        if missing:
            raise GitHubAppAuthError("Missing GitHub App configuration: " + ", ".join(missing))

        pem_path = Path(self.private_key_path)
        if not pem_path.exists():
            raise GitHubAppAuthError(
                "Configured GitHub App private key file was not found."
            )

    @staticmethod
    def _safe_response_body(response: requests.Response) -> str:
        body = sanitize_error_text(response.text, max_length=240)
        return body or "<empty>"

    def _read_private_key_bytes(self) -> bytes:
        self._validate_base_config()
        return Path(self.private_key_path).read_bytes()

    def build_app_jwt(self) -> str:
        private_key_pem = self._read_private_key_bytes()
        now = int(time.time())
        issuer = self.client_id or self.app_id

        payload = {
            "iat": now - 60,
            "exp": now + 9 * 60,
            "iss": issuer,
        }

        encoded = jwt.encode(
            payload,
            private_key_pem,
            algorithm="RS256",
        )
        if isinstance(encoded, bytes):
            return encoded.decode("utf-8")
        return str(encoded)

    def _base_headers(self, bearer_token: str) -> Dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {bearer_token}",
            "X-GitHub-Api-Version": self.api_version,
            "User-Agent": "internal-support-copilot-github-app",
        }

    @staticmethod
    def _parse_github_datetime(value: str) -> datetime:
        normalized = str(value).strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)

    def get_authenticated_app(self) -> Dict[str, Any]:
        app_jwt = self.build_app_jwt()
        response = self._session.get(
            f"{self.api_base}/app",
            headers=self._base_headers(app_jwt),
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise GitHubAppAuthError(
                f"GET /app failed. status={response.status_code} "
                f"body={self._safe_response_body(response)}"
            )

        return response.json()

    def get_repo_installation(self, owner: str, repo: str) -> Dict[str, Any]:
        app_jwt = self.build_app_jwt()
        response = self._session.get(
            f"{self.api_base}/repos/{owner}/{repo}/installation",
            headers=self._base_headers(app_jwt),
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise GitHubAppAuthError(
                f"GET /repos/{owner}/{repo}/installation failed. "
                f"status={response.status_code} body={self._safe_response_body(response)}"
            )

        return response.json()

    def get_org_installation(self, org: str) -> Dict[str, Any]:
        app_jwt = self.build_app_jwt()
        response = self._session.get(
            f"{self.api_base}/orgs/{org}/installation",
            headers=self._base_headers(app_jwt),
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise GitHubAppAuthError(
                f"GET /orgs/{org}/installation failed. "
                f"status={response.status_code} body={self._safe_response_body(response)}"
            )

        return response.json()

    def get_installation_token(self, installation_id: str) -> str:
        installation_id = str(installation_id or "").strip()
        if not installation_id:
            raise GitHubAppAuthError("installation_id must not be empty")

        with self._lock:
            cached_token = self._installation_tokens.get(installation_id)
            cached_expiry = self._installation_expiries.get(installation_id)

            if cached_token and cached_expiry:
                now_utc = datetime.now(timezone.utc)
                if now_utc.timestamp() < cached_expiry.timestamp() - 60:
                    return cached_token

            app_jwt = self.build_app_jwt()
            response = self._session.post(
                f"{self.api_base}/app/installations/{installation_id}/access_tokens",
                headers=self._base_headers(app_jwt),
                timeout=self.timeout,
            )

            if response.status_code >= 400:
                raise GitHubAppAuthError(
                    f"POST /app/installations/{installation_id}/access_tokens failed. "
                    f"status={response.status_code} body={self._safe_response_body(response)}"
                )

            data = response.json()
            token = data.get("token")
            expires_at = data.get("expires_at")

            if not token or not expires_at:
                raise GitHubAppAuthError("GitHub did not return a valid installation token response.")

            expiry_dt = self._parse_github_datetime(expires_at)
            self._installation_tokens[installation_id] = token
            self._installation_expiries[installation_id] = expiry_dt
            return token

    def auth_health(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "configured": self.is_configured(),
            "issuer_present": bool(self.client_id or self.app_id),
            "private_key_configured": bool(self.private_key_path),
            "private_key_exists": (
                Path(self.private_key_path).expanduser().exists()
                if self.private_key_path
                else False
            ),
            "api_base": self.api_base,
            "api_version": self.api_version,
        }

        if not self.is_configured():
            return payload

        try:
            jwt_token = self.build_app_jwt()
            payload["jwt_ready"] = bool(jwt_token)
        except Exception as exc:
            payload["jwt_ready"] = False
            payload["error"] = sanitize_error_text(exc, max_length=240)

        return payload
