from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol

from fastapi import Request

from src.core.settings import (
    AUTH_ALLOW_ANONYMOUS_READS,
    AUTH_DEFAULT_ROLE,
    AUTH_ENABLED,
    AUTH_REQUIRE_USER_ID_FOR_OPERATOR,
    AUTH_ROLE_HEADER,
    AUTH_USER_HEADER,
)


VALID_ROLES = ("viewer", "operator")
ROLE_RANK = {
    "viewer": 10,
    "operator": 20,
}
ANONYMOUS_USER_ID = "anonymous"


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    role: str
    auth_source: str = "headers"
    authenticated: bool = False
    claims: Dict[str, Any] = field(default_factory=dict)

    def has_role(self, required_role: str) -> bool:
        return ROLE_RANK.get(self.role, 0) >= ROLE_RANK.get(str(required_role or "").strip().lower(), 0)


@dataclass(frozen=True)
class AuthorizationPolicy:
    name: str
    min_role: str
    description: str
    write_capable: bool = True


@dataclass(frozen=True)
class EndpointPolicy:
    path: str
    methods: tuple[str, ...]
    capability: str
    description: str
    min_role: str = "viewer"


class AuthorizationError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int = 403):
        super().__init__(detail)
        self.detail = str(detail or "").strip()
        self.status_code = int(status_code)


class IdentityProvider(Protocol):
    def resolve_identity(self, headers: Mapping[str, str]) -> AuthContext:
        ...


WRITE_ACTION_POLICIES: Dict[str, AuthorizationPolicy] = {
    "create_repo": AuthorizationPolicy(
        name="create_repo",
        min_role="operator",
        description="Create a GitHub repository.",
    ),
    "create_issue": AuthorizationPolicy(
        name="create_issue",
        min_role="operator",
        description="Create a GitHub issue.",
    ),
    "commit": AuthorizationPolicy(
        name="commit",
        min_role="operator",
        description="Create a local git commit.",
    ),
    "create_issue_comment": AuthorizationPolicy(
        name="create_issue_comment",
        min_role="operator",
        description="Create a GitHub issue comment.",
    ),
    "dispatch_workflow": AuthorizationPolicy(
        name="dispatch_workflow",
        min_role="operator",
        description="Dispatch a GitHub Actions workflow.",
    ),
    "push": AuthorizationPolicy(
        name="push",
        min_role="operator",
        description="Push local git changes to a remote.",
    ),
}

ENDPOINT_AUTH_AUDIT: Dict[str, EndpointPolicy] = {
    "/health": EndpointPolicy(
        path="/health",
        methods=("GET",),
        capability="read",
        description="Health snapshot endpoint.",
    ),
    "/ready": EndpointPolicy(
        path="/ready",
        methods=("GET",),
        capability="read",
        description="Readiness probe endpoint.",
    ),
    "/ask": EndpointPolicy(
        path="/ask",
        methods=("POST",),
        capability="read",
        description="Plain retrieval and answer endpoint.",
    ),
    "/agent/ask": EndpointPolicy(
        path="/agent/ask",
        methods=("POST",),
        capability="mixed",
        description="Read-only by default, but operator role is required when chat input triggers a write action.",
    ),
    "/multi-agent/ask": EndpointPolicy(
        path="/multi-agent/ask",
        methods=("POST",),
        capability="mixed",
        description="Read-only by default, but operator role is required when chat input triggers a write action.",
    ),
    "/multi-agent/actions/create-repo": EndpointPolicy(
        path="/multi-agent/actions/create-repo",
        methods=("POST",),
        capability="write",
        description="Direct repository creation action.",
        min_role="operator",
    ),
    "/multi-agent/actions/create-issue": EndpointPolicy(
        path="/multi-agent/actions/create-issue",
        methods=("POST",),
        capability="write",
        description="Direct GitHub issue creation action.",
        min_role="operator",
    ),
    "/multi-agent/actions/commit": EndpointPolicy(
        path="/multi-agent/actions/commit",
        methods=("POST",),
        capability="write",
        description="Direct local git commit action.",
        min_role="operator",
    ),
}


class HeaderIdentityProvider:
    def resolve_identity(self, headers: Mapping[str, str]) -> AuthContext:
        raw_user_id = str(headers.get(AUTH_USER_HEADER) or "").strip()
        raw_role = str(headers.get(AUTH_ROLE_HEADER) or "").strip().lower()

        if raw_role and raw_role not in VALID_ROLES:
            raise AuthorizationError(
                f"Unsupported role '{raw_role}'. Allowed roles: {', '.join(VALID_ROLES)}.",
                status_code=401,
            )

        if not AUTH_ENABLED:
            resolved_role = raw_role or "operator"
            resolved_user_id = raw_user_id or "local-dev"
            return AuthContext(
                user_id=resolved_user_id,
                role=resolved_role,
                auth_source="disabled",
                authenticated=bool(raw_user_id),
                claims={
                    "user_header": AUTH_USER_HEADER,
                    "role_header": AUTH_ROLE_HEADER,
                },
            )

        resolved_role = raw_role or AUTH_DEFAULT_ROLE
        if resolved_role not in VALID_ROLES:
            raise AuthorizationError(
                f"Unsupported default role '{resolved_role}'. Allowed roles: {', '.join(VALID_ROLES)}.",
                status_code=500,
            )

        if resolved_role == "operator" and AUTH_REQUIRE_USER_ID_FOR_OPERATOR and not raw_user_id:
            raise AuthorizationError(
                f"Operator requests must include {AUTH_USER_HEADER}.",
                status_code=401,
            )

        if raw_user_id:
            resolved_user_id = raw_user_id
            authenticated = True
            auth_source = "headers" if raw_role else "default_role"
        elif AUTH_ALLOW_ANONYMOUS_READS:
            resolved_user_id = ANONYMOUS_USER_ID
            authenticated = False
            auth_source = "anonymous"
        else:
            raise AuthorizationError(
                "Authentication is required for this request.",
                status_code=401,
            )

        return AuthContext(
            user_id=resolved_user_id,
            role=resolved_role,
            auth_source=auth_source,
            authenticated=authenticated,
            claims={
                "user_header": AUTH_USER_HEADER,
                "role_header": AUTH_ROLE_HEADER,
                "raw_role": raw_role,
            },
        )


_IDENTITY_PROVIDER: IdentityProvider | None = None


def configure_identity_provider(provider: IdentityProvider) -> None:
    global _IDENTITY_PROVIDER
    _IDENTITY_PROVIDER = provider


def get_identity_provider() -> IdentityProvider:
    global _IDENTITY_PROVIDER
    if _IDENTITY_PROVIDER is None:
        _IDENTITY_PROVIDER = HeaderIdentityProvider()
    return _IDENTITY_PROVIDER


def resolve_identity_from_headers(headers: Mapping[str, str]) -> AuthContext:
    return get_identity_provider().resolve_identity(headers)


def resolve_request_auth_context(request: Request) -> AuthContext:
    existing = getattr(request.state, "auth_context", None)
    if existing is not None:
        return existing

    resolved = resolve_identity_from_headers(request.headers)
    request.state.auth_context = resolved
    return resolved


def get_action_policy(action_name: str) -> AuthorizationPolicy:
    normalized = str(action_name or "").strip().lower()
    if normalized in WRITE_ACTION_POLICIES:
        return WRITE_ACTION_POLICIES[normalized]
    return AuthorizationPolicy(
        name=normalized or "unknown",
        min_role="operator",
        description="Unknown write-capable action defaults to operator-only access.",
    )


def authorize_context_for_action(actor: AuthContext, action_name: str) -> AuthorizationPolicy:
    policy = get_action_policy(action_name)
    if actor.has_role(policy.min_role):
        return policy
    raise AuthorizationError(
        f"{policy.min_role.capitalize()} role is required for action '{policy.name}'.",
        status_code=403,
    )


def require_action_permission(request: Request, action_name: str) -> AuthContext:
    actor = resolve_request_auth_context(request)
    authorize_context_for_action(actor, action_name)
    return actor


def build_authorization_summary() -> Dict[str, Any]:
    return {
        "enabled": AUTH_ENABLED,
        "provider": "header",
        "user_header": AUTH_USER_HEADER,
        "role_header": AUTH_ROLE_HEADER,
        "default_role": AUTH_DEFAULT_ROLE,
        "allow_anonymous_reads": AUTH_ALLOW_ANONYMOUS_READS,
        "require_user_id_for_operator": AUTH_REQUIRE_USER_ID_FOR_OPERATOR,
        "valid_roles": list(VALID_ROLES),
    }


__all__ = [
    "ANONYMOUS_USER_ID",
    "AuthContext",
    "AuthorizationError",
    "AuthorizationPolicy",
    "ENDPOINT_AUTH_AUDIT",
    "EndpointPolicy",
    "HeaderIdentityProvider",
    "IdentityProvider",
    "VALID_ROLES",
    "WRITE_ACTION_POLICIES",
    "authorize_context_for_action",
    "build_authorization_summary",
    "configure_identity_provider",
    "get_action_policy",
    "get_identity_provider",
    "require_action_permission",
    "resolve_identity_from_headers",
    "resolve_request_auth_context",
]
