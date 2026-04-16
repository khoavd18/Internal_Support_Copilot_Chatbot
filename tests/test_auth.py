from __future__ import annotations

import pytest

from src.core import auth
from src.core.auth import (
    AuthContext,
    AuthorizationError,
    authorize_context_for_action,
    build_authorization_summary,
    resolve_identity_from_headers,
)


def test_resolve_identity_defaults_to_anonymous_viewer():
    actor = resolve_identity_from_headers({})

    assert actor.user_id == "anonymous"
    assert actor.role == "viewer"
    assert actor.authenticated is False


def test_resolve_identity_rejects_invalid_role():
    with pytest.raises(AuthorizationError, match="Unsupported role"):
        resolve_identity_from_headers(
            {
                auth.AUTH_USER_HEADER: "alice",
                auth.AUTH_ROLE_HEADER: "admin",
            }
        )


def test_resolve_identity_requires_user_id_for_operator():
    with pytest.raises(AuthorizationError, match=auth.AUTH_USER_HEADER):
        resolve_identity_from_headers(
            {
                auth.AUTH_ROLE_HEADER: "operator",
            }
        )


def test_authorize_context_for_action_rejects_viewer():
    actor = AuthContext(user_id="viewer-1", role="viewer", authenticated=True)

    with pytest.raises(AuthorizationError, match="Operator role is required"):
        authorize_context_for_action(actor, "create_issue")


def test_build_authorization_summary_lists_header_provider():
    summary = build_authorization_summary()

    assert summary["provider"] == "header"
    assert "viewer" in summary["valid_roles"]
    assert "operator" in summary["valid_roles"]
