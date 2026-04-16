from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from qdrant_client import QdrantClient

from src.core.security import redact_url_credentials, sanitize_error_text
from src.core.settings import (
    AUTH_ALLOW_ANONYMOUS_READS,
    AUTH_DEFAULT_ROLE,
    AUTH_ENABLED,
    AUTH_REQUIRE_USER_ID_FOR_OPERATOR,
    AUTH_ROLE_HEADER,
    AUTH_USER_HEADER,
    ACTION_STORE_BACKEND,
    API_CORS_ORIGINS,
    CROSS_ENCODER_MODEL_NAME,
    DOCUMENTS_PATH,
    GRAPH_CHECKPOINTER_BACKEND,
    GITHUB_ACTIONS_ENABLED,
    GITHUB_ALLOWED_ORGS,
    GITHUB_ALLOWED_REPOS,
    GITHUB_APP_ID,
    GITHUB_CLIENT_ID,
    GITHUB_INSTALLATION_ID,
    GITHUB_PRIVATE_KEY_PATH,
    INCLUDE_TICKETS,
    LOCAL_GIT_ACTIONS_ENABLED,
    LOCAL_GIT_ALLOWED_ROOTS,
    LOCAL_GIT_DEFAULT_REPO_PATH,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_MODE,
    QDRANT_PREFER_GRPC,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
    REDIS_URL,
    REDIS_KEY_PREFIX,
    SESSION_STORE_BACKEND,
    TICKETS_PATH,
    USE_CROSS_ENCODER,
    USE_QDRANT_HYBRID,
)
from src.integrations.github_app_auth import GitHubAppAuth
from src.integrations.local_git_client import LocalGitClient
from src.persistence.action_store import get_action_store
from src.persistence.checkpoints import get_graph_checkpointer
from src.persistence.session_store import get_session_state_store


VALID_AUTH_ROLES = {"viewer", "operator"}


def _sanitize_error_message(error: Exception | str) -> str:
    return sanitize_error_text(
        error,
        extra_secret_values=[QDRANT_API_KEY, GITHUB_PRIVATE_KEY_PATH],
    )


def _is_within_any_root(candidate: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _redact_redis_url(url: str) -> str:
    return redact_url_credentials(url)


def validate_environment_settings() -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if SESSION_STORE_BACKEND not in {"memory", "redis"}:
        errors.append("SESSION_STORE_BACKEND must be one of: memory, redis.")

    if ACTION_STORE_BACKEND not in {"memory", "redis"}:
        errors.append("ACTION_STORE_BACKEND must be one of: memory, redis.")

    if GRAPH_CHECKPOINTER_BACKEND not in {"memory", "redis"}:
        errors.append("GRAPH_CHECKPOINTER_BACKEND must be one of: memory, redis.")

    if "redis" in {SESSION_STORE_BACKEND, ACTION_STORE_BACKEND, GRAPH_CHECKPOINTER_BACKEND} and not REDIS_URL:
        errors.append("Redis-backed persistence requires REDIS_URL.")

    if QDRANT_MODE not in {"server", "local"}:
        errors.append("QDRANT_MODE must be one of: server, local.")

    if QDRANT_MODE == "server" and not str(QDRANT_URL).strip():
        errors.append("QDRANT_MODE=server requires QDRANT_URL.")

    if not str(QDRANT_COLLECTION_NAME).strip():
        errors.append("QDRANT_COLLECTION_NAME must not be empty.")

    if not str(QDRANT_VECTOR_NAME).strip():
        errors.append("QDRANT_VECTOR_NAME must not be empty.")

    if USE_QDRANT_HYBRID and not str(QDRANT_SPARSE_VECTOR_NAME).strip():
        errors.append("USE_QDRANT_HYBRID=true requires QDRANT_SPARSE_VECTOR_NAME.")

    if USE_CROSS_ENCODER and not str(CROSS_ENCODER_MODEL_NAME).strip():
        errors.append("USE_CROSS_ENCODER=true requires CROSS_ENCODER_MODEL_NAME.")

    if not API_CORS_ORIGINS:
        warnings.append("API_CORS_ORIGINS is empty; browser clients may fail cross-origin requests.")

    if not AUTH_USER_HEADER:
        errors.append("AUTH_USER_HEADER must not be empty.")

    if not AUTH_ROLE_HEADER:
        errors.append("AUTH_ROLE_HEADER must not be empty.")

    if AUTH_DEFAULT_ROLE not in VALID_AUTH_ROLES:
        errors.append("AUTH_DEFAULT_ROLE must be one of: viewer, operator.")

    if AUTH_ENABLED and AUTH_ALLOW_ANONYMOUS_READS and AUTH_DEFAULT_ROLE == "operator":
        errors.append(
            "AUTH_ALLOW_ANONYMOUS_READS=true cannot be combined with AUTH_DEFAULT_ROLE=operator."
        )

    if not AUTH_ENABLED:
        warnings.append(
            "AUTH_ENABLED is false; write actions will bypass role checks and run as local-dev operator."
        )

    if AUTH_ENABLED and not AUTH_REQUIRE_USER_ID_FOR_OPERATOR:
        warnings.append(
            "AUTH_REQUIRE_USER_ID_FOR_OPERATOR is false; operator actions may have weaker auditability."
        )

    if GITHUB_ACTIONS_ENABLED:
        if not (str(GITHUB_APP_ID).strip() or str(GITHUB_CLIENT_ID).strip()):
            errors.append(
                "GITHUB_ACTIONS_ENABLED=true requires GITHUB_APP_ID or GITHUB_CLIENT_ID."
            )

        if not str(GITHUB_PRIVATE_KEY_PATH).strip():
            errors.append("GITHUB_ACTIONS_ENABLED=true requires GITHUB_PRIVATE_KEY_PATH.")
        elif not Path(GITHUB_PRIVATE_KEY_PATH).expanduser().exists():
            errors.append(
                "GITHUB_PRIVATE_KEY_PATH must point to an existing private key file."
            )

        if not GITHUB_ALLOWED_REPOS and not GITHUB_ALLOWED_ORGS:
            errors.append(
                "GITHUB_ACTIONS_ENABLED=true requires GITHUB_ALLOWED_REPOS and/or GITHUB_ALLOWED_ORGS."
            )

        if not str(GITHUB_INSTALLATION_ID).strip():
            warnings.append(
                "GITHUB_INSTALLATION_ID is empty; installation lookup will be resolved dynamically."
            )

    if LOCAL_GIT_ACTIONS_ENABLED:
        if not LOCAL_GIT_ALLOWED_ROOTS:
            errors.append(
                "LOCAL_GIT_ACTIONS_ENABLED=true requires LOCAL_GIT_ALLOWED_ROOTS to be set."
            )

        if not LOCAL_GIT_DEFAULT_REPO_PATH.exists():
            errors.append("LOCAL_GIT_DEFAULT_REPO_PATH must point to an existing directory.")
        elif LOCAL_GIT_ALLOWED_ROOTS and not _is_within_any_root(
            LOCAL_GIT_DEFAULT_REPO_PATH,
            LOCAL_GIT_ALLOWED_ROOTS,
        ):
            errors.append(
                "LOCAL_GIT_DEFAULT_REPO_PATH must stay within LOCAL_GIT_ALLOWED_ROOTS."
            )

    if not DOCUMENTS_PATH.exists():
        warnings.append(
            f"Documents file is missing at {DOCUMENTS_PATH}. Run data ingestion before expecting retrieval to work."
        )

    if INCLUDE_TICKETS and not TICKETS_PATH.exists():
        warnings.append(
            f"Tickets file is missing at {TICKETS_PATH}. Ticket retrieval will stay unavailable until ingestion runs."
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def check_persistence_readiness() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "required": "redis" in {SESSION_STORE_BACKEND, ACTION_STORE_BACKEND, GRAPH_CHECKPOINTER_BACKEND},
        "ok": False,
        "session_store_backend": SESSION_STORE_BACKEND,
        "action_store_backend": ACTION_STORE_BACKEND,
        "graph_checkpointer_backend": GRAPH_CHECKPOINTER_BACKEND,
        "redis_url": (
            _redact_redis_url(REDIS_URL)
            if "redis" in {SESSION_STORE_BACKEND, ACTION_STORE_BACKEND, GRAPH_CHECKPOINTER_BACKEND}
            else ""
        ),
        "redis_key_prefix": REDIS_KEY_PREFIX,
    }

    try:
        session_store = get_session_state_store()
        session_health = (
            session_store.healthcheck()
            if hasattr(session_store, "healthcheck")
            else {"backend": SESSION_STORE_BACKEND, "ok": True}
        )
    except Exception as exc:
        payload["session_store"] = {"backend": SESSION_STORE_BACKEND, "ok": False}
        payload["error"] = _sanitize_error_message(exc)
        return payload

    try:
        action_store = get_action_store()
        action_health = (
            action_store.healthcheck()
            if hasattr(action_store, "healthcheck")
            else {"backend": ACTION_STORE_BACKEND, "ok": True}
        )
    except Exception as exc:
        payload["session_store"] = session_health
        payload["action_store"] = {"backend": ACTION_STORE_BACKEND, "ok": False}
        payload["error"] = _sanitize_error_message(exc)
        return payload

    try:
        graph_checkpointer = get_graph_checkpointer()
        graph_health = (
            graph_checkpointer.healthcheck()
            if hasattr(graph_checkpointer, "healthcheck")
            else {"backend": GRAPH_CHECKPOINTER_BACKEND, "ok": True}
        )
    except Exception as exc:
        payload["session_store"] = session_health
        payload["action_store"] = action_health
        payload["graph_checkpointer"] = {
            "backend": GRAPH_CHECKPOINTER_BACKEND,
            "ok": False,
        }
        payload["error"] = _sanitize_error_message(exc)
        return payload

    payload["session_store"] = session_health
    payload["action_store"] = action_health
    payload["graph_checkpointer"] = graph_health
    payload["ok"] = (
        bool(session_health.get("ok"))
        and bool(action_health.get("ok"))
        and bool(graph_health.get("ok"))
    )
    return payload


def _build_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )


def check_qdrant_readiness() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "required": True,
        "ok": False,
        "mode": QDRANT_MODE,
        "url": redact_url_credentials(QDRANT_URL),
        "collection": QDRANT_COLLECTION_NAME,
        "hybrid_enabled": USE_QDRANT_HYBRID,
        "server_reachable": False,
        "collection_exists": False,
    }

    try:
        client = _build_qdrant_client()
        collections = client.get_collections()
        collection_names = sorted(item.name for item in getattr(collections, "collections", []))

        payload["server_reachable"] = True
        payload["available_collections"] = collection_names[:10]

        if QDRANT_COLLECTION_NAME not in collection_names:
            payload["error"] = (
                f"Collection '{QDRANT_COLLECTION_NAME}' was not found in Qdrant."
            )
            if DOCUMENTS_PATH.exists():
                payload["hint"] = "Run `python scripts/dev.py ingest-data` to rebuild the Qdrant collection."
            else:
                payload["hint"] = (
                    "Start Qdrant, add source data under data_source/raw, then run "
                    "`python scripts/dev.py ingest-data`."
                )
            return payload

        collection = client.get_collection(QDRANT_COLLECTION_NAME)
        payload["collection_exists"] = True
        payload["points_count"] = getattr(collection, "points_count", None)
        payload["ok"] = True
        return payload
    except Exception as exc:
        payload["error"] = _sanitize_error_message(exc)
        payload["hint"] = (
            "Check that Qdrant is reachable at the configured URL and that ingestion has been run."
        )
        return payload


def check_processed_data() -> Dict[str, Any]:
    documents_present = DOCUMENTS_PATH.exists()
    tickets_present = TICKETS_PATH.exists()

    return {
        "required": False,
        "ok": documents_present and (tickets_present or not INCLUDE_TICKETS),
        "documents_path": str(DOCUMENTS_PATH),
        "documents_present": documents_present,
        "tickets_path": str(TICKETS_PATH),
        "tickets_enabled": INCLUDE_TICKETS,
        "tickets_present": tickets_present,
    }


def check_github_actions_readiness() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "required": False,
        "enabled": GITHUB_ACTIONS_ENABLED,
        "allowed_repo_count": len(GITHUB_ALLOWED_REPOS),
        "allowed_org_count": len(GITHUB_ALLOWED_ORGS),
        "installation_id_configured": bool(str(GITHUB_INSTALLATION_ID).strip()),
    }

    if not GITHUB_ACTIONS_ENABLED:
        payload["ok"] = True
        payload["status"] = "disabled"
        return payload

    auth_health = GitHubAppAuth().auth_health()
    payload["auth"] = auth_health
    payload["ok"] = bool(auth_health.get("configured")) and bool(auth_health.get("jwt_ready"))
    if not payload["ok"]:
        payload["error"] = auth_health.get("error") or "GitHub App authentication is not ready."

    return payload


def check_local_git_readiness() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "required": False,
        "enabled": LOCAL_GIT_ACTIONS_ENABLED,
        "default_repo_path": str(LOCAL_GIT_DEFAULT_REPO_PATH),
        "allowed_roots": [str(item) for item in LOCAL_GIT_ALLOWED_ROOTS],
    }

    if not LOCAL_GIT_ACTIONS_ENABLED:
        payload["ok"] = True
        payload["status"] = "disabled"
        return payload

    try:
        client = LocalGitClient()
        repo_root = client.resolve_repo_path()
        payload["ok"] = True
        payload["repo_root"] = str(repo_root)
        return payload
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = _sanitize_error_message(exc)
        return payload


def build_readiness_report() -> Dict[str, Any]:
    checks = {
        "persistence": check_persistence_readiness(),
        "qdrant": check_qdrant_readiness(),
        "processed_data": check_processed_data(),
        "github_actions": check_github_actions_readiness(),
        "local_git": check_local_git_readiness(),
    }

    ready = all(check["ok"] for check in checks.values() if check.get("required"))

    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "checks": checks,
    }
