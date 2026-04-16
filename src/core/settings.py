from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

from src.core.security import read_env, read_secret_env


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_set(value: str | None) -> set[str]:
    items: set[str] = set()
    for raw in str(value or "").split(","):
        normalized = raw.strip().lower()
        if normalized:
            items.add(normalized)
    return items


def _parse_csv_list(value: str | None) -> list[str]:
    items: list[str] = []
    for raw in str(value or "").split(","):
        normalized = raw.strip()
        if normalized:
            items.append(normalized)
    return items


def _parse_csv_paths(value: str | None) -> list[Path]:
    paths: list[Path] = []
    for raw in str(value or "").split(","):
        normalized = raw.strip()
        if normalized:
            paths.append(Path(normalized).expanduser().resolve())
    return paths


DATA_SOURCE_DIR = Path(read_env("DATA_SOURCE_DIR", ROOT_DIR / "data_source"))
PROCESSED_DIR = Path(read_env("PROCESSED_DIR", DATA_SOURCE_DIR / "processed"))

DOCUMENTS_PATH = Path(read_env("DOCUMENTS_PATH", PROCESSED_DIR / "documents.jsonl"))
TICKETS_PATH = Path(read_env("TICKETS_PATH", PROCESSED_DIR / "tickets.jsonl"))

COLLECTION_NAME = read_env("COLLECTION_NAME", "internal_support_copilot")

INCLUDE_TICKETS = _parse_bool(read_env("INCLUDE_TICKETS", "true"), default=True)

CHUNK_SIZE = int(read_env("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(read_env("CHUNK_OVERLAP", "150"))

RETRIEVAL_TOP_K = int(read_env("RETRIEVAL_TOP_K", "8"))
FINAL_TOP_K = int(read_env("FINAL_TOP_K", "4"))

# Cross-encoder rerank
USE_CROSS_ENCODER = _parse_bool(read_env("USE_CROSS_ENCODER", "true"), default=True)
CROSS_ENCODER_MODEL_NAME = read_env(
    "CROSS_ENCODER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)
CROSS_ENCODER_TOP_K = int(read_env("CROSS_ENCODER_TOP_K", "5"))
CROSS_ENCODER_BATCH_SIZE = int(read_env("CROSS_ENCODER_BATCH_SIZE", "16"))

QDRANT_MODE = read_env("QDRANT_MODE", "server")  # server | local
QDRANT_URL = read_env("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = read_secret_env("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = read_env("QDRANT_COLLECTION_NAME", "internal_support_copilot_qdrant")

QDRANT_VECTOR_NAME = read_env("QDRANT_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = read_env("QDRANT_SPARSE_VECTOR_NAME", "sparse")
QDRANT_PREFER_GRPC = _parse_bool(read_env("QDRANT_PREFER_GRPC", "true"), default=True)
USE_QDRANT_HYBRID = _parse_bool(read_env("USE_QDRANT_HYBRID", "true"), default=True)

API_CORS_ORIGINS = _parse_csv_list(
    read_env(
        "API_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501,http://127.0.0.1:8501",
    )
)

LOG_LEVEL = read_env("LOG_LEVEL", "INFO").strip().upper()
LOG_FORMAT = read_env("LOG_FORMAT", "json").strip().lower()
OBSERVABILITY_BACKEND = read_env("OBSERVABILITY_BACKEND", "memory").strip().lower()
OBSERVABILITY_TRACE_HISTORY_LIMIT = int(read_env("OBSERVABILITY_TRACE_HISTORY_LIMIT", "200"))
AUTH_ENABLED = _parse_bool(read_env("AUTH_ENABLED", "true"), default=True)
AUTH_USER_HEADER = read_env("AUTH_USER_HEADER", "X-User-ID").strip()
AUTH_ROLE_HEADER = read_env("AUTH_ROLE_HEADER", "X-User-Role").strip()
AUTH_DEFAULT_ROLE = read_env("AUTH_DEFAULT_ROLE", "viewer").strip().lower()
AUTH_ALLOW_ANONYMOUS_READS = _parse_bool(
    read_env("AUTH_ALLOW_ANONYMOUS_READS", "true"),
    default=True,
)
AUTH_REQUIRE_USER_ID_FOR_OPERATOR = _parse_bool(
    read_env("AUTH_REQUIRE_USER_ID_FOR_OPERATOR", "true"),
    default=True,
)

SESSION_HISTORY_MAX_MESSAGES = int(read_env("SESSION_HISTORY_MAX_MESSAGES", "6"))
SESSION_STORE_BACKEND = read_env("SESSION_STORE_BACKEND", "memory").strip().lower()
ACTION_STORE_BACKEND = read_env("ACTION_STORE_BACKEND", SESSION_STORE_BACKEND).strip().lower()
GRAPH_CHECKPOINTER_BACKEND = read_env(
    "GRAPH_CHECKPOINTER_BACKEND",
    SESSION_STORE_BACKEND,
).strip().lower()
REDIS_URL = read_secret_env("REDIS_URL", "redis://localhost:6379/0").strip()
REDIS_KEY_PREFIX = read_env("REDIS_KEY_PREFIX", "internal_support_copilot").strip()
SESSION_STATE_TTL_SECONDS = int(read_env("SESSION_STATE_TTL_SECONDS", "0"))
ACTION_STATE_TTL_SECONDS = int(read_env("ACTION_STATE_TTL_SECONDS", str(SESSION_STATE_TTL_SECONDS)))
GRAPH_CHECKPOINT_TTL_SECONDS = int(read_env("GRAPH_CHECKPOINT_TTL_SECONDS", "0"))

# GitHub App integration
GITHUB_ACTIONS_ENABLED = _parse_bool(read_env("GITHUB_ACTIONS_ENABLED", "false"), default=False)
GITHUB_API_BASE = read_env("GITHUB_API_BASE", "https://api.github.com").rstrip("/")
GITHUB_API_VERSION = read_env("GITHUB_API_VERSION", "2026-03-10")
GITHUB_HTTP_TIMEOUT = int(read_env("GITHUB_HTTP_TIMEOUT", "30"))

GITHUB_APP_ID = read_env("GITHUB_APP_ID", "").strip()
GITHUB_CLIENT_ID = read_env("GITHUB_CLIENT_ID", "").strip()
GITHUB_INSTALLATION_ID = read_env("GITHUB_INSTALLATION_ID", "").strip()
GITHUB_PRIVATE_KEY_PATH = read_secret_env("GITHUB_PRIVATE_KEY_PATH", "").strip()

GITHUB_ALLOWED_REPOS = _parse_csv_set(read_env("GITHUB_ALLOWED_REPOS", ""))
GITHUB_ALLOWED_ORGS = _parse_csv_set(read_env("GITHUB_ALLOWED_ORGS", ""))
GITHUB_TEST_REPO = read_env("GITHUB_TEST_REPO", "").strip()

GITHUB_REQUIRE_CONFIRM_FOR_WRITE = _parse_bool(
    read_env("GITHUB_REQUIRE_CONFIRM_FOR_WRITE", "true"),
    default=True,
)
GITHUB_REQUIRE_CONFIRM_FOR_WORKFLOW = _parse_bool(
    read_env("GITHUB_REQUIRE_CONFIRM_FOR_WORKFLOW", "true"),
    default=True,
)

LOCAL_GIT_ACTIONS_ENABLED = _parse_bool(read_env("LOCAL_GIT_ACTIONS_ENABLED", "false"), default=False)
LOCAL_GIT_DEFAULT_REPO_PATH = Path(read_env("LOCAL_GIT_DEFAULT_REPO_PATH", ROOT_DIR)).expanduser().resolve()
LOCAL_GIT_ALLOWED_ROOTS = _parse_csv_paths(read_env("LOCAL_GIT_ALLOWED_ROOTS", str(ROOT_DIR)))
LOCAL_GIT_REQUIRE_CONFIRM_FOR_WRITE = _parse_bool(
    read_env("LOCAL_GIT_REQUIRE_CONFIRM_FOR_WRITE", "true"),
    default=True,
)
