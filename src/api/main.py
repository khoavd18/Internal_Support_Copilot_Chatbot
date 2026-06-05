from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.agent.action_registry import (
    detect_action_request,
    execute_registered_action,
    get_action_permission_name,
    is_cancellation_message,
    is_confirmation_message,
)
from src.agent.actions import AgentActionError
from src.agent.chat_actions import maybe_handle_chat_action
from src.agent.graph.supervisor import get_supervisor_graph
from src.agent.memory import append_turn, get_pending_action
from src.agent.service import get_agent
from src.agent.session_utils import prepare_question_with_history
from src.core.auth import (
    AuthorizationError,
    build_authorization_summary,
    require_action_permission,
    resolve_request_auth_context,
)
from src.core.logging_utils import bind_log_context, configure_logging
from src.core.observability import observe_duration
from src.core.runtime_checks import build_readiness_report, validate_environment_settings
from src.core.schema import (
    AgentAskRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    CommitRequest,
    CreateIssueRequest,
    CreateRepoRequest,
    CustomerSummaryRequest,
    CustomerSummaryResponse,
    GraphRAGAskRequest,
    GraphRAGAskResponse,
    SlaCheckResponse,
    SuggestedReplyResponse,
    TicketAutomationRequest,
    TicketTriageResponse,
)
from src.core.security import sanitize_error_text
from src.core.settings import API_CORS_ORIGINS
from src.data.enterprise_support_service import (
    EnterpriseSupportDataError,
    build_customer_summary,
    check_ticket_sla,
    get_enterprise_support_dataset,
    suggest_ticket_reply,
    triage_ticket,
)
from src.ml.anomaly import RiskScoringError, explain_risk_score
from src.ml.schemas import CustomerRiskScoreRequest, CustomerRiskScoreResponse
from src.pipeline import build_pipeline, get_default_pipeline
from src.rag.graphrag import format_context_for_answer, retrieve_enterprise_context

configure_logging(force=True)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Internal Support Copilot API",
    version="1.2.0",
    description=(
        "Enterprise-oriented support copilot API with local RAG, "
        "single-agent routing, and LangGraph multi-agent orchestration."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(request.headers.get("X-Request-ID") or "").strip() or uuid4().hex
    request.state.request_id = request_id

    try:
        auth_context = resolve_request_auth_context(request)
    except AuthorizationError as exc:
        with bind_log_context(request_id=request_id):
            logger.warning(
                "Request rejected during authentication",
                extra={
                    "event": "auth.request.rejected",
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": exc.status_code,
                },
            )
        response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        response.headers["X-Request-ID"] = request_id
        return response

    request.state.user_id = auth_context.user_id
    request.state.user_role = auth_context.role

    with bind_log_context(request_id=request_id, user_id=auth_context.user_id):
        with observe_duration(
            "http.request",
            metric_name="http.server.request.duration_ms",
            metric_attributes={
                "method": request.method,
                "path": request.url.path,
            },
            span_attributes={
                "method": request.method,
                "path": request.url.path,
            },
        ) as observation:
            logger.info(
                "HTTP request started",
                extra={
                    "event": "http.request.started",
                    "method": request.method,
                    "path": request.url.path,
                    "auth_role": auth_context.role,
                    "auth_source": auth_context.auth_source,
                },
            )
            try:
                response = await call_next(request)
            except Exception as exc:
                observation.record_exception(exc)
                observation.set_metric_attribute("status_code", 500)
                observation.set_attribute("status_code", 500)
                observation.set_attribute("session_id", getattr(request.state, "session_id", ""))
                observation.set_attribute("action_id", getattr(request.state, "action_id", ""))
                duration_ms = observation.finish(status="error")
                logger.exception(
                    "Unhandled HTTP request failure",
                    extra={
                        "event": "http.request.failed",
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": duration_ms,
                    },
                )
                raise

            observation.set_metric_attribute("status_code", response.status_code)
            observation.set_attribute("status_code", response.status_code)
            observation.set_attribute("session_id", getattr(request.state, "session_id", ""))
            observation.set_attribute("action_id", getattr(request.state, "action_id", ""))
            duration_ms = observation.finish(status="ok")
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "HTTP request completed",
                extra={
                    "event": "http.request.completed",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "session_id": getattr(request.state, "session_id", ""),
                    "action_id": getattr(request.state, "action_id", ""),
                    "auth_role": auth_context.role,
                },
            )
            return response


def _run_startup_validation() -> Dict[str, Any]:
    validation = validate_environment_settings()

    if validation["warnings"]:
        logger.warning(
            "Startup configuration warnings: %s",
            " | ".join(validation["warnings"]),
        )

    if not validation["ok"]:
        logger.error(
            "Startup configuration validation failed: %s",
            " | ".join(validation["errors"]),
        )
        raise RuntimeError(
            "Startup configuration validation failed. Review the logged error details."
        )

    logger.info("Startup configuration validation passed.")
    return validation


def _build_health_payload(app_instance: FastAPI) -> Dict[str, Any]:
    from src.core.settings import GITHUB_ACTIONS_ENABLED, LOCAL_GIT_ACTIONS_ENABLED

    pipeline_cached = get_default_pipeline.cache_info().currsize > 0
    readiness = build_readiness_report()

    payload: Dict[str, Any] = {
        "status": "ok",
        "service": "internal-support-copilot-api",
        "startup_validation": getattr(
            app_instance.state,
            "startup_validation",
            validate_environment_settings(),
        ),
        "dependency_ready": readiness["ready"],
        "dependencies": readiness["checks"],
        "pipeline_cached": pipeline_cached,
        "default_top_k": 4,
        "llm_loaded": False,
        "vector_store_ready": readiness["checks"]["qdrant"]["ok"],
        "agent_ready": False,
        "github_write_actions_enabled": GITHUB_ACTIONS_ENABLED,
        "local_git_actions_enabled": LOCAL_GIT_ACTIONS_ENABLED,
        "cors_origins": API_CORS_ORIGINS,
        "authorization": build_authorization_summary(),
    }

    try:
        agent = get_agent()
        payload["agent_ready"] = agent is not None
    except Exception as exc:
        payload["agent_error"] = sanitize_error_text(exc, max_length=240)

    if pipeline_cached:
        try:
            pipeline = get_default_pipeline()
            payload["llm_loaded"] = pipeline.llm is not None
        except Exception as exc:
            payload["llm_error"] = sanitize_error_text(exc, max_length=240)

    return payload


def _handle_startup() -> None:
    app.state.startup_validation = _run_startup_validation()


# FastAPI 0.135+ removed app.add_event_handler(), but the router-level API
# remains available across the versions we support.
app.router.add_event_handler("startup", _handle_startup)


def _normalize_debug_payload(value: Any) -> list:
    return value if isinstance(value, list) else []


def _append_session_turns(session_id: str | None, question: str, answer: str) -> None:
    if not session_id:
        return

    append_turn(session_id, "user", question)
    append_turn(session_id, "assistant", answer)


def _resolve_requested_chat_action(question: str, session_id: str | None) -> str:
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        return ""

    pending_action = get_pending_action(session_id)
    if pending_action and (
        is_confirmation_message(cleaned_question) or is_cancellation_message(cleaned_question)
    ):
        return str(pending_action.get("action") or "").strip()

    intent = detect_action_request(cleaned_question)
    if intent is None:
        return ""
    return str(intent.action or "").strip()


def _enforce_action_permission(request: Request, action_name: str) -> None:
    try:
        actor = require_action_permission(request, action_name)
    except AuthorizationError as exc:
        logger.warning(
            "Action authorization denied",
            extra={
                "event": "auth.action.denied",
                "action_name": action_name,
                "path": request.url.path,
                "status_code": exc.status_code,
            },
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    logger.info(
        "Action authorization granted",
        extra={
            "event": "auth.action.allowed",
            "action_name": action_name,
            "path": request.url.path,
            "auth_role": actor.role,
        },
    )


def _execute_registered_action_request(
    *,
    request: Request,
    action_name: str,
    payload: Dict[str, Any],
    confirmed: bool,
    idempotency_key: str | None,
    log_fields: Dict[str, Any],
    failure_detail: str,
) -> AgentResponse:
    permission_name = get_action_permission_name(action_name)
    _enforce_action_permission(request, permission_name)

    with bind_log_context(agent_name="write_action"):
        logger.info(
            "Write action endpoint invoked",
            extra={
                "event": f"action.{action_name}.requested",
                "action_name": action_name,
                "confirmed": confirmed,
                **log_fields,
            },
        )
        try:
            result = execute_registered_action(
                action_name,
                payload=payload,
                confirmed=confirmed,
                idempotency_key=idempotency_key,
            )
            request.state.action_id = str(result.get("stats", {}).get("action_id") or "")
            return AgentResponse(**result)
        except AgentActionError as exc:
            safe_detail = sanitize_error_text(exc, max_length=240)
            logger.warning(
                "Write action rejected",
                extra={
                    "event": f"action.{action_name}.rejected",
                    "action_name": action_name,
                    "detail": safe_detail,
                },
            )
            raise HTTPException(status_code=400, detail=safe_detail) from exc
        except Exception:
            logger.exception(
                "Write action endpoint failed",
                extra={
                    "event": f"action.{action_name}.failed",
                    "action_name": action_name,
                },
            )
            raise HTTPException(status_code=500, detail=failure_detail)


def _raise_enterprise_support_error(exc: Exception) -> None:
    if isinstance(exc, EnterpriseSupportDataError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.exception(
        "Enterprise support automation endpoint failed",
        extra={"event": "enterprise_support.endpoint.failed"},
    )
    raise HTTPException(
        status_code=500,
        detail="Failed to process enterprise support request.",
    ) from exc


def _build_graphrag_placeholder_answer(question: str, context: Dict[str, Any]) -> str:
    merged_count = len(context.get("merged_context") or [])
    vector_count = len(context.get("vector_evidence") or [])
    graph_count = len(context.get("graph_evidence") or [])

    if merged_count == 0:
        return (
            "No enterprise support context was found for this question. "
            "This endpoint does not call an LLM; it returns fused retrieval evidence for inspection."
        )

    return (
        "GraphRAG evidence was retrieved for the enterprise support question. "
        f"Question: {question.strip()} "
        f"Fused evidence items: {merged_count} "
        f"(vector={vector_count}, graph={graph_count}). "
        "This first version returns a placeholder answer plus grounded evidence; "
        "LLM answer generation has not been wired into this endpoint."
    )


@app.get("/health", tags=["system"])
def health_check(request: Request) -> Dict[str, Any]:
    logger.info(
        "Health endpoint requested",
        extra={"event": "health.requested"},
    )
    return _build_health_payload(request.app)


@app.get("/ready", tags=["system"])
def readiness_check() -> JSONResponse:
    payload = build_readiness_report()
    status_code = 200 if payload["ready"] else 503
    logger.info(
        "Readiness endpoint requested",
        extra={
            "event": "health.readiness",
            "ready": payload["ready"],
            "status_code": status_code,
        },
    )
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/crm/customer-summary", response_model=CustomerSummaryResponse, tags=["crm"])
def crm_customer_summary(payload: CustomerSummaryRequest) -> CustomerSummaryResponse:
    try:
        return CustomerSummaryResponse(**build_customer_summary(payload.customer_id))
    except Exception as exc:
        _raise_enterprise_support_error(exc)


@app.post("/support/ticket-triage", response_model=TicketTriageResponse, tags=["support"])
def support_ticket_triage(payload: TicketAutomationRequest) -> TicketTriageResponse:
    try:
        return TicketTriageResponse(**triage_ticket(payload.ticket_id))
    except Exception as exc:
        _raise_enterprise_support_error(exc)


@app.post("/support/suggest-reply", response_model=SuggestedReplyResponse, tags=["support"])
def support_suggest_reply(payload: TicketAutomationRequest) -> SuggestedReplyResponse:
    try:
        return SuggestedReplyResponse(**suggest_ticket_reply(payload.ticket_id))
    except Exception as exc:
        _raise_enterprise_support_error(exc)


@app.post("/support/sla-check", response_model=SlaCheckResponse, tags=["support"])
def support_sla_check(payload: TicketAutomationRequest) -> SlaCheckResponse:
    try:
        return SlaCheckResponse(**check_ticket_sla(payload.ticket_id))
    except Exception as exc:
        _raise_enterprise_support_error(exc)


@app.post("/risk/customer-score", response_model=CustomerRiskScoreResponse, tags=["risk"])
def risk_customer_score(payload: CustomerRiskScoreRequest) -> CustomerRiskScoreResponse:
    try:
        dataset = get_enterprise_support_dataset()
        return CustomerRiskScoreResponse(**explain_risk_score(payload.customer_id, dataset))
    except RiskScoringError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Customer risk scoring endpoint failed",
            extra={"event": "risk.customer_score.failed"},
        )
        raise HTTPException(status_code=500, detail="Customer risk scoring failed.") from exc


@app.post("/enterprise/ask", response_model=GraphRAGAskResponse, tags=["enterprise"])
def enterprise_ask(payload: GraphRAGAskRequest) -> GraphRAGAskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        context = retrieve_enterprise_context(
            question,
            top_k=payload.top_k,
            graph_depth=payload.graph_depth,
        )
        return GraphRAGAskResponse(
            answer=_build_graphrag_placeholder_answer(question, context),
            vector_evidence=context.get("vector_evidence", []),
            graph_evidence=context.get("graph_evidence", []),
            merged_context=context.get("merged_context", []),
            citations=context.get("citations", []),
            metadata={
                "formatted_context": format_context_for_answer(context),
                "query": question,
                "mode": "graphrag_placeholder",
            },
            stats=context.get("stats", {}),
        )
    except Exception:
        logger.exception(
            "Enterprise GraphRAG endpoint failed",
            extra={"event": "enterprise.graphrag.failed"},
        )
        raise HTTPException(status_code=500, detail="Enterprise GraphRAG request failed.")


@app.post("/ask", response_model=AskResponse, tags=["chat"])
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    request.state.session_id = payload.session_id or ""
    with bind_log_context(session_id=payload.session_id):
        logger.info(
            "Plain ask request received",
            extra={
                "event": "chat.ask",
                "top_k": payload.top_k or 4,
                "debug_requested": payload.debug,
                "question_length": len(question),
            },
        )
        try:
            pipeline = (
                build_pipeline(top_k=payload.top_k, rebuild=False)
                if payload.top_k is not None
                else get_default_pipeline()
            )
            result = pipeline.ask(question, debug=payload.debug)
            return AskResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                stats=result.get("stats", {}),
                debug=_normalize_debug_payload(result.get("debug", [])) if payload.debug else [],
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception(
                "Plain ask request failed",
                extra={"event": "chat.ask.failed"},
            )
            raise HTTPException(status_code=500, detail="Failed to process question.")


@app.post("/agent/ask", response_model=AgentResponse, tags=["chat"])
def ask_agent(payload: AgentAskRequest, request: Request) -> AgentResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    request.state.session_id = payload.session_id or ""
    with bind_log_context(session_id=payload.session_id, agent_name="internal_support_agent"):
        logger.info(
            "Single-agent request received",
            extra={
                "event": "agent.ask",
                "mode": payload.mode,
                "top_k": payload.top_k or 4,
                "confirmed": payload.confirmed,
                "question_length": len(question),
            },
        )
        try:
            requested_action = _resolve_requested_chat_action(question, payload.session_id)
            if requested_action:
                _enforce_action_permission(request, get_action_permission_name(requested_action))

            action_result = maybe_handle_chat_action(
                question=question,
                confirmed=payload.confirmed,
                session_id=payload.session_id,
                backend_mode="agent",
                idempotency_key=payload.idempotency_key,
            )
            if action_result is not None:
                _append_session_turns(payload.session_id, question, action_result.get("answer", ""))
                return AgentResponse(**action_result)

            agent = get_agent()
            result = agent.ask(
                question=question,
                debug=payload.debug,
                top_k=payload.top_k,
                mode=payload.mode,
                session_id=payload.session_id,
            )
            return AgentResponse(**result)
        except HTTPException:
            raise
        except Exception:
            logger.exception(
                "Single-agent request failed",
                extra={"event": "agent.ask.failed"},
            )
            raise HTTPException(status_code=500, detail="Failed to process agent request.")


@app.post("/multi-agent/ask", response_model=AgentResponse, tags=["chat"])
def ask_multi_agent(payload: AgentAskRequest, request: Request) -> AgentResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    request.state.session_id = payload.session_id or ""
    with bind_log_context(session_id=payload.session_id, agent_name="supervisor"):
        logger.info(
            "Multi-agent request received",
            extra={
                "event": "multi_agent.ask",
                "mode": payload.mode,
                "top_k": payload.top_k or 4,
                "confirmed": payload.confirmed,
                "question_length": len(question),
            },
        )
        try:
            requested_action = _resolve_requested_chat_action(question, payload.session_id)
            if requested_action:
                _enforce_action_permission(request, get_action_permission_name(requested_action))

            action_result = maybe_handle_chat_action(
                question=question,
                confirmed=payload.confirmed,
                session_id=payload.session_id,
                backend_mode="multi_agent",
                idempotency_key=payload.idempotency_key,
            )
            if action_result is not None:
                _append_session_turns(payload.session_id, question, action_result.get("answer", ""))
                return AgentResponse(**action_result)

            session_ctx = prepare_question_with_history(
                question=question,
                session_id=payload.session_id,
            )

            graph = get_supervisor_graph()
            thread_id = payload.session_id or "default-thread"
            result = graph.invoke(
                {
                    "question": session_ctx["question"],
                    "effective_question": session_ctx["effective_question"],
                    "history": session_ctx["history"],
                    "mode": payload.mode,
                    "debug": payload.debug,
                    "top_k": payload.top_k or 4,
                    "session_id": payload.session_id or "",
                },
                config={"configurable": {"thread_id": thread_id}},
            )

            answer = result.get("answer", "")
            _append_session_turns(payload.session_id, question, answer)

            fallback_agent = {
                "route": "clarify",
                "reason": "No agent metadata returned.",
                "tool_calls": [],
            }
            normalized_debug = _normalize_debug_payload(result.get("debug", [])) if payload.debug else []

            stats = dict(result.get("stats", {}))
            stats["debug_requested"] = payload.debug
            stats["original_question"] = session_ctx["question"]
            stats["effective_question"] = session_ctx["effective_question"]
            stats["used_history"] = session_ctx["used_history"]
            stats["history_turns"] = session_ctx["history_turns"]

            return AgentResponse(
                answer=answer,
                sources=result.get("sources", []) or result.get("merged_sources", []),
                stats=stats,
                debug=normalized_debug,
                agent=result.get("agent", fallback_agent),
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception(
                "Multi-agent request failed",
                extra={"event": "multi_agent.ask.failed"},
            )
            raise HTTPException(status_code=500, detail="Multi-agent request failed.")


@app.post("/multi-agent/actions/create-repo", response_model=AgentResponse, tags=["actions"])
def create_repo(payload: CreateRepoRequest, request: Request) -> AgentResponse:
    return _execute_registered_action_request(
        request=request,
        action_name="create_repo",
        payload={
            "org": payload.org,
            "name": payload.name,
            "description": payload.description,
            "private": payload.private,
            "auto_init": payload.auto_init,
        },
        confirmed=payload.confirmed,
        idempotency_key=payload.idempotency_key,
        log_fields={
            "org": payload.org,
            "repo_name": payload.name,
        },
        failure_detail="Repository creation failed.",
    )


@app.post("/multi-agent/actions/create-issue", response_model=AgentResponse, tags=["actions"])
def create_issue(payload: CreateIssueRequest, request: Request) -> AgentResponse:
    return _execute_registered_action_request(
        request=request,
        action_name="create_issue",
        payload={
            "repo_full_name": payload.repo_full_name,
            "title": payload.title,
            "body": payload.body,
            "labels": payload.labels,
            "assignees": payload.assignees,
        },
        confirmed=payload.confirmed,
        idempotency_key=payload.idempotency_key,
        log_fields={
            "repo": payload.repo_full_name,
            "title_length": len(payload.title),
            "labels_count": len(payload.labels),
            "assignees_count": len(payload.assignees),
        },
        failure_detail="Issue creation failed.",
    )


@app.post("/multi-agent/actions/commit", response_model=AgentResponse, tags=["actions"])
def commit_changes(payload: CommitRequest, request: Request) -> AgentResponse:
    return _execute_registered_action_request(
        request=request,
        action_name="commit",
        payload={
            "message": payload.message,
            "repo_path": payload.repo_path,
            "paths": payload.paths,
            "stage_all": payload.stage_all,
            "include_untracked": payload.include_untracked,
        },
        confirmed=payload.confirmed,
        idempotency_key=payload.idempotency_key,
        log_fields={
            "repo_path": payload.repo_path or "",
            "paths_count": len(payload.paths),
            "stage_all": payload.stage_all,
            "include_untracked": payload.include_untracked,
        },
        failure_detail="Commit action failed.",
    )
