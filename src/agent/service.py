from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional

from src.agent.guardrails import evaluate_answer_guardrails
from src.agent.memory import append_turn, get_history
from src.agent.rewrite import is_follow_up_question, rewrite_with_history
from src.agent.router import RouteDecision, decide_route_for_mode
from src.agent.tools import search_knowledge_base
from src.core.logging_utils import bind_log_context
from src.pipeline import LocalRAGPipeline, build_pipeline, get_default_pipeline

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_default_agent_pipeline() -> LocalRAGPipeline:
    return get_default_pipeline()


def _build_clarify_answer() -> str:
    return (
        "The current question is not specific enough to retrieve the right evidence.\n\n"
        "Try asking in a more concrete format such as:\n"
        "- what task you are trying to complete\n"
        "- what error you are seeing\n"
        "- which system or platform is involved\n\n"
        "Examples:\n"
        "- How do I sign in with a passkey?\n"
        "- How do I push a local repository to GitHub over SSH?"
    )


def _build_guardrail_clarify_answer(reason: str) -> str:
    return (
        "I retrieved some relevant evidence, but the signal is still too weak to answer confidently.\n\n"
        f"Fallback reason: {reason}\n\n"
        "Please ask a more specific follow-up with the exact task, platform, or error message involved."
    )


def _format_retrieve_only_answer(result: Dict, fallback_reason: str | None = None) -> str:
    docs = result.get("documents", [])
    if not docs:
        return (
            "I could not find enough relevant evidence for this request.\n\n"
            "Please restate the question with more specific context."
        )

    lines: List[str] = []

    if fallback_reason:
        lines.append(f"I am returning sources only because: {fallback_reason}")
        lines.append("")

    lines.append("Here are the most relevant documents I found:")
    lines.append("")

    for item in docs:
        title = item.get("title", "Unknown")
        source = item.get("source", "unknown")
        path = item.get("path", "")
        preview = item.get("preview", "")

        lines.append(f"{item['index']}. {title} ({source})")
        if path:
            lines.append(f"   - path: {path}")
        if preview:
            lines.append(f"   - preview: {preview}")

    lines.append("")
    lines.append(
        "Switch to mode='answer' or ask a clearer how-to question if you want a synthesized final answer."
    )

    return "\n".join(lines)


def _build_base_stats(
    *,
    use_top_k: int,
    mode: str,
    decision: RouteDecision,
    session_id: Optional[str],
    question: str,
    effective_question: str,
    history_turns: int,
) -> Dict:
    return {
        "top_k_requested": use_top_k,
        "agent_mode": mode,
        "agent_route": decision.route,
        "session_id": session_id or "",
        "history_turns": history_turns,
        "used_history": effective_question != question,
        "original_question": question,
        "effective_question": effective_question,
    }


class InternalSupportAgent:
    def __init__(self, default_top_k: int = 4):
        self.default_top_k = default_top_k

    def _get_pipeline(self, top_k: int) -> LocalRAGPipeline:
        if top_k == self.default_top_k:
            return get_default_agent_pipeline()
        return build_pipeline(top_k=top_k, rebuild=False)

    def _resolve_decision(self, question: str, mode: str) -> RouteDecision:
        return decide_route_for_mode(question, mode)

    def ask(
        self,
        question: str,
        debug: bool = False,
        top_k: Optional[int] = None,
        mode: str = "auto",
        session_id: Optional[str] = None,
    ) -> Dict:
        question = (question or "").strip()
        use_top_k = top_k or self.default_top_k

        with bind_log_context(session_id=session_id, agent_name="internal_support_agent"):
            history = get_history(session_id)
            effective_question = question
            if is_follow_up_question(question, history):
                effective_question = rewrite_with_history(question, history)

            decision = self._resolve_decision(question=effective_question, mode=mode)
            tool_calls: List[Dict] = []

            logger.info(
                "Agent route decided",
                extra={
                    "event": "agent.route.decided",
                    "route": decision.route,
                    "mode": mode,
                    "reason": decision.reason,
                    "question_length": len(question),
                    "effective_question_length": len(effective_question),
                    "history_turns": len(history),
                },
            )

            base_stats = _build_base_stats(
                use_top_k=use_top_k,
                mode=mode,
                decision=decision,
                session_id=session_id,
                question=question,
                effective_question=effective_question,
                history_turns=len(history),
            )

            if decision.route == "clarify":
                answer = _build_clarify_answer()

                if session_id:
                    append_turn(session_id, "user", question)
                    append_turn(session_id, "assistant", answer)

                return {
                    "answer": answer,
                    "sources": [],
                    "stats": {
                        **base_stats,
                        "retrieved_docs": 0,
                        "used_fallback": True,
                    },
                    "debug": [],
                    "agent": {
                        "route": decision.route,
                        "reason": decision.reason,
                        "tool_calls": tool_calls,
                    },
                }

            if decision.route == "retrieve_only":
                tool_input = {
                    "query": effective_question,
                    "top_k": use_top_k,
                    "rebuild": False,
                }
                logger.info(
                    "Agent executing retrieval-only tool",
                    extra={
                        "event": "agent.tool.search.started",
                        "tool_name": "search_knowledge_base",
                        "top_k": use_top_k,
                        "question_length": len(effective_question),
                    },
                )
                try:
                    result = search_knowledge_base(**tool_input)
                    tool_calls.append(
                        {
                            "tool_name": "search_knowledge_base",
                            "tool_input": {
                                **tool_input,
                                "original_question": question,
                            },
                            "status": "ok",
                            "note": "Successfully retrieved related evidence.",
                        }
                    )

                    answer = _format_retrieve_only_answer(result)

                    if session_id:
                        append_turn(session_id, "user", question)
                        append_turn(session_id, "assistant", answer)

                    stats = dict(result.get("stats", {}))
                    stats.update(
                        {
                            **base_stats,
                            "used_fallback": False,
                        }
                    )

                    logger.info(
                        "Agent retrieval-only tool completed",
                        extra={
                            "event": "agent.tool.search.completed",
                            "retrieved_docs": result.get("stats", {}).get("retrieved_docs", 0),
                        },
                    )
                    return {
                        "answer": answer,
                        "sources": result.get("sources", []),
                        "stats": stats,
                        "debug": result.get("debug", []) if debug else [],
                        "agent": {
                            "route": decision.route,
                            "reason": decision.reason,
                            "tool_calls": tool_calls,
                        },
                    }
                except Exception:
                    logger.exception(
                        "Agent retrieval-only tool failed",
                        extra={
                            "event": "agent.tool.search.failed",
                            "tool_name": "search_knowledge_base",
                        },
                    )
                    tool_calls.append(
                        {
                            "tool_name": "search_knowledge_base",
                            "tool_input": {
                                **tool_input,
                                "original_question": question,
                            },
                            "status": "error",
                            "note": "Tool execution failed.",
                        }
                    )
                    return {
                        "answer": "I could not complete the search-only request right now.",
                        "sources": [],
                        "stats": {
                            **base_stats,
                            "retrieved_docs": 0,
                            "used_fallback": True,
                        },
                        "debug": [],
                        "agent": {
                            "route": decision.route,
                            "reason": decision.reason,
                            "tool_calls": tool_calls,
                        },
                    }

            search_tool_input = {
                "query": effective_question,
                "top_k": use_top_k,
                "rebuild": False,
            }
            answer_tool_input = {
                "question": effective_question,
                "top_k": use_top_k,
                "debug": debug,
            }

            logger.info(
                "Agent executing answer flow preflight retrieval",
                extra={
                    "event": "agent.answer.preflight.started",
                    "top_k": use_top_k,
                    "question_length": len(effective_question),
                },
            )
            try:
                preflight = search_knowledge_base(**search_tool_input)
                guardrail = evaluate_answer_guardrails(
                    search_result=preflight,
                    requested_top_k=use_top_k,
                )
                tool_calls.append(
                    {
                        "tool_name": "search_knowledge_base",
                        "tool_input": {
                            **search_tool_input,
                            "original_question": question,
                        },
                        "status": "ok",
                        "note": f"Preflight retrieval completed. guardrail={guardrail.action}",
                    }
                )

                logger.info(
                    "Agent guardrail evaluated",
                    extra={
                        "event": "agent.answer.guardrail",
                        "guardrail_action": guardrail.action,
                        "guardrail_reason": guardrail.reason,
                        "retrieved_docs": preflight.get("stats", {}).get("retrieved_docs", 0),
                    },
                )

                if guardrail.action == "fallback_to_clarify":
                    answer = _build_guardrail_clarify_answer(guardrail.reason)

                    if session_id:
                        append_turn(session_id, "user", question)
                        append_turn(session_id, "assistant", answer)

                    return {
                        "answer": answer,
                        "sources": preflight.get("sources", []),
                        "stats": {
                            **dict(preflight.get("stats", {})),
                            **base_stats,
                            "used_fallback": True,
                            "guardrail_action": guardrail.action,
                            "guardrail_reason": guardrail.reason,
                            **guardrail.metrics,
                        },
                        "debug": preflight.get("debug", []) if debug else [],
                        "agent": {
                            "route": "clarify",
                            "reason": f"Guardrail blocked answer generation: {guardrail.reason}",
                            "tool_calls": tool_calls,
                        },
                    }

                if guardrail.action == "fallback_to_search":
                    answer = _format_retrieve_only_answer(
                        preflight,
                        fallback_reason=guardrail.reason,
                    )

                    if session_id:
                        append_turn(session_id, "user", question)
                        append_turn(session_id, "assistant", answer)

                    return {
                        "answer": answer,
                        "sources": preflight.get("sources", []),
                        "stats": {
                            **dict(preflight.get("stats", {})),
                            **base_stats,
                            "used_fallback": True,
                            "guardrail_action": guardrail.action,
                            "guardrail_reason": guardrail.reason,
                            **guardrail.metrics,
                        },
                        "debug": preflight.get("debug", []) if debug else [],
                        "agent": {
                            "route": "retrieve_only",
                            "reason": f"Guardrail switched the response to search-only: {guardrail.reason}",
                            "tool_calls": tool_calls,
                        },
                    }

                pipeline = self._get_pipeline(use_top_k)
                result = pipeline.answer_from_documents(
                    question=effective_question,
                    documents=preflight.get("documents_raw", []),
                    debug=debug,
                )
                tool_calls.append(
                    {
                        "tool_name": "answer_from_documents",
                        "tool_input": {
                            **answer_tool_input,
                            "original_question": question,
                            "documents_count": len(preflight.get("documents_raw", [])),
                        },
                        "status": "ok",
                        "note": "Generated an answer from the preflight evidence set without retrieving twice.",
                    }
                )

                answer = result.get("answer", "")

                if session_id:
                    append_turn(session_id, "user", question)
                    append_turn(session_id, "assistant", answer)

                stats = dict(result.get("stats", {}))
                stats.update(
                    {
                        **base_stats,
                        "guardrail_action": guardrail.action,
                        "guardrail_reason": guardrail.reason,
                        **guardrail.metrics,
                    }
                )

                logger.info(
                    "Agent answer flow completed",
                    extra={
                        "event": "agent.answer.completed",
                        "retrieved_docs": stats.get("retrieved_docs", 0),
                        "used_fallback": stats.get("used_fallback", False),
                        "stage": stats.get("stage", ""),
                    },
                )
                return {
                    "answer": answer,
                    "sources": result.get("sources", []),
                    "stats": stats,
                    "debug": result.get("debug", []) if debug else [],
                    "agent": {
                        "route": decision.route,
                        "reason": decision.reason,
                        "tool_calls": tool_calls,
                    },
                }
            except Exception:
                logger.exception(
                    "Agent answer flow failed",
                    extra={"event": "agent.answer.failed"},
                )
                tool_calls.append(
                    {
                        "tool_name": "answer_from_knowledge_base",
                        "tool_input": {
                            **answer_tool_input,
                            "original_question": question,
                        },
                        "status": "error",
                        "note": "Answer flow failed.",
                    }
                )
                return {
                    "answer": "I could not complete the answer generation flow right now.",
                    "sources": [],
                    "stats": {
                        **base_stats,
                        "retrieved_docs": 0,
                        "used_fallback": True,
                    },
                    "debug": [],
                    "agent": {
                        "route": decision.route,
                        "reason": decision.reason,
                        "tool_calls": tool_calls,
                    },
                }


@lru_cache(maxsize=1)
def get_agent() -> InternalSupportAgent:
    return InternalSupportAgent(default_top_k=4)
