from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from src.agent.guardrails import evaluate_answer_guardrails
from src.agent.memory import append_turn, get_history
from src.agent.rewrite import is_follow_up_question, rewrite_with_history
from src.agent.router import RouteDecision, decide_route
from src.agent.tools import answer_from_knowledge_base, search_knowledge_base
from src.pipeline import LocalRAGPipeline, build_pipeline, get_default_pipeline


@lru_cache(maxsize=1)
def get_default_agent_pipeline() -> LocalRAGPipeline:
    return get_default_pipeline()


def _build_clarify_answer() -> str:
    return (
        "Câu hỏi hiện chưa đủ rõ để tôi truy xuất đúng tài liệu.\n\n"
        "Bạn nên hỏi cụ thể hơn theo dạng:\n"
        "- cách làm gì\n"
        "- lỗi gì\n"
        "- thao tác trên hệ thống nào\n\n"
        "Ví dụ:\n"
        "- Làm thế nào để đăng nhập bằng passkey?\n"
        "- Cách push local repo lên GitHub bằng SSH là gì?"
    )


def _build_guardrail_clarify_answer(reason: str) -> str:
    return (
        "Tôi đã retrieve được một ít tài liệu, nhưng tín hiệu hiện chưa đủ mạnh để trả lời chắc chắn.\n\n"
        f"Lý do fallback: {reason}\n\n"
        "Bạn nên hỏi cụ thể hơn theo đúng thao tác hoặc nền tảng cần làm.\n"
        "Ví dụ: 'Cách đăng nhập GitHub bằng passkey' hoặc 'Cách push local repo lên GitHub bằng SSH'."
    )


def _format_retrieve_only_answer(result: Dict, fallback_reason: str | None = None) -> str:
    docs = result.get("documents", [])
    if not docs:
        return (
            "Tôi chưa tìm được tài liệu đủ liên quan cho câu hỏi này.\n\n"
            "Bạn hãy hỏi lại cụ thể hơn để tôi retrieve chính xác hơn."
        )

    lines: List[str] = []

    if fallback_reason:
        lines.append(f"Tôi chưa trả lời tổng hợp ngay vì: {fallback_reason}")
        lines.append("")

    lines.append("Tôi đã tìm được các tài liệu liên quan nhất:")
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
        "Bạn có thể chuyển sang mode='answer' hoặc hỏi lại theo kiểu how-to để tôi tổng hợp thành câu trả lời hoàn chỉnh."
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
        if mode == "auto":
            return decide_route(question)
        if mode == "answer":
            return RouteDecision(
                route="answer_from_kb",
                reason="Mode được ép thành answer.",
            )
        return RouteDecision(
            route="retrieve_only",
            reason="Mode được ép thành search.",
        )

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

        history = get_history(session_id)
        effective_question = question
        if is_follow_up_question(question, history):
            effective_question = rewrite_with_history(question, history)

        decision = self._resolve_decision(question=effective_question, mode=mode)
        tool_calls: List[Dict] = []

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
                        "note": "Retrieve tài liệu liên quan thành công.",
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

            except Exception as exc:
                tool_calls.append(
                    {
                        "tool_name": "search_knowledge_base",
                        "tool_input": {
                            **tool_input,
                            "original_question": question,
                        },
                        "status": "error",
                        "note": str(exc),
                    }
                )
                return {
                    "answer": f"Agent gọi tool search_knowledge_base bị lỗi: {exc}",
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
                    "note": f"Preflight retrieval trước khi answer. guardrail={guardrail.action}",
                }
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
                        "reason": f"Guardrail chặn answer: {guardrail.reason}",
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
                        "reason": f"Guardrail chuyển sang search-only: {guardrail.reason}",
                        "tool_calls": tool_calls,
                    },
                }

            pipeline = self._get_pipeline(use_top_k)
            result = answer_from_knowledge_base(
                question=effective_question,
                top_k=use_top_k,
                debug=debug,
                pipeline=pipeline,
            )
            tool_calls.append(
                {
                    "tool_name": "answer_from_knowledge_base",
                    "tool_input": {
                        **answer_tool_input,
                        "original_question": question,
                    },
                    "status": "ok",
                    "note": "Trả lời từ knowledge base thành công sau khi qua guardrails.",
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

        except Exception as exc:
            tool_calls.append(
                {
                    "tool_name": "answer_from_knowledge_base",
                    "tool_input": {
                        **answer_tool_input,
                        "original_question": question,
                    },
                    "status": "error",
                    "note": str(exc),
                }
            )
            return {
                "answer": f"Agent gọi tool answer_from_knowledge_base bị lỗi: {exc}",
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