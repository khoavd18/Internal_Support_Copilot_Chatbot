from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


GuardrailAction = Literal["allow_answer", "fallback_to_search", "fallback_to_clarify"]


@dataclass
class GuardrailDecision:
    action: GuardrailAction
    reason: str
    metrics: Dict[str, Any]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics(search_result: Dict[str, Any]) -> Dict[str, Any]:
    sources = search_result.get("sources", []) or []
    stats = search_result.get("stats", {}) or {}

    scores = []
    for item in sources:
        score = _safe_float(item.get("rerank_score"))
        if score is not None:
            scores.append(score)

    retrieved_docs = int(stats.get("retrieved_docs", len(sources)) or len(sources))
    top1_score = scores[0] if scores else None
    top2_score = scores[1] if len(scores) >= 2 else None
    score_gap = None
    if top1_score is not None and top2_score is not None:
        score_gap = top1_score - top2_score

    return {
        "retrieved_docs": retrieved_docs,
        "scored_docs": len(scores),
        "top1_score": top1_score,
        "top2_score": top2_score,
        "score_gap": score_gap,
    }


def evaluate_answer_guardrails(
    search_result: Dict[str, Any],
    requested_top_k: int,
) -> GuardrailDecision:
    metrics = _extract_metrics(search_result)
    retrieved_docs = metrics["retrieved_docs"]
    top1_score = metrics["top1_score"]
    top2_score = metrics["top2_score"]
    score_gap = metrics["score_gap"]

    if retrieved_docs == 0:
        return GuardrailDecision(
            action="fallback_to_clarify",
            reason="Retriever không tìm được tài liệu liên quan.",
            metrics=metrics,
        )

    if retrieved_docs == 1:
        if top1_score is None:
            return GuardrailDecision(
                action="fallback_to_search",
                reason="Chỉ có 1 tài liệu và chưa có tín hiệu score đủ rõ, nên ưu tiên hiển thị source trước.",
                metrics=metrics,
            )
        if top1_score < 4.0:
            return GuardrailDecision(
                action="fallback_to_clarify",
                reason="Chỉ có 1 tài liệu và độ khớp còn thấp, chưa đủ chắc để trả lời tổng hợp.",
                metrics=metrics,
            )
        if top1_score < 7.0:
            return GuardrailDecision(
                action="fallback_to_search",
                reason="Chỉ có 1 tài liệu ở mức khớp trung bình, nên ưu tiên show nguồn trước khi trả lời mạnh tay.",
                metrics=metrics,
            )

    if top1_score is not None and top1_score < 2.5:
        return GuardrailDecision(
            action="fallback_to_clarify",
            reason="Tài liệu top đầu có score quá thấp.",
            metrics=metrics,
        )

    if requested_top_k >= 3 and retrieved_docs < 2:
        return GuardrailDecision(
            action="fallback_to_search",
            reason="Số tài liệu tốt quá ít so với top_k yêu cầu, nên ưu tiên show sources.",
            metrics=metrics,
        )

    if top1_score is not None and top2_score is not None:
        if top1_score < 5.0 and top2_score < 4.0:
            return GuardrailDecision(
                action="fallback_to_search",
                reason="Top 2 tài liệu đều chưa đủ mạnh, nên ưu tiên show sources để tránh hallucination.",
                metrics=metrics,
            )

        if score_gap is not None and score_gap < 0.35 and top1_score < 6.0:
            return GuardrailDecision(
                action="fallback_to_search",
                reason="Các tài liệu top đầu đang khá sít nhau và chưa đủ mạnh, nên chưa nên kết luận chắc chắn.",
                metrics=metrics,
            )

    return GuardrailDecision(
        action="allow_answer",
        reason="Retriever cho tín hiệu đủ ổn để trả lời tổng hợp.",
        metrics=metrics,
    )