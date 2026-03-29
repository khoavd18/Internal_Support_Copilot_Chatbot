from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


RouteName = Literal["answer_from_kb", "retrieve_only", "clarify"]


@dataclass
class RouteDecision:
    route: RouteName
    reason: str


def decide_route(question: str) -> RouteDecision:
    q = (question or "").strip()
    q_lower = q.lower()

    if not q:
        return RouteDecision(
            route="clarify",
            reason="Câu hỏi rỗng.",
        )

    normalized = re.sub(r"\s+", "", q)
    if len(normalized) < 8:
        return RouteDecision(
            route="clarify",
            reason="Câu hỏi quá ngắn nên chưa đủ rõ để truy xuất tài liệu chính xác.",
        )

    retrieve_only_signals = [
        "cho tôi source",
        "cho tôi nguồn",
        "liệt kê tài liệu",
        "liệt kê source",
        "show source",
        "show sources",
        "related docs",
        "tài liệu liên quan",
        "nguồn liên quan",
        "debug retrieval",
    ]

    if any(signal in q_lower for signal in retrieve_only_signals):
        return RouteDecision(
            route="retrieve_only",
            reason="Người dùng có vẻ muốn xem tài liệu/sources thay vì câu trả lời tổng hợp.",
        )

    return RouteDecision(
        route="answer_from_kb",
        reason="Mặc định dùng knowledge base để trả lời tổng hợp.",
    )