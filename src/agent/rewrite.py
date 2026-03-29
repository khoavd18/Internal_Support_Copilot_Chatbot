from __future__ import annotations

import re
from typing import Dict, List

FOLLOW_UP_PREFIXES = (
    "thế còn",
    "còn",
    "ý là",
    "ý tôi là",
    "tiếp tục",
    "còn github",
    "còn gitlab",
    "trên github",
    "trên gitlab",
    "vậy còn",
)

FOLLOW_UP_MARKERS = (
    "nó",
    "đó",
    "cái đó",
    "phần đó",
    "cách đó",
    "ý đó",
    "phần trên",
    "ở trên",
    "bên trên",
)

COMPARISON_PATTERN = re.compile(
    r"^(thế\s+còn|vậy\s+còn|còn)\s+(.+?)\s+thì\s+sao\??$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_lower(text: str) -> str:
    return _normalize(text).lower()


def _last_user_message(history: List[Dict[str, str]]) -> str:
    for item in reversed(history):
        if item.get("role") == "user" and item.get("content"):
            return item["content"].strip()
    return ""


def is_follow_up_question(question: str, history: List[Dict[str, str]]) -> bool:
    if not history:
        return False

    q = _normalize_lower(question)
    if not q:
        return False

    if len(q) <= 18:
        return True

    if q.startswith(FOLLOW_UP_PREFIXES):
        return True

    if any(marker in q for marker in FOLLOW_UP_MARKERS):
        return True

    if q.endswith("thì sao?") or q.endswith("được không?"):
        return True

    return False


def _rewrite_comparison(question: str, previous_user_question: str) -> str | None:
    """
    Ví dụ:
    - Previous: Làm thế nào để push project lên GitHub bằng SSH?
    - Current: Còn HTTPS thì sao?
    => Làm thế nào để push project lên GitHub bằng HTTPS thay vì SSH?
    """
    q = _normalize(question)
    prev = _normalize(previous_user_question)

    match = COMPARISON_PATTERN.match(q)
    if not match:
        return None

    new_topic = match.group(2).strip()
    if not new_topic:
        return None

    prev_lower = prev.lower()
    new_topic_lower = new_topic.lower()

    # Trường hợp thay thế "SSH" bằng "HTTPS"
    if " bằng ssh" in prev_lower and new_topic_lower == "https":
        return re.sub(
            r"\bbằng SSH\b",
            "bằng HTTPS",
            prev,
            flags=re.IGNORECASE,
        ).rstrip(" ?.") + " thay vì SSH?"

    if " bằng https" in prev_lower and new_topic_lower == "ssh":
        return re.sub(
            r"\bbằng HTTPS\b",
            "bằng SSH",
            prev,
            flags=re.IGNORECASE,
        ).rstrip(" ?.") + " thay vì HTTPS?"

    # Trường hợp chung: giữ nguyên ngữ cảnh và thay follow-up thành câu độc lập
    return f"{prev.rstrip(' ?.')} — trường hợp {new_topic} thì làm như thế nào?"


def _rewrite_pronoun_followup(question: str, previous_user_question: str) -> str | None:
    """
    Ví dụ:
    - Previous: Passkey là gì?
    - Current: Cách bật nó?
    => Trong ngữ cảnh câu hỏi 'Passkey là gì?', hãy trả lời: Cách bật nó?
    """
    q_lower = _normalize_lower(question)
    if not any(marker in q_lower for marker in FOLLOW_UP_MARKERS):
        return None

    prev = _normalize(previous_user_question)
    q = _normalize(question)
    return f"Trong ngữ cảnh '{prev}', hãy trả lời câu hỏi sau một cách cụ thể: {q}"


def rewrite_with_history(question: str, history: List[Dict[str, str]]) -> str:
    question = _normalize(question)
    if not question or not history:
        return question

    previous_user_question = _last_user_message(history)
    if not previous_user_question:
        return question

    if _normalize_lower(previous_user_question) == _normalize_lower(question):
        return question

    if not is_follow_up_question(question, history):
        return question

    rewritten = _rewrite_comparison(question, previous_user_question)
    if rewritten:
        return rewritten

    rewritten = _rewrite_pronoun_followup(question, previous_user_question)
    if rewritten:
        return rewritten

    return f"{_normalize(previous_user_question)}. Câu hỏi tiếp theo liên quan: {question}"