from __future__ import annotations

from typing import Dict, List, Optional

from src.agent.memory import get_history
from src.agent.rewrite import is_follow_up_question, rewrite_with_history


def prepare_question_with_history(
    question: str,
    session_id: Optional[str] = None,
) -> Dict:
    question = (question or "").strip()
    history = get_history(session_id)
    effective_question = question

    if is_follow_up_question(question, history):
        effective_question = rewrite_with_history(question, history)

    return {
        "question": question,
        "effective_question": effective_question,
        "history": history,
        "history_turns": len(history),
        "used_history": effective_question != question,
    }