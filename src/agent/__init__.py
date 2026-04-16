from __future__ import annotations

"""Agent package.

Giữ __init__ nhẹ để import các module con (ví dụ src.agent.guardrails)
không bị kéo theo toàn bộ service layer và dependency nặng.
"""

__all__ = ["get_agent"]


def get_agent():
    from src.agent.service import get_agent as _get_agent

    return _get_agent()