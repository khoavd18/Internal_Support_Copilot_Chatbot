from __future__ import annotations


def get_prompt_templates() -> dict[str, str]:
    return {
        "grounded_answer": (
            "Answer using only retrieved enterprise support evidence. "
            "Cite source metadata for customer, ticket, policy, service, GitHub issue, and risk claims."
        ),
        "missing_information": (
            "If the retrieved evidence is insufficient, state what information is missing "
            "instead of guessing."
        ),
        "support_reply": (
            "Draft a customer-safe support response grounded in ticket context and approved policies."
        ),
        "risk_explanation": (
            "Explain customer risk using ticket signals, risk events, service context, and policy evidence."
        ),
    }


__all__ = ["get_prompt_templates"]
