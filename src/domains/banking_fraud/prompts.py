from __future__ import annotations


def get_prompt_templates() -> dict[str, str]:
    return {
        "fraud_investigation": (
            "Use only retrieved synthetic banking evidence to explain fraud alerts, "
            "transactions, merchant risk, and recommended controls."
        ),
        "aml_review": (
            "Ground AML case summaries in linked alerts, transactions, source-of-funds needs, "
            "and synthetic policy guidance."
        ),
        "customer_safe_response": (
            "Draft customer-facing fraud messages without revealing detection thresholds or rules."
        ),
        "missing_information": (
            "If retrieved evidence does not include a fact, say what banking fraud evidence is missing."
        ),
    }


__all__ = ["get_prompt_templates"]
