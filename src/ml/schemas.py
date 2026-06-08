from __future__ import annotations

from pydantic import BaseModel, Field

FEATURE_NAMES: tuple[str, ...] = (
    "ticket_count_7d",
    "critical_ticket_count_30d",
    "escalation_count_30d",
    "failed_login_count_7d",
    "api_timeout_count_7d",
    "refund_request_count_30d",
    "negative_signal_count_30d",
)


class CustomerRiskFeatureRow(BaseModel):
    customer_id: str
    features: dict[str, float]
    related_ticket_ids: list[str] = Field(default_factory=list)
    related_risk_event_ids: list[str] = Field(default_factory=list)
    anchor_date: str = ""
