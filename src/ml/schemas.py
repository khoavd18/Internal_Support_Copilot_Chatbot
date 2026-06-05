from __future__ import annotations

from typing import Any, Literal

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

RiskLevel = Literal["low", "medium", "high", "critical"]


class CustomerRiskFeatureRow(BaseModel):
    customer_id: str
    features: dict[str, float]
    related_ticket_ids: list[str] = Field(default_factory=list)
    related_risk_event_ids: list[str] = Field(default_factory=list)
    anchor_date: str = ""


class CustomerRiskScoreRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, description="Synthetic customer_id")


class CustomerRiskScoreResponse(BaseModel):
    customer_id: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    top_reasons: list[str] = Field(default_factory=list)
    related_events: list[dict[str, Any]] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
