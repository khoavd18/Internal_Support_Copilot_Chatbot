from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high", "critical"]
RiskScoringMode = Literal["heuristic", "ml"]


class CustomerRiskScoreRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, description="Synthetic customer_id")
    mode: RiskScoringMode = Field(
        default="heuristic",
        description="Use deterministic heuristic scoring or optional ML anomaly scoring.",
    )


class CustomerRiskScoreResponse(BaseModel):
    customer_id: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    top_reasons: list[str] = Field(default_factory=list)
    related_events: list[dict[str, Any]] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
