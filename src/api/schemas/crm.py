from __future__ import annotations

from pydantic import BaseModel, Field
from src.api.schemas.enterprise import EnterpriseContextItem


class CustomerSummaryRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, description="Synthetic customer_id")


class CustomerSummaryResponse(BaseModel):
    customer_id: str
    customer_name: str = ""
    account_id: str = ""
    account_name: str = ""
    summary: str
    tickets: list[EnterpriseContextItem] = Field(default_factory=list)
    risk_events: list[EnterpriseContextItem] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
