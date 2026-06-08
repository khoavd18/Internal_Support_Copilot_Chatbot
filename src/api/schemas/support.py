from __future__ import annotations

from pydantic import BaseModel, Field
from src.api.schemas.enterprise import EnterpriseContextItem


class TicketAutomationRequest(BaseModel):
    ticket_id: str = Field(..., min_length=1, description="Synthetic ticket_id")


class TicketTriageResponse(BaseModel):
    ticket_id: str
    current_priority: str
    recommended_priority: str
    current_status: str
    recommended_status: str
    escalation_required: bool
    reasoning: list[str] = Field(default_factory=list)
    context: list[EnterpriseContextItem] = Field(default_factory=list)


class SuggestedReplyResponse(BaseModel):
    ticket_id: str
    draft_reply: str
    used_policy_ids: list[str] = Field(default_factory=list)
    evidence: list[EnterpriseContextItem] = Field(default_factory=list)


class SlaCheckResponse(BaseModel):
    ticket_id: str
    priority: str
    status: str
    sla_status: str
    first_response_due_at: str = ""
    resolution_due_at: str = ""
    escalation_required: bool
    recommendation: str
    reasoning: list[str] = Field(default_factory=list)
    policy: EnterpriseContextItem | None = None
