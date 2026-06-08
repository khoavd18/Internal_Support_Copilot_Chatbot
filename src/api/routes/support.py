from __future__ import annotations

from fastapi import APIRouter
from src.api.routes._errors import raise_enterprise_support_error
from src.api.schemas.support import (
    SlaCheckResponse,
    SuggestedReplyResponse,
    TicketAutomationRequest,
    TicketTriageResponse,
)
from src.data.enterprise_support_service import (
    check_ticket_sla,
    suggest_ticket_reply,
    triage_ticket,
)

router = APIRouter()


@router.post("/support/ticket-triage", response_model=TicketTriageResponse, tags=["support"])
def support_ticket_triage(payload: TicketAutomationRequest) -> TicketTriageResponse:
    try:
        return TicketTriageResponse(**triage_ticket(payload.ticket_id))
    except Exception as exc:
        raise_enterprise_support_error(exc)


@router.post("/support/suggest-reply", response_model=SuggestedReplyResponse, tags=["support"])
def support_suggest_reply(payload: TicketAutomationRequest) -> SuggestedReplyResponse:
    try:
        return SuggestedReplyResponse(**suggest_ticket_reply(payload.ticket_id))
    except Exception as exc:
        raise_enterprise_support_error(exc)


@router.post("/support/sla-check", response_model=SlaCheckResponse, tags=["support"])
def support_sla_check(payload: TicketAutomationRequest) -> SlaCheckResponse:
    try:
        return SlaCheckResponse(**check_ticket_sla(payload.ticket_id))
    except Exception as exc:
        raise_enterprise_support_error(exc)
