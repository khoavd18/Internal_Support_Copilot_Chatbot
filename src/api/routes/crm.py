from __future__ import annotations

from fastapi import APIRouter
from src.api.routes._errors import raise_enterprise_support_error
from src.api.schemas.crm import CustomerSummaryRequest, CustomerSummaryResponse
from src.data.enterprise_support_service import build_customer_summary

router = APIRouter()


@router.post("/crm/customer-summary", response_model=CustomerSummaryResponse, tags=["crm"])
def crm_customer_summary(payload: CustomerSummaryRequest) -> CustomerSummaryResponse:
    try:
        return CustomerSummaryResponse(**build_customer_summary(payload.customer_id))
    except Exception as exc:
        raise_enterprise_support_error(exc)
