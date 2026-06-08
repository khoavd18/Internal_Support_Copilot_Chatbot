from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from src.api.schemas.risk import CustomerRiskScoreRequest, CustomerRiskScoreResponse
from src.data.enterprise_support_service import get_enterprise_support_dataset
from src.ml.anomaly import RiskScoringError, explain_risk_score

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/risk/customer-score", response_model=CustomerRiskScoreResponse, tags=["risk"])
def risk_customer_score(payload: CustomerRiskScoreRequest) -> CustomerRiskScoreResponse:
    try:
        dataset = get_enterprise_support_dataset()
        return CustomerRiskScoreResponse(
            **explain_risk_score(payload.customer_id, dataset, mode=payload.mode)
        )
    except RiskScoringError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Customer risk scoring endpoint failed",
            extra={"event": "risk.customer_score.failed"},
        )
        raise HTTPException(status_code=500, detail="Customer risk scoring failed.") from exc
