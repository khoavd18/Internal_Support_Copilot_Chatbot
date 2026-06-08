from __future__ import annotations

import logging

from fastapi import HTTPException
from src.data.enterprise_support_service import EnterpriseSupportDataError

logger = logging.getLogger(__name__)


def raise_enterprise_support_error(exc: Exception) -> None:
    if isinstance(exc, EnterpriseSupportDataError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.exception(
        "Enterprise support automation endpoint failed",
        extra={"event": "enterprise_support.endpoint.failed"},
    )
    raise HTTPException(
        status_code=500,
        detail="Failed to process enterprise support request.",
    ) from exc
