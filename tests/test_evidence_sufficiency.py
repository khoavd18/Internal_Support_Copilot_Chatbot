from __future__ import annotations

from src.rag.evidence_sufficiency import score_evidence_sufficiency


def _evidence(
    *,
    source_type: str,
    entity_id: str,
    text: str,
    created_at: str = "2026-06-01T00:00:00+00:00",
    **metadata,
) -> dict:
    return {
        "id": f"{source_type}:{entity_id}",
        "text": text,
        "source_type": source_type,
        "title": metadata.get("title", entity_id),
        "metadata": {
            "source_type": source_type,
            "entity_id": entity_id,
            "created_at": created_at,
            **metadata,
        },
        "context_source": "both",
    }


def test_score_evidence_sufficiency_high_with_ticket_policy_and_service_evidence() -> None:
    result = score_evidence_sufficiency(
        "Why should tkt_001 API timeout be escalated under SLA and which service owns it?",
        [
            _evidence(
                source_type="ticket",
                entity_id="tkt_001",
                ticket_id="tkt_001",
                text="Ticket ID: tkt_001\nSLA status: breached\nPriority: p1",
            ),
            _evidence(
                source_type="knowledge_base",
                entity_id="pol_sla",
                policy_id="pol_sla",
                text="Policy ID: pol_sla\nP1 tickets require escalation under the SLA policy.",
            ),
            _evidence(
                source_type="service",
                entity_id="svc_api_gateway",
                service_id="svc_api_gateway",
                text="Service ID: svc_api_gateway\nOwner team: Reliability Engineering",
            ),
        ],
    )

    assert result["level"] == "high"
    assert result["score"] >= 0.75
    assert result["missing_source_types"] == []
    assert any("Exact entity matches found: tkt_001" in reason for reason in result["reasons"])


def test_score_evidence_sufficiency_low_with_unrelated_evidence() -> None:
    result = score_evidence_sufficiency(
        "Why should tkt_001 API timeout be escalated under SLA?",
        [
            _evidence(
                source_type="customer",
                entity_id="cust_999",
                customer_id="cust_999",
                text="Customer ID: cust_999\nName: Synthetic unrelated customer",
            )
        ],
    )

    assert result["level"] == "low"
    assert result["score"] < 0.45
    assert "ticket" in result["missing_source_types"]
    assert "knowledge_base" in result["missing_source_types"]
    assert "service" in result["missing_source_types"]
    assert any("No exact entity IDs" in reason for reason in result["reasons"])


def test_score_evidence_sufficiency_detects_missing_source_types() -> None:
    result = score_evidence_sufficiency(
        "What SLA policy applies to tkt_001 and which service owns it?",
        [
            _evidence(
                source_type="ticket",
                entity_id="tkt_001",
                ticket_id="tkt_001",
                text="Ticket ID: tkt_001\nTitle: API timeout during batch sync",
            )
        ],
    )

    assert result["level"] == "medium"
    assert result["missing_source_types"] == ["knowledge_base", "service"]
    assert any("Missing critical source types" in reason for reason in result["reasons"])
