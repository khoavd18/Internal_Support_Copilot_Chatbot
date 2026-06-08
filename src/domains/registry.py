from __future__ import annotations

from src.domains.banking_fraud import BankingFraudAdapter
from src.domains.base import DomainAdapter
from src.domains.enterprise_support import EnterpriseSupportAdapter

_ADAPTERS: dict[str, DomainAdapter] = {
    "banking_fraud": BankingFraudAdapter(),
    "enterprise_support": EnterpriseSupportAdapter(),
}


def get_domain_adapter(domain_name: str) -> DomainAdapter:
    normalized = str(domain_name or "").strip().lower().replace("-", "_")
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unknown domain '{domain_name}'. Available domains: {available}") from exc


def list_domain_names() -> list[str]:
    return sorted(_ADAPTERS)


__all__ = ["get_domain_adapter", "list_domain_names"]
