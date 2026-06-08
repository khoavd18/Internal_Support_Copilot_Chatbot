# Enterprise Support Knowledge Graph Schema

This project includes a lightweight in-memory Knowledge Graph for the synthetic enterprise support dataset. It is intentionally local, deterministic, and dependency-free. It does not use Neo4j yet and does not change the existing RAG generation flow.

## Node Types

| Node Type | ID Example | Description |
| --- | --- | --- |
| `Customer` | `Customer:cust_001` | Synthetic customer/contact record. |
| `Account` | `Account:acct_001` | Synthetic company or organization. |
| `Product` | `Product:prod_api` | Product or plan surface. |
| `Ticket` | `Ticket:tkt_001` | Support case. |
| `TicketMessage` | `TicketMessage:msg_001` | Public or internal support conversation message. |
| `Policy` | `Policy:pol_sla` | Knowledge base policy, runbook, or troubleshooting document. |
| `Service` | `Service:svc_api_gateway` | Engineering-owned service from the service catalog. |
| `GitHubIssue` | `GitHubIssue:gh_001` | Synthetic GitHub-like engineering issue evidence. |
| `RiskEvent` | `RiskEvent:risk_001` | Customer, account, ticket, product, or service risk signal. |
| `Team` | `Team:eng_api_platform` | Synthetic support, engineering, security, incident, or customer success team. |

## Edge Types

| Edge Type | Example | Description |
| --- | --- | --- |
| `HAS_ACCOUNT` | `Customer:cust_001 -> Account:acct_001` | Connects a customer to their account. |
| `CREATED_TICKET` | `Customer:cust_001 -> Ticket:tkt_001` | Connects a customer to a ticket they opened. |
| `HAS_MESSAGE` | `Ticket:tkt_001 -> TicketMessage:msg_001` | Connects a ticket to its conversation messages. |
| `HAS_RESOLUTION` | `Ticket:tkt_003 -> Policy:pol_refund` | Represents a ticket resolution as edge metadata linked to its policy evidence. |
| `MENTIONS_PRODUCT` | `Ticket:tkt_001 -> Product:prod_api` | Connects tickets, policies, issues, or risk events to products. |
| `REFERENCES_POLICY` | `Ticket:tkt_003 -> Policy:pol_refund` | Connects services or resolved tickets to policy/runbook evidence. |
| `AFFECTS_SERVICE` | `Ticket:tkt_001 -> Service:svc_api_gateway` | Connects tickets, products, issues, or risks to services. |
| `OWNED_BY_TEAM` | `Service:svc_api_gateway -> Team:eng_api_platform` | Connects owned entities to support or engineering teams. |
| `RELATED_TO_ISSUE` | `Ticket:tkt_001 -> GitHubIssue:gh_001` | Connects support tickets to engineering issue evidence. |
| `HAS_RISK_EVENT` | `Ticket:tkt_001 -> RiskEvent:risk_001` | Connects customers, accounts, tickets, products, or services to risk events. |

Ticket resolutions are not modeled as standalone nodes in the first version because the requested node set does not include `TicketResolution`. Resolution fields such as `resolution_id`, `resolution_type`, `summary`, and `resolved_at` are stored on `HAS_RESOLUTION` edge metadata.

## Example Traversal

Starting from an API timeout ticket:

```text
Ticket:tkt_001
  -[CREATED_TICKET incoming]- Customer:cust_001
  -[HAS_ACCOUNT]- Account:acct_001
  -[MENTIONS_PRODUCT]- Product:prod_api
  -[AFFECTS_SERVICE]- Service:svc_api_gateway
  -[OWNED_BY_TEAM]- Team:eng_api_platform
  -[RELATED_TO_ISSUE]- GitHubIssue:gh_001
  -[HAS_RISK_EVENT]- RiskEvent:risk_001
```

In code:

```python
from pathlib import Path

from src.data.enterprise_support_loader import load_enterprise_support_dataset
from src.kg.builder import build_graph_from_enterprise_support_dataset
from src.kg.store import get_neighbors
from src.kg.retriever import retrieve_graph_context

data_dir = Path("data/sample_enterprise_support")
dataset = load_enterprise_support_dataset(data_dir)
graph = build_graph_from_enterprise_support_dataset(dataset)

neighbors = get_neighbors("Ticket:tkt_001", depth=2)
context = retrieve_graph_context("api timeout escalation", depth=2)
```

## GraphRAG Support

The in-memory graph gives the RAG layer structured context that vector search alone may miss:

- Ticket retrieval can expand to customer, account, product, service owner, policy, issue, and risk evidence.
- Policy lookup can expand to affected products and services.
- Customer summaries can combine CRM, support, risk, and engineering evidence.
- SLA and escalation answers can include both ticket fields and policy/runbook context.
- Future GraphRAG can merge vector-ranked chunks with graph neighborhoods before generation.

This layer is intentionally read-only and in-memory. A future Neo4j or graph database integration should preserve the stable node IDs and edge types defined here.
