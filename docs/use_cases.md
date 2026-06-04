# Enterprise Support Use Cases

These demo use cases describe how the synthetic dataset under `data/sample_enterprise_support/` can exercise an Enterprise Support Intelligence Copilot. They are designed for portfolio demos, evaluation prompts, and future integration tests.

Use synthetic data only. Do not use private or real customer records.

## 1. Customer Summary

**User prompt**

```text
Summarize the current state of account acct_001. Include open tickets, risk level, affected products, and recommended next steps.
```

**Primary data sources**

- `crm/accounts.csv`
- `crm/customers.csv`
- `support/tickets.csv`
- `support/ticket_messages.csv`
- `support/ticket_resolutions.csv`
- `risk/risk_events.csv`

**Expected behavior**

The copilot should summarize account health, support tier, active or recent tickets, customer sentiment, risk events, and next actions. It should cite the ticket and risk event records used as evidence.

**Demo value**

Shows CRM intelligence grounded in support history instead of a generic customer summary.

## 2. Ticket Triage

**User prompt**

```text
Triage ticket tkt_001. What priority should it have, which team should own it, and what evidence supports that decision?
```

**Primary data sources**

- `support/tickets.csv`
- `support/ticket_messages.csv`
- `crm/accounts.csv`
- `crm/products.csv`
- `engineering/service_catalog.csv`
- `knowledge_base/sla_policy.md`
- `knowledge_base/security_escalation_policy.md`

**Expected behavior**

The copilot should inspect the ticket category, severity, support tier, customer impact, product, service owner, and relevant policy. It should recommend a priority or escalation path without inventing facts.

**Demo value**

Shows operational decision support with explainable routing and policy grounding.

## 3. Policy Lookup

**User prompt**

```text
What does the support policy say about enterprise SLA escalation for a P1 incident?
```

**Primary data sources**

- `knowledge_base/sla_policy.md`
- `knowledge_base/enterprise_support_policy.md`
- `knowledge_base/incident_response_policy.md`

**Expected behavior**

The copilot should retrieve the relevant policy sections, answer directly, and cite the specific policy documents. If the policy is ambiguous, it should say what is missing.

**Demo value**

Shows classic RAG over enterprise policy and runbook content.

## 4. Support Reply Generation

**User prompt**

```text
Draft a customer-safe reply for ticket tkt_001 using the latest messages, applicable policy, and known engineering status.
```

**Primary data sources**

- `support/tickets.csv`
- `support/ticket_messages.csv`
- `knowledge_base/api_timeout_runbook.md`
- `knowledge_base/incident_response_policy.md`
- `engineering/github_issues.jsonl`

**Expected behavior**

The copilot should produce a concise customer-facing response, avoid exposing internal notes, reference known status, and include a clear next step. It should not promise fixes or timelines unless the evidence supports them.

**Demo value**

Shows guarded response drafting from mixed support, policy, and engineering evidence.

## 5. Similar Issue Retrieval

**User prompt**

```text
Find previous tickets and engineering issues similar to this API timeout report for prod_api.
```

**Primary data sources**

- `support/tickets.csv`
- `support/ticket_messages.csv`
- `support/ticket_resolutions.csv`
- `engineering/github_issues.jsonl`
- `crm/products.csv`
- `engineering/service_catalog.csv`

**Expected behavior**

The copilot should retrieve similar tickets and related engineering issues, explain why each match is relevant, and identify any prior resolution or workaround.

**Demo value**

Shows semantic retrieval across support cases and engineering issue evidence.

## 6. Service Owner Lookup

**User prompt**

```text
Who owns the service behind the login product, and where should a support escalation go?
```

**Primary data sources**

- `crm/products.csv`
- `engineering/service_catalog.csv`
- `knowledge_base/login_troubleshooting.md`
- `knowledge_base/access_policy.md`

**Expected behavior**

The copilot should map the product to `service_id`, identify the engineering owner, support escalation team, relevant runbook, and any escalation channel metadata.

**Demo value**

Shows structured lookup and relationship reasoning, which is useful for future Knowledge Graph and GraphRAG demos.

## 7. SLA Escalation Check

**User prompt**

```text
Is ticket tkt_001 at risk of breaching SLA, and what should happen next?
```

**Primary data sources**

- `support/tickets.csv`
- `support/ticket_messages.csv`
- `crm/accounts.csv`
- `knowledge_base/sla_policy.md`
- `knowledge_base/enterprise_support_policy.md`
- `risk/risk_events.csv`

**Expected behavior**

The copilot should compare ticket priority, support tier, current status, response due time, resolution due time, and risk events. It should recommend escalation only when supported by policy and ticket evidence.

**Demo value**

Shows deadline-aware support automation and risk-aware triage.

## 8. Risk Or Anomaly Explanation

**User prompt**

```text
Explain why account acct_001 is marked high risk. Include tickets, incidents, sentiment, and engineering evidence.
```

**Primary data sources**

- `crm/accounts.csv`
- `support/tickets.csv`
- `support/ticket_messages.csv`
- `engineering/github_issues.jsonl`
- `risk/risk_events.csv`
- `knowledge_base/customer_risk_policy.md`

**Expected behavior**

The copilot should explain the risk score using source evidence, group signals by cause, identify whether the risk is account, ticket, service, or incident related, and recommend mitigation steps.

**Demo value**

Shows future anomaly and risk scoring in an explainable form, while keeping the current implementation grounded in simple synthetic records.

## Demo Quality Checklist

- Answers cite stable IDs such as `account_id`, `ticket_id`, `service_id`, `policy_id`, `issue_id`, and `risk_event_id`.
- Responses distinguish policy facts from inferred recommendations.
- Customer-facing drafts exclude internal-only ticket messages.
- Risk explanations include evidence and avoid unsupported causal claims.
- Similar issue retrieval returns both support and engineering evidence when available.
- SLA decisions reference the policy and the ticket deadlines.
