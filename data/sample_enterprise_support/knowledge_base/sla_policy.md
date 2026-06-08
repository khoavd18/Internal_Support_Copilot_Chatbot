---
policy_id: pol_sla
title: Enterprise SLA Policy
policy_type: policy
owner_team: customer_success_operations
effective_date: 2026-01-01
review_date: 2026-10-01
tags: [sla, escalation, enterprise]
summary: Response and resolution targets for enterprise support tiers.
---

# Enterprise SLA Policy

## Severity Levels

- `p1` means production is unavailable or a critical customer workflow is blocked.
- `p2` means a major workflow is degraded and no acceptable workaround exists.
- `p3` means a noncritical workflow is impaired.
- `p4` means a how-to request or low impact question.

## Enterprise Targets

Enterprise `p1` tickets require first response within 15 minutes and resolution or active mitigation within 4 hours. Enterprise `p2` tickets require first response within 1 hour and resolution or workaround within 12 hours.

## Escalation

Escalate a ticket when the SLA status is `at_risk` or `breached`. Escalation should include the ticket ID, account ID, affected product, affected service, customer impact, and the current owner.

## Customer Communication

For active `p1` or `sev1` incidents, send regular updates even when there is no material change. Do not promise a fix time unless engineering has confirmed it.
