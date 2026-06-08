---
policy_id: pol_api_timeout
title: API Timeout Runbook
policy_type: runbook
product_id: prod_api
service_id: svc_api_gateway
owner_team: api_support
effective_date: 2026-02-01
review_date: 2026-11-01
tags: [api, timeout, latency]
summary: Steps for diagnosing API timeouts and regional latency.
---

# API Timeout Runbook

## Initial Checks

Confirm the endpoint, tenant, region, request volume, error code, and time window. Compare customer reports with gateway latency, worker queue depth, and rate limit metrics.

## Common Causes

Common causes include gateway shard saturation, cache warmup failures, regional routing issues, queue backlog, and client retries without backoff.

## Workarounds

For noncritical exports, recommend smaller time windows or retry with exponential backoff. For production outages, escalate according to the incident response policy.

## Escalation

Escalate to `eng_api_platform` when timeout rates exceed baseline, a `p1` account is blocked, or the issue affects payment or authentication flows.
