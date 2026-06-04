---
policy_id: pol_security_escalation
title: Security Escalation Policy
policy_type: policy
owner_team: security_operations
effective_date: 2026-02-15
review_date: 2026-11-15
tags: [security, mfa, api-key]
summary: How to handle MFA resets, API key rotation issues, and suspected security impact.
---

# Security Escalation Policy

## Verification

Support must verify the requester before assisting with MFA reset, API key rotation, or privileged access changes.

## MFA Reset

MFA reset requires identity verification and an audit note. Do not bypass MFA for unverified users.

## API Key Rotation

For API key rotation failures, confirm the old key is retired, the new key is scoped correctly, and deployment environments reference the new alias.

## Escalation

Escalate to security operations when an access issue may expose customer data, block an admin from security controls, or indicate credential misuse.
