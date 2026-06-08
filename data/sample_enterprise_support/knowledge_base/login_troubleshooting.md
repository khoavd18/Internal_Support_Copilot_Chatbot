---
policy_id: pol_login_troubleshooting
title: Login Troubleshooting Guide
policy_type: troubleshooting
product_id: prod_identity
service_id: svc_auth
owner_team: identity_support
effective_date: 2026-02-01
review_date: 2026-11-01
tags: [login, sso, browser]
summary: Troubleshooting steps for SSO loops, stale sessions, and browser-specific login failures.
---

# Login Troubleshooting Guide

## SSO Login Loop

Check whether IdP metadata was recently rotated. Verify the certificate fingerprint, entity ID, callback URL, and tenant metadata cache.

## Browser Specific Failures

If login works in one browser but not another, ask the customer to clear session cookies and retry in a private window. Capture the browser version and timestamp.

## SCIM And Role Mapping

When SCIM groups sync but roles do not update, verify the group mapping rule and check whether the target role identifier is current.

## Escalation

Escalate to identity engineering when cache refresh fails, role propagation fails, or login is unavailable for many users.
