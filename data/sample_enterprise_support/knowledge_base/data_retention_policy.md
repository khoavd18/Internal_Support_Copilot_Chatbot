---
policy_id: pol_data_retention
title: Data Retention And Export Policy
policy_type: policy
product_id: prod_data_export
service_id: svc_data_export
owner_team: data_support
effective_date: 2026-04-01
review_date: 2027-01-01
tags: [data-export, retention, audit]
summary: Retention, export availability, and audit support guidance for synthetic support data.
---

# Data Retention And Export Policy

## Export Availability

Generated export files are available for the documented retention window. Customers should create a new export if the prior file expired.

## Scheduled Exports

Admins can schedule weekly exports for audit workflows. Scheduled exports should include configured custom fields when the schema flag is enabled.

## Large Exports

For very large exports, support may recommend smaller time windows while engineering investigates worker performance.

## Data Safety

Sample datasets must remain synthetic. Do not include private customer data, credentials, or production support transcripts in export examples.
