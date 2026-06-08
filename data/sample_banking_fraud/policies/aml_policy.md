# Synthetic AML Policy

## Purpose

This synthetic policy describes how analysts should handle potential anti-money-laundering risk in demo banking data. It is not legal advice and must not be used for real customers.

## High-Risk Signals

- Large wires to high-risk jurisdictions.
- Crypto purchases followed by offshore transfers.
- Cash-out behavior after a held or suspicious transfer.
- Missing invoice, source-of-funds, or beneficial ownership evidence.

## Analyst Actions

Hold suspicious outbound wires when policy conditions are met. Open an AML case, request source-of-funds evidence, and document the rationale. Escalate p1 cases to AML investigations when multiple high-risk signals appear in one session.
