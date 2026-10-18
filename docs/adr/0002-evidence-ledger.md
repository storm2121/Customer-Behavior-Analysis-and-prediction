# ADR 0002: an evidence ledger governs every published number

Status: accepted (architect, 2026-10-18). Expand during M3.

## Context

Numbers in READMEs and CVs drift from the code that produced them (the detector metrics in a sibling
project disagreed between the CV and the README). Hand-typed figures also lose their sample size and
uncertainty.

## Decision

Every number in documentation, dashboard copy and reports is produced by a registered claim function
returning value, n, bootstrap CI, method, caveats and a data fingerprint. Documents are templates
rendered from the claims JSON. CI fails on drift or on an unregistered number inside governed sections.
Public documents may render only allow-listed claims.

## Consequences

Writing docs requires registering claims first, which is slower and far more honest. The dashboard
and the report share one source. Private claims never leave the machine.
