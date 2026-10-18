# ADR 0002: an evidence ledger governs every published number

Status: accepted (architect, 2026-10-18). Filled in M0; the implementation details are expanded in M3.

## Context

Numbers in READMEs and CVs drift from the code that produced them (the detector metrics in a sibling
project disagreed between the CV and the README). Hand-typed figures also lose their sample size and
uncertainty.

A second problem is specific to this project: most real findings must stay private, while a few
harmless aggregates are worth publishing. Without a single registry, the line between the two is a
matter of memory.

## Decision

Every number in documentation, dashboard copy and reports is produced by a registered claim function
returning value, n, bootstrap CI, method, caveats and a data fingerprint. Documents are templates
rendered from the claims JSON. CI fails on drift or on an unregistered number inside governed sections.
Public documents may render only allow-listed claims.

The allowlist (`kashf/ledger/allowlist.py`) exists from M0 because the privacy guard enforces it on
every commit, before the ledger itself is built.

## Alternatives considered

- **Numbers in notebooks, copied into documents.** The status quo that produced the drift.
- **Generated documents without a check.** Rendering is not enough: a hand edit to a rendered file,
  or a hand-typed number next to a placeholder, must fail CI.
- **Publishing everything with differential privacy.** Considered for the twin (ADR 0004); for
  findings, a short allowlist of coarse aggregates is simpler to reason about and to audit.

## Consequences

Writing docs requires registering claims first, which is slower and far more honest. The dashboard
and the report share one source. Private claims never leave the machine.
