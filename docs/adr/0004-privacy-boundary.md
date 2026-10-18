# ADR 0004: a structural privacy boundary with a public parametric generator and a private twin

Status: accepted (architect, 2026-10-18). Expand during M0 and M9.

## Context

The extract is an anonymised sample of a bank's clients provided for an academic challenge. The
repository is public and linked from CVs. The project must run end to end for any reader without the
data, and the real findings must stay private except for a few harmless aggregates.

## Decision

Real data is reachable only through `KASHF_DATA_DIR`; private outputs go to `KASHF_PRIVATE_DIR`; a
pre-commit guard blocks extract signatures, large CSVs and identifier-like tokens; an allowlist
decides which claims may be rendered publicly. The public demo uses a hand-specified parametric
generator that reproduces the schema and quirks without touching real rows. A separate private
generator fits a Gaussian copula on real marginals with Laplace noise and reports fidelity and
membership-inference risk; its output never enters the repository and is used only for local experiments.

## Consequences

Two generators to maintain. The public demo's numbers are illustrative by construction, which the
docs state on every page that shows them. The private report is the artefact for interviews.
