# ADR 0011: the review store, and where real alerts may live

Status: accepted (executing agent, M6, 2026-10-18). Narrows ARCHITECTURE.md sections 10 and 11.

## Context

The monitoring layer stores its runs, alerts, analyst feedback and annotations so that labels can
be recorded against a queue and its precision measured on them. ARCHITECTURE.md specifies SQLAlchemy
2 with Alembic migrations over PostgreSQL, with a SQLite fallback selected through `STRATA_PG_URL`.

An alert names a client. On the real extract that is a real identifier, which AGENT_RULES.md and
ADR 0008 allow only under `STRATA_PRIVATE_DIR`. A PostgreSQL server keeps its data wherever its
volume lives (in Docker Compose, a named volume managed by Docker), which the privacy boundary can
neither see nor check.

## Decision

- The schema is defined once (`strata/review/models.py`) and created by Alembic migrations
  (`strata/review/migrations`), run programmatically before every use, on PostgreSQL and SQLite
  alike.
- **Public zone**: `STRATA_PG_URL` when set (the PostgreSQL service of Docker Compose, synthetic
  data only), otherwise `.strata/review/review.sqlite`.
- **Private zone**: always `<STRATA_PRIVATE_DIR>/review/review.sqlite`, whatever `STRATA_PG_URL`
  says; the path passes the same private-output check as every other private artefact.
- The CLI shows client ids only for synthetic data; a private queue is read by alert number, and the
  ids stay in the private store.

## Consequences

- Real alerts can never reach a database server through configuration alone.
- Analysts working on real data use the private store (and, in M8, a dashboard served from the
  private machine), not the Compose stack.
- A schema change is a new Alembic revision; the tests migrate a fresh store up, down and up again.
