# ADR 0012: the API inside the package, and a Compose stack built on synthetic data

Status: accepted (executing agent, M7, 2026-10-18). Adjusts the layout of ARCHITECTURE.md section 4.

## Context

ARCHITECTURE.md places the FastAPI application, its SQLAlchemy models and its Alembic migrations in
`services/api/`, next to a `docker-compose.yml` and a `Dockerfile.api` at the root. Since M6 the
review store (models and migrations) is shared by the CLI and the API, and the API reads the same
warehouse, ledger, registry and simulator files as every command.

## Decision

- The application lives in the package: `strata/api/app.py` (`create_app(settings, zone)`), served
  by `strata serve` or by uvicorn in the container. The review store stays in `strata/review/`. One
  typed, tested package serves the CLI, the tests and the container, so the API cannot drift from
  the code that produced what it serves. `services/api/` is not created.
- `Dockerfile.api` builds the public workspace (synthetic extract, warehouse, claims at quick
  effort, simulator inputs, models) into the image, so `docker compose up` serves at once; its entry
  point stores the alert queue in PostgreSQL and starts the API. `Dockerfile.mlflow` serves the MLflow
  UI over the runs the image logged. The dashboard of M8 is a static placeholder until then.
- Ports are Strata's defaults: API 8000, PostgreSQL 5432, dashboard 5173, MLflow 5000. Another
  project on the reference machine uses 8100, 5433, 6380, 3100, 9100 and 5174, and none of those
  is taken.
- The build context excludes `.env*`, `.strata/`, data folders and every warehouse or database file
  (`.dockerignore`), so nothing local or private can reach an image.

## Consequences

- The stack runs on synthetic data only. Serving real data means `strata --private serve` on the
  machine that holds the extract, with the review store in the private folder (ADR 0011).
- The image is large (the scientific stack and the models are inside); it is rebuilt when the code
  changes, and its claims and models are at quick effort, which they say themselves.
