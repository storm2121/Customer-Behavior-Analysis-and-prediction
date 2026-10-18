# Architecture decision records

Each record states its status, context, decision and consequences. A deviation from
`ARCHITECTURE.md` needs a record here before the code that implements it.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-duckdb-dbt-warehouse.md) | DuckDB with dbt as the analytical core | accepted |
| [0002](0002-evidence-ledger.md) | An evidence ledger governs every published number | accepted |
| [0003](0003-time-based-validation.md) | Time-based validation for every client-level outcome | accepted |
| [0004](0004-privacy-boundary.md) | A structural privacy boundary with a public generator and a private twin | accepted |
| [0005](0005-airflow-orchestration.md) | Airflow for orchestration, running only in Docker Compose | accepted |
| [0006](0006-dashboard-stack.md) | React 19 with TypeScript for the dashboard | accepted |
| [0007](0007-bayesian-backend.md) | PyMC for the Bayesian calendar model | accepted |
| [0008](0008-private-zone-outside-the-working-tree.md) | Private runs write nothing inside the working tree | accepted |
| [0009](0009-calendar-model-terms.md) | Calendar model terms for a single year of data | accepted |
| [0010](0010-canonical-layer.md) | The canonical raw layer, its landing tables and its provenance | accepted |
