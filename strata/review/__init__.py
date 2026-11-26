"""The review store: alert runs, alerts, analyst feedback and annotations.

SQLAlchemy models over PostgreSQL (``STRATA_PG_URL``, in the public zone, as Docker Compose runs it)
or a SQLite file in the workspace; Alembic migrations create and evolve the schema.
"""
