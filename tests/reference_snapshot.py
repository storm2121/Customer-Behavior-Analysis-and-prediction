"""A snapshot of what the pipeline produces from the default synthetic extract.

``tests/fixtures/reference_snapshot.json`` was written by this module from the M1 to M4 pipeline,
before the source mappings of M4.5 existed. The mapping test rebuilds the warehouse through
``mappings/synthetic.yaml`` and compares: the same manifest (files and fingerprint), the same
data-quality report and the same number of rows in every staging, intermediate and mart model. It
is synthetic data only: counts, never rows.

Regenerate (only when the generator or a model changes on purpose)::

    uv run python -m tests.reference_snapshot
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from strata.config import Settings, Zone
from strata.ingest.load import load_manifest
from strata.warehouse import connect, read_relation, row_count

FIXTURE = Path(__file__).parent / "fixtures" / "reference_snapshot.json"
SCHEMAS = ("staging", "intermediate", "marts")


def _missing(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def snapshot(settings: Settings, zone: Zone = Zone.PUBLIC) -> dict[str, Any]:
    """Manifest, data-quality report and model row counts of ``settings``' warehouse."""
    workspace = settings.workspace(zone)
    manifest = load_manifest(workspace)
    con = connect(workspace.warehouse, read_only=True)
    try:
        report = read_relation(con, "raw.dq_report")
        tables = con.execute(
            "select table_schema, table_name from information_schema.tables "
            "where table_schema in ('staging', 'intermediate', 'marts') order by all"
        ).fetchall()
        counts = {f"{schema}.{table}": row_count(con, schema, table) for schema, table in tables}
    finally:
        con.close()
    rows = report.sort_values(["check_id", "extract"]).to_dict(orient="records")
    return {
        "fingerprint": manifest.fingerprint,
        "files": [item.model_dump(mode="json") for item in manifest.files],
        "dq_report": [
            {key: (None if _missing(value) else value) for key, value in row.items()}
            for row in rows
        ],
        "row_counts": counts,
    }


def write_fixture() -> Path:
    """Build the default synthetic warehouse in a temporary folder and store its snapshot."""
    from strata.synthetic.parametric import ParametricConfig, generate
    from strata.synthetic.writer import write_extract
    from tests.conftest import build_warehouse, public_settings

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        root = Path(folder)
        write_extract(generate(ParametricConfig()), root / "synth")
        settings = public_settings(root / "work")
        build_warehouse(settings, Zone.PUBLIC, root / "synth")
        payload = snapshot(settings)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return FIXTURE


if __name__ == "__main__":
    sys.stdout.write(f"wrote {write_fixture().as_posix()}\n")
